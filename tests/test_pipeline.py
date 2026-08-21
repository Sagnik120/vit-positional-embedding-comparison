"""
tests/test_pipeline.py

Consolidated diagnostic / pipeline test suite for this project. This is the
SINGLE canonical location for all diagnostic checks -- `scripts/run_diagnostics.py`
is kept only as a thin backwards-compatible wrapper that calls into this file.

Deep sanity-check suite. Run this BEFORE any real training to catch shape
mismatches, device issues, or broken imports early, without burning time on
a full training run.

Checks performed, in order (each prints PASS/FAIL, script exits non-zero on
any failure so it's CI-friendly):

  1. Environment / imports          - torch, torchvision, einops importable
  2. Device availability            - reports cpu/mps/cuda
  3. Config sanity                  - patch_size divides image_size, dims consistent
  4. Dataset loader                 - CIFAR-10 downloads/loads, batch shapes correct
  5. Original ViT forward pass      - single batch, output shape == (B, num_classes)
  6. Original ViT backward pass     - loss.backward() runs, gradients are non-None/non-nan
  7. Modified ViT forward pass      - same as (5) for RoPE variant
  8. Modified ViT backward pass     - same as (6) for RoPE variant
  9. Output shape equality          - both variants produce identical output shape
 10. Parameter count sanity         - both variants have comparable (not wildly
                                      different) parameter counts, since only the
                                      PE mechanism should differ (RoPE adds ~0 params)
 11. RoPE-specific numerical check  - rotating q,k does not change their norm
                                      (rotation must be norm-preserving)
 12. One-batch overfit smoke test   - both models can drive loss down sharply on a
                                      SINGLE repeated batch in a few steps (verifies
                                      gradients actually flow and optimizer works,
                                      catches silent no-op bugs)
 13. Full mini end-to-end pipeline  - 1 tiny epoch (few batches) of train+val loop
                                      exactly as train.py would run it, both variants

Usage:
    python tests/test_pipeline.py
    python tests/test_pipeline.py --skip-download   # if data already downloaded

    (or, equivalently, via the thin wrapper: python scripts/run_diagnostics.py)
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = []


def check(name, fn):
    print(f"\n--- {name} ---")
    try:
        fn()
        print(f"[{PASS}] {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"[{FAIL}] {name}\n{traceback.format_exc()}")
        results.append((name, False, str(e)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # ---- 1. imports ----
    def check_imports():
        import torch, torchvision, einops  # noqa
        print(f"torch={torch.__version__}, torchvision={torchvision.__version__}, "
              f"einops={einops.__version__}")
    check("1. Environment / imports", check_imports)

    # ---- 2. device ----
    device_holder = {}

    def check_device():
        from common.utils import get_device
        device = get_device("auto")
        device_holder["device"] = device
        print(f"selected device: {device}")
        if device.type == "mps":
            assert torch.backends.mps.is_available()
    check("2. Device availability", check_device)
    device = device_holder.get("device", torch.device("cpu"))

    # ---- 3. config sanity ----
    def check_config():
        from common.config import ModelConfig
        cfg = ModelConfig()
        assert cfg.image_size % cfg.patch_size == 0, "patch_size must divide image_size"
        grid = cfg.image_size // cfg.patch_size
        num_patches = grid * grid
        print(f"image_size={cfg.image_size}, patch_size={cfg.patch_size}, "
              f"grid={grid}x{grid}, num_patches={num_patches}, dim={cfg.dim}, "
              f"heads={cfg.heads}, dim_head={cfg.dim_head}")
        assert cfg.dim_head % 4 == 0, "dim_head must be divisible by 4 for 2D-RoPE"
    check("3. Config sanity", check_config)

    # ---- 4. dataset loader ----
    loaders_holder = {}

    def check_dataset():
        from common.config import DataConfig, TrainConfig
        from common.dataset import get_dataloaders
        data_cfg = DataConfig()
        train_cfg = TrainConfig()
        train_cfg.batch_size = args.batch_size
        if args.skip_download:
            train_cfg.num_workers = 0
        train_loader, val_loader, test_loader = get_dataloaders(data_cfg, train_cfg)
        xb, yb = next(iter(train_loader))
        print(f"train batches={len(train_loader)} val batches={len(val_loader)} "
              f"test batches={len(test_loader)}")
        print(f"batch x shape={tuple(xb.shape)} y shape={tuple(yb.shape)} "
              f"y dtype={yb.dtype} y range=({yb.min().item()},{yb.max().item()})")
        assert xb.shape[1:] == (3, 32, 32), f"unexpected image shape {xb.shape}"
        assert yb.min() >= 0 and yb.max() <= 9, "labels out of expected CIFAR-10 range"
        loaders_holder["train"] = train_loader
        loaders_holder["val"] = val_loader
    check("4. Dataset loader", check_dataset)

    # ---- 5/6. original ViT forward+backward ----
    model_holder = {}

    def check_original_forward():
        from common.config import ModelConfig
        from common.train import build_model
        cfg = ModelConfig()
        model = build_model("original", cfg).to(device)
        x = torch.randn(args.batch_size, 3, cfg.image_size, cfg.image_size, device=device)
        out = model(x)
        print(f"output shape: {tuple(out.shape)}")
        assert out.shape == (args.batch_size, cfg.num_classes), \
            f"expected ({args.batch_size},{cfg.num_classes}), got {tuple(out.shape)}"
        model_holder["original"] = model
    check("5. Original ViT forward pass", check_original_forward)

    def check_original_backward():
        cfg_model = model_holder["original"]
        x = torch.randn(args.batch_size, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (args.batch_size,), device=device)
        out = cfg_model(x)
        loss = nn.functional.cross_entropy(out, y)
        cfg_model.zero_grad()
        loss.backward()
        n_none = 0
        n_nan = 0
        for name, p in cfg_model.named_parameters():
            if p.requires_grad:
                if p.grad is None:
                    n_none += 1
                elif torch.isnan(p.grad).any():
                    n_nan += 1
        print(f"loss={loss.item():.4f}, params_with_none_grad={n_none}, "
              f"params_with_nan_grad={n_nan}")
        assert n_none == 0, "some parameters received no gradient"
        assert n_nan == 0, "NaN gradients detected"
    check("6. Original ViT backward pass", check_original_backward)

    # ---- 7/8. modified ViT forward+backward ----
    def check_modified_forward():
        from common.config import ModelConfig
        from common.train import build_model
        cfg = ModelConfig()
        model = build_model("modified", cfg).to(device)
        x = torch.randn(args.batch_size, 3, cfg.image_size, cfg.image_size, device=device)
        out = model(x)
        print(f"output shape: {tuple(out.shape)}")
        assert out.shape == (args.batch_size, cfg.num_classes), \
            f"expected ({args.batch_size},{cfg.num_classes}), got {tuple(out.shape)}"
        model_holder["modified"] = model
    check("7. Modified ViT (RoPE) forward pass", check_modified_forward)

    def check_modified_backward():
        model = model_holder["modified"]
        x = torch.randn(args.batch_size, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (args.batch_size,), device=device)
        out = model(x)
        loss = nn.functional.cross_entropy(out, y)
        model.zero_grad()
        loss.backward()
        n_none, n_nan = 0, 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                if p.grad is None:
                    n_none += 1
                elif torch.isnan(p.grad).any():
                    n_nan += 1
        print(f"loss={loss.item():.4f}, params_with_none_grad={n_none}, "
              f"params_with_nan_grad={n_nan}")
        assert n_none == 0, "some parameters received no gradient (check RoPE wiring)"
        assert n_nan == 0, "NaN gradients detected"
    check("8. Modified ViT (RoPE) backward pass", check_modified_backward)

    # ---- 9. output shape equality across variants ----
    def check_shape_equality():
        m1, m2 = model_holder["original"], model_holder["modified"]
        x = torch.randn(args.batch_size, 3, 32, 32, device=device)
        o1, o2 = m1(x), m2(x)
        print(f"original out={tuple(o1.shape)} modified out={tuple(o2.shape)}")
        assert o1.shape == o2.shape, "output shapes differ between variants"
    check("9. Output shape equality (original vs modified)", check_shape_equality)

    # ---- 10. parameter count sanity ----
    def check_param_counts():
        from common.utils import count_parameters
        m1, m2 = model_holder["original"], model_holder["modified"]
        n1, n2 = count_parameters(m1), count_parameters(m2)
        diff_pct = abs(n1 - n2) / n1 * 100
        print(f"original params={n1:,} modified params={n2:,} diff={diff_pct:.2f}%")
        # RoPE adds ~0 learnable params (buffers only) and removes the
        # pos_embedding table, so modified should have SLIGHTLY fewer params.
        assert n2 <= n1, "modified model unexpectedly has more params than original"
        assert diff_pct < 5.0, "parameter counts differ by more than 5%, investigate"
    check("10. Parameter count sanity (original vs modified)", check_param_counts)

    # ---- 11. RoPE norm-preservation check ----
    def check_rope_math():
        from vit_modified.positional_embeddings import RotaryPositionalEmbedding2D
        dim_head, grid = 64, 8
        rope = RotaryPositionalEmbedding2D(dim_head=dim_head, grid_size=grid, num_cls_tokens=1).to(device)
        seq_len = grid * grid + 1
        q = torch.randn(2, 4, seq_len, dim_head, device=device)
        k = torch.randn(2, 4, seq_len, dim_head, device=device)
        q_rot, k_rot = rope.rotate_queries_and_keys(q, k)
        assert q_rot.shape == q.shape and k_rot.shape == k.shape
        norm_before = q.norm(dim=-1)
        norm_after = q_rot.norm(dim=-1)
        max_diff = (norm_before - norm_after).abs().max().item()
        print(f"max |‖q‖-‖q_rot‖| = {max_diff:.6f} (should be ~0, rotation preserves norm)")
        assert max_diff < 1e-3, "RoPE rotation is not norm-preserving -- check implementation"
    check("11. RoPE norm-preservation numerical check", check_rope_math)

    # ---- 12. one-batch overfit smoke test ----
    def check_overfit_smoke(model_key):
        from common.config import ModelConfig
        from common.train import build_model
        cfg = ModelConfig()
        torch.manual_seed(0)
        model = build_model(model_key, cfg).to(device)
        x = torch.randn(16, 3, cfg.image_size, cfg.image_size, device=device)
        y = torch.randint(0, cfg.num_classes, (16,), device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        for step in range(40):
            opt.zero_grad()
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"[{model_key}] loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f}")
        assert losses[-1] < losses[0] * 0.5, (
            f"[{model_key}] loss did not drop enough on a single repeated batch "
            f"({losses[0]:.4f} -> {losses[-1]:.4f}); gradients may not be flowing correctly"
        )

    check("12a. One-batch overfit smoke test (original)", lambda: check_overfit_smoke("original"))
    check("12b. One-batch overfit smoke test (modified)", lambda: check_overfit_smoke("modified"))

    # ---- 13. full mini end-to-end pipeline (few real batches, both variants) ----
    def check_mini_pipeline(model_key):
        from common.config import ModelConfig, TrainConfig
        from common.train import build_model, run_epoch
        cfg = ModelConfig()
        train_cfg = TrainConfig()
        model = build_model(model_key, cfg).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

        train_loader = loaders_holder["train"]
        val_loader = loaders_holder["val"]

        # Truncate to a few batches for a FAST smoke test (not a real epoch).
        from itertools import islice

        class LimitedLoader:
            def __init__(self, loader, n): self.loader, self.n = loader, n
            def __iter__(self): return islice(iter(self.loader), self.n)

        mini_train = LimitedLoader(train_loader, 3)
        mini_val = LimitedLoader(val_loader, 3)

        train_loss, train_acc = run_epoch(model, mini_train, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, mini_val, criterion, optimizer, device, train=False)
        print(f"[{model_key}] mini-epoch: train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        assert not torch.isnan(torch.tensor(train_loss)), "train loss is NaN"
        assert not torch.isnan(torch.tensor(val_loss)), "val loss is NaN"

    check("13a. Full mini pipeline (original, 3 train + 3 val batches)",
          lambda: check_mini_pipeline("original"))
    check("13b. Full mini pipeline (modified, 3 train + 3 val batches)",
          lambda: check_mini_pipeline("modified"))

    # ---- summary ----
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, err in results:
        status = PASS if ok else FAIL
        print(f"[{status}] {name}")
    print(f"\n{n_pass}/{len(results)} checks passed.")

    if n_pass != len(results):
        sys.exit(1)
    print("\nAll diagnostics passed. Safe to launch full training.")


if __name__ == "__main__":
    main()
