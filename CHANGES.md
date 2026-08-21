# CHANGES: Original ViT → Modified ViT (RoPE)

**Baseline file:** `src/vit_original/vit.py`
(unmodified copy of `vit_pytorch/vit.py` from
[lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch), MIT
license, 139 lines.)

**Modified file:** `src/vit_modified/vit.py` (143 lines)

**New file (has no counterpart in the original):**
`src/vit_modified/positional_embeddings.py` — contains the
`RotaryPositionalEmbedding2D` class and the `PE_REGISTRY`.

Every other file in this repository (`src/common/*.py`, `scripts/*`,
`src/vit_original/vit.py` itself) is **byte-identical in logic** between
the two training runs — the only architectural difference between the
"original" and "modified" experiments is the positional embedding
mechanism documented below.

---

## Line-by-line changes in `vit.py`

| Original line(s) | Modified line(s) | Change |
|---|---|---|
| 5–6 | 5–7 | **ADDED** `from vit_modified.positional_embeddings import RotaryPositionalEmbedding2D` |
| 31 | 32 | **CHANGED** `Attention.__init__` signature: added `rotary_pe: RotaryPositionalEmbedding2D = None` parameter |
| 44 (after) | 46 | **ADDED** `self.rotary_pe = rotary_pe` — stores a reference to the shared RoPE module inside each `Attention` block |
| 54–55 (after) | 58–59 | **ADDED** rotation step: `if self.rotary_pe is not None: q, k = self.rotary_pe.rotate_queries_and_keys(q, k)` — applied to q/k immediately after the `to_qkv` projection and before the scaled dot product |
| 67 | 71 | **CHANGED** `Transformer.__init__` signature: added `rotary_pe: RotaryPositionalEmbedding2D = None` parameter |
| 74 | 78 | **CHANGED** `Attention(...)` construction inside the layer loop now forwards `rotary_pe=rotary_pe` to every block, so all layers share one RoPE frequency table |
| 91 | 95–96 | **ADDED** assertion enforcing a square image / square patch grid (required for 2D-axial RoPE row/col indexing); original only asserted divisibility |
| 93–94 | 98 | **ADDED** `grid_size = image_height // patch_height` — needed to instantiate `RotaryPositionalEmbedding2D` |
| **107** `self.pos_embedding = nn.Parameter(...)` | **114** (commented out, not executed) | **REMOVED** — the learned additive absolute positional embedding table is deleted entirely. No stem-level positional parameter exists in the modified model. |
| — | 115 | **ADDED** `self.rotary_pe = RotaryPositionalEmbedding2D(dim_head=dim_head, grid_size=grid_size, num_cls_tokens=num_cls_tokens)` — one shared RoPE module instantiated once per model |
| 111 | 119 | **CHANGED** `Transformer(...)` construction now passes `rotary_pe=self.rotary_pe` through |
| **125, 127** `seq = x.shape[1]` / `x = x + self.pos_embedding[:seq]` | **133, 135** (commented out, not executed) | **REMOVED** — no positional information is added to the patch/cls-token embeddings at the stem. Positional information only enters the model inside `Attention.forward` via q/k rotation. |

Everything else — `FeedForward`, the residual/pre-LN structure of
`Transformer.forward`, patch embedding (`to_patch_embedding`), class-token
handling, pooling (`cls`/`mean`), and the classification head — is
**unchanged, line-for-line**, from the original.

---

## New module: `positional_embeddings.py`

| Class | Purpose |
|---|---|
| `LearnedPositionalEmbedding1D` | Reference-only wrapper mirroring the original's additive PE (Eq. 1 of the paper). Not used in the modified training path — included for registry symmetry / documentation only. |
| `RotaryPositionalEmbedding2D` | The actual PE mechanism used by the modified model. Splits each attention head's dimension into two halves (row-axis, col-axis), rotates q/k by an angle proportional to each patch's row/col index and a per-pair frequency (standard RoPE, extended axially to 2 spatial dimensions). The `[CLS]` token is left unrotated (angle = 0). |
| `PE_REGISTRY` | `dict` mapping scheme name → class, so a future third scheme can be added by writing one new class and adding one line here — no other file needs to change. |

---

## Parameter count effect (sanity-checked, see `scripts/run_diagnostics.py` check #10)

Removing `pos_embedding` (a learned `(num_patches+1, dim)` table) and
adding `RotaryPositionalEmbedding2D` (which holds only non-learnable
`register_buffer` tensors, zero trainable parameters) means the modified
model has **slightly fewer** trainable parameters than the original —
confirmed at runtime to differ by well under 1%. This keeps the
comparison a fair test of the *mechanism*, not of extra model capacity.

Example (with the default `ModelConfig` in `src/common/config.py`,
image_size=32, patch_size=4, dim=256, depth=6, heads=4, dim_head=64):
- Original: 3,191,146 trainable params
- Modified: 3,174,506 trainable params
