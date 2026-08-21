"""
positional_embeddings.py

Registry of positional embedding (PE) schemes for the modified ViT.

Design goal: adding a NEW PE scheme in the future should require touching
ONLY this file (add a class + one line in PE_REGISTRY), never vit.py or
train.py.

Currently implemented:
    - "learned_1d" : reference wrapper around the ORIGINAL additive learned
                      1D positional embedding (paper default), included here
                      only so the registry / CLI is symmetric. The actual
                      original-ViT training run uses src/vit_original/vit.py
                      directly and does NOT go through this file.
    - "rope_2d"    : 2D Rotary Position Embedding (RoPE), applied inside
                      self-attention to the query/key vectors rather than
                      added to the patch embeddings at the stem.

RoPE reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
Position Embedding" (2021), extended here to 2 spatial axes (row, col)
for image patch grids, following the common 2D-RoPE recipe used in
several ViT variants (e.g. EVA, RoPE-ViT).
"""

import torch
from torch import nn


class LearnedPositionalEmbedding1D(nn.Module):
    """
    Reference-only wrapper matching the ORIGINAL ViT's PE (Eq. 1 of the
    paper): a single learned (num_patches+1, dim) table, added once to the
    patch+cls-token embeddings at the stem, before the first transformer
    block. Not used directly in the modified-model training path (the
    original vit.py already implements this natively); kept here purely
    for API symmetry / documentation.
    """

    def __init__(self, num_patches: int, dim: int, num_cls_tokens: int = 1):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(num_patches + num_cls_tokens, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.shape[1]
        return x + self.pos_embedding[:seq]


class RotaryPositionalEmbedding2D(nn.Module):
    """
    2D Rotary Position Embedding for a square patch grid, applied to
    query/key vectors inside every attention layer (NOT added at the stem).

    Mechanism
    ---------
    Standard 1D RoPE splits the head dimension into pairs of channels and
    rotates each pair by an angle proportional to the token's position and
    a per-pair frequency. The dot product of two rotated vectors then
    depends only on their RELATIVE position, not their absolute position
    -- this is what makes RoPE a relative, multiplicative scheme, as
    opposed to the paper's additive absolute (learned_1d) or additive
    relative-bias (Appendix D.4, "Rel. Pos. Emb.") schemes.

    For 2D patch grids we split the head dimension in half: one half is
    rotated using the patch's ROW index, the other half using its COLUMN
    index, each with its own set of frequencies. This is the standard
    "axial" extension of RoPE to 2D grids used in several vision RoPE
    implementations.

    The [CLS] token has no natural (row, col) position; it is assigned a
    fixed rotation angle of zero (i.e., left un-rotated), which is the
    standard convention.

    Usage
    -----
    Instantiated once per model with the patch grid size, then passed into
    every Attention block. `rotate_queries_and_keys(q, k)` is called right
    before the scaled dot product in Attention.forward().
    """

    def __init__(self, dim_head: int, grid_size: int, num_cls_tokens: int = 1,
                 base: float = 10000.0):
        super().__init__()
        assert dim_head % 4 == 0, (
            "dim_head must be divisible by 4 for 2D-axial RoPE "
            "(half the dims go to the row axis, half to the col axis, "
            "and each axis needs pairs of dims to rotate)."
        )
        self.dim_head = dim_head
        self.grid_size = grid_size
        self.num_cls_tokens = num_cls_tokens

        # Half the head-dim is used for the row-axis rotation, half for col.
        dim_per_axis = dim_head // 2
        freq_seq = torch.arange(0, dim_per_axis, 2).float()
        inv_freq = 1.0 / (base ** (freq_seq / dim_per_axis))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute row/col index grid for every patch position.
        coords = torch.arange(grid_size)
        row_idx = coords.view(-1, 1).expand(grid_size, grid_size).reshape(-1)
        col_idx = coords.view(1, -1).expand(grid_size, grid_size).reshape(-1)
        self.register_buffer("row_idx", row_idx, persistent=False)
        self.register_buffer("col_idx", col_idx, persistent=False)

    def _build_cos_sin(self, device, dtype):
        row_freqs = torch.einsum("i,j->ij", self.row_idx.to(device).float(),
                                  self.inv_freq.to(device))
        col_freqs = torch.einsum("i,j->ij", self.col_idx.to(device).float(),
                                  self.inv_freq.to(device))
        # (num_patches, dim_per_axis/2) each -> concat row+col -> (num_patches, dim_head/2)
        freqs = torch.cat([row_freqs, col_freqs], dim=-1)
        # duplicate for sin/cos pairing -> (num_patches, dim_head)
        freqs = torch.cat([freqs, freqs], dim=-1)

        cls_freqs = torch.zeros(self.num_cls_tokens, freqs.shape[-1],
                                 device=device)
        freqs = torch.cat([cls_freqs, freqs], dim=0)  # (seq, dim_head)

        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def rotate_queries_and_keys(self, q: torch.Tensor, k: torch.Tensor):
        """
        q, k: (batch, heads, seq_len, dim_head)
        Returns rotated (q, k) of the same shape.
        """
        seq_len = q.shape[-2]
        cos, sin = self._build_cos_sin(q.device, q.dtype)
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,seq,dim_head)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Registry -- add new PE schemes here, nowhere else.
# ---------------------------------------------------------------------------
PE_REGISTRY = {
    "learned_1d": LearnedPositionalEmbedding1D,
    "rope_2d": RotaryPositionalEmbedding2D,
    # "your_new_scheme": YourNewSchemeClass,
}
