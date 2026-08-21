# 02. Positional Embedding Choice

## Constraint

Assignment explicitly disallows reusing the paper's own Appendix D.4
ablations (no PE, 1D learned, 2D learned, relative-bias PE) verbatim as
"our" alternative scheme.

## Options considered

| Scheme | Considered? | Verdict |
|---|---|---|
| Fixed 2D sinusoidal (non-learned, added at stem) | Yes | Rejected — too close in spirit to the paper's own "1-D/2-D Pos. Emb." ablation (still additive, still at the stem), even though non-learned. Weaker justification story. |
| Relative positional bias (paper's own scheme, reworded) | No | Explicitly disallowed by the assignment. |
| ALiBi (linear bias added to attention logits by distance) | Considered | Deferred to `PE_REGISTRY` as a future extension; still an additive-bias-to-logits scheme, conceptually adjacent to the paper's relative-bias ablation. |
| **2D Rotary Position Embedding (RoPE)** | **Chosen** | Mechanistically distinct: multiplicative rotation of Q/K inside every attention layer, zero extra parameters, relative-by-construction. See `report/justification.md` for the full argument. |

## Implementation notes

- RoPE applied per-axis (row, col) by splitting the attention head
  dimension in half — standard "axial" 2D-RoPE extension.
- `[CLS]` token assigned rotation angle 0 (left un-rotated) since it has
  no natural grid position — standard convention.
- Implemented as a single shared `RotaryPositionalEmbedding2D` module
  instantiated once per model and passed by reference into every
  `Attention` block, so all layers rotate consistently with the same
  frequency table (see `src/vit_modified/vit.py`, `CHANGES.md`).
- Verified numerically that the rotation is norm-preserving
  (`scripts/run_diagnostics.py`, check #11) — a basic correctness
  property RoPE must satisfy.
