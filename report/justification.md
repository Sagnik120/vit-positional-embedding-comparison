# Justification for the Chosen Alternative Positional Embedding

**Chosen scheme:** 2D Rotary Position Embedding (RoPE), applied to
query/key vectors inside every self-attention layer.

## Why this scheme, and why it is a genuinely different mechanism

The original ViT (Dosovitskiy et al., 2021) encodes position with a
single learned vector **added** to each patch embedding once, at the
model's input stem (Eq. 1 in the paper). The paper's own Appendix D.4
ablation only varies *what* is added (1D, 2D, or relative-bias) and
*where* it is added (stem vs. every layer) — but every variant they test
is still an **additive** scheme that injects position once as a vector
sum.

RoPE is structurally different on two axes:

1. **Multiplicative, not additive.** Instead of adding a position vector
   to the token embedding, RoPE *rotates* the query and key vectors by an
   angle that depends on the token's spatial position. Position never
   appears as a term being summed into the representation; it appears as
   a transformation applied to Q/K right before the attention dot
   product.
2. **Relative by construction, not by an extra learned bias term.** The
   dot product of two RoPE-rotated vectors depends mathematically only on
   the *difference* between their positions, not their absolute
   positions. The paper's "Rel. Pos. Emb." ablation (Appendix D.4) also
   aims at relative position, but does so by learning an explicit bias
   term added to the attention logits for every relative offset — an
   extra set of learned parameters bolted onto content-based attention.
   RoPE achieves relative-position sensitivity with **zero extra
   parameters**, as a direct property of the rotation itself.

Concretely, in this repository RoPE removes the `pos_embedding`
`nn.Parameter` entirely (see `CHANGES.md`) and instead rotates Q/K inside
every `Attention.forward()` call, using a fixed (non-learned) frequency
table split across the patch grid's row and column axes.

## What I expected this to change

- **No parameter-count confound.** Because RoPE adds no learnable
  parameters, any accuracy difference between the two runs is
  attributable to the *mechanism*, not to one model simply having more
  capacity — unlike, say, comparing against a per-layer learned PE, which
  would add parameters at every depth.
- **Better relative-position generalization / possibly faster
  convergence.** Because attention scores become an explicit function of
  patch-to-patch relative offset at every layer (not just at the input),
  I expected the model to more directly exploit the 2D grid structure of
  the image without having to "learn" a position table from random
  initialization for every one of the ~65 patch positions from scratch.
- **A smaller generalization gap, especially in this low-data, small-model,
  from-scratch regime.** CIFAR-10 with a small ViT trained from scratch is
  exactly the setting where the paper says ViT struggles without either
  scale or strong regularization (Section 4.3). Since RoPE contributes no
  extra free parameters to overfit and encodes a useful geometric prior
  (relative offsets) without hand-tuning, I expected it to overfit somewhat
  less than the freely-learned absolute position table, and possibly reach
  a comparable or higher peak validation accuracy before val loss starts
  climbing.
- I did **not** expect a dramatic accuracy jump — on a dataset this small,
  most of the ViT's difficulty comes from lacking convolutional inductive
  bias altogether (translation equivariance, locality), which no
  positional embedding scheme alone fixes. I expected the effect of the PE
  choice to be a **second-order** effect on top of that larger limitation.
