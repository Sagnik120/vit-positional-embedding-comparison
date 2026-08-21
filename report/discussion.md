# Discussion: Results vs. Expectation

> **NOTE TO SELF: fill this in with actual numbers after both training
> runs complete.** Pull the final numbers from
> `results/comparison/comparison_table.md` and
> `results/comparison/top1_test_accuracy_comparison.json`, and look at
> `results/comparison/combined_loss_curves.png` before writing this.

## Results summary

| Metric | Original ViT | Modified ViT (RoPE) |
|---|---|---|
| Top-1 Test Accuracy | `<fill in>` | `<fill in>` |
| Best Val Accuracy | `<fill in>` | `<fill in>` |
| Best Epoch | `<fill in>` | `<fill in>` |
| Generalization Gap (train_acc − val_acc @ best epoch) | `<fill in>` | `<fill in>` |

(Auto-generated version of this table lives at
`results/comparison/comparison_table.md` — copy the numbers over once
both runs finish.)

## Did the results match the expectation?

`<Answer directly: yes / partially / no, then justify with the specific
numbers above.>`

Points to address explicitly, using the actual curves and numbers:

1. **Test accuracy delta.** Was RoPE higher, lower, or statistically
   indistinguishable from the original? By how much (in percentage
   points)? Is this within the noise you'd expect from a single run with
   this model size/dataset, or does it look like a real effect?

2. **Generalization gap.** Did RoPE show a smaller train−val accuracy gap
   at its best epoch, as hypothesized in `justification.md`? Look at
   `results/comparison/generalization_gap_comparison.csv`. If the gap was
   *not* smaller, consider whether the zero-extra-parameter argument
   actually matters as much as expected at this model scale (a few million
   parameters), where the position table itself is a tiny fraction of
   total parameters.

3. **Convergence speed / loss curve shape.** Looking at
   `combined_loss_curves.png`: did one variant reach a given val-loss
   level in fewer epochs? Did val loss inflect upward earlier for one
   variant (a visible overfitting-onset difference)?

4. **Where the prediction was wrong, if it was.** If results did not
   match expectations (e.g., no meaningful accuracy or generalization-gap
   difference), the most likely explanations to consider are:
   - At CIFAR-10 scale (32×32 images, 64 patches), the *absolute* position
     table has very few entries to learn (65 positions × dim), so it may
     already be easy enough to fit that RoPE's parameter-efficiency
     advantage doesn't show up meaningfully.
   - The paper's own finding that ViT's *architectural* lack of
     convolutional inductive bias — not its choice of PE scheme — is the
     dominant factor limiting from-scratch performance on small datasets
     (Section 4.3) is consistent with the PE choice being a comparatively
     minor lever here.
   - A single training run per variant is subject to run-to-run
     variance from initialization/augmentation randomness; a difference
     within a point or two of test accuracy should not be over-interpreted
     without multiple seeds.

5. **Honest bottom line.** State plainly whether this experiment supports,
   contradicts, or is inconclusive about the hypothesis that a relative,
   multiplicative, parameter-free PE scheme improves generalization over
   the original's additive learned PE in this small-model/small-data
   regime, and what a natural next experiment would be (e.g. multiple
   seeds, a larger dataset such as CIFAR-100 or ImageNet-scale, or testing
   position generalization to a different resolution than trained on —
   an area where RoPE's relative formulation is expected to help most).
