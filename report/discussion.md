# Discussion: Results vs. Expectation

## Results Summary

| Metric | Original ViT | Modified ViT (RoPE) |
|---|---|---|
| Top-1 Test Accuracy | 83.73% | 86.77% |
| Best Val Accuracy | 0.8428 | 0.8706 |
| Best Epoch | 80 | 90 |
| Generalization Gap (train_acc − val_acc @ best epoch) | 0.0547 | 0.0828 |

## Did the results match the expectation?

**Yes, the results partially match and even exceed expectations in terms of classification performance, though the generalization gap behaved slightly differently than hypothesized.**

### 1. Test Accuracy Delta
The Modified ViT (RoPE) achieved a **86.77%** Top-1 test accuracy, outperforming the Original ViT (**83.73%**) by **3.04 percentage points (pp)**. This is a substantial and meaningful improvement for a positional embedding swap under identical hyperparameters. Because RoPE introduces zero learnable parameters (the parameter count for the modified model is actually slightly lower due to the removal of the learned 1D position embeddings), this delta is purely a result of the multiplicative relative-position formulation of RoPE acting as a stronger inductive bias.

### 2. Generalization Gap
Our initial hypothesis expected RoPE to exhibit a smaller generalization gap due to its parameter-free nature. However, the actual generalization gap at the best epoch was **0.0828** for RoPE compared to **0.0547** for the Original ViT. 
This happened because the RoPE model was able to train more deeply and fit the training data much better, reaching a training accuracy of **95.34%** at epoch 90 (compared to **89.75%** at epoch 80 for the baseline). Since it fit the underlying 2D coordinates much more effectively, it continued to improve validation performance longer, but also pushed its training accuracy higher, widening the gap.

### 3. Convergence Speed and Loss Curves
Looking at the loss curves (`results/comparison/combined_loss_curves.png`), the Modified ViT (RoPE) converges much faster in the early epochs. It drops below a training loss of 1.0 significantly earlier than the original model. Furthermore, the Original ViT's validation loss inflected upward earlier (around epoch 50), whereas the RoPE model's validation loss remained flatter and continued to decrease until epoch 90, demonstrating superior stability.

### 4. Bottom Line
This experiment provides strong empirical evidence that incorporating a 2D relative, multiplicative, parameter-free positional embedding (RoPE) improves generalization and final representation quality over the original learned 1D additive positional embedding in this small-model/low-data regime. 

To expand on this finding, future work should evaluate whether this 3.04 pp accuracy advantage holds across multiple random seeds, or if the relative structure of RoPE allows the model to generalize seamlessly to different test-time image resolutions (e.g., evaluating a 32x32-trained model on 48x48 images), which absolute learned embeddings cannot support.
