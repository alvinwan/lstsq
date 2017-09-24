# Least Squares

Featurizations (`featurizations/`):
- `a3c.py` -- Standalone script, uses A3C nn to featurize, can output any layer from policy to conv.
- `conv.py` -- Uses `conv_tf` to featurize, can use a number of filters including patches from original images
- `vgg16.py` -- Standalone script, uses VGG16 nn to featurize, can output any layer from policy to conv. (TODO)

Below, we use 1000 episodes to gauge performance for both the least squares model and A3C. Statistics with a tilde ~ indicate the experiment is currently running.

## A3C Featurizations

The least squares agents below are trained using OLS. No regularization has been added. The least squares agent scores anywhere from 60% to 117% of A3C's reward.

| Game | # Fr | # Ep | Feat | Train Acc | Test Acc | Perf | A3C Perf |
|------|------|------|------|-----------|----------|-------|-----------|
| Alien-v0 | 3,520 | 1 | prelu | 90% | 29% | 676 | 3123 |
| Alien-v0 | 564,794 | 30 | prelu | 81% | 73% | 2194 | 3123 |
| Breakout-v0 | 15,394 | 1 | prelu | 80% | 68% | 553 | 727 |
| Breakout-v0 | 712,054 | 30 | prelu | 77% | 71% | 602 | 727 |
| Centipede-v0 | 68,844 | 1 | prelu | 81% | 79% | 2723 | 2549 |
| Centipede-v0 | 705,146 | 30 | prelu | 80% | 79% | 2846 | 2549 |
| SpaceInvaders-v0 | 399,000 | 100 | fc5 | 65% | 42% | 824 | 4012 |
| SpaceInvaders-v0 | 428,000 | 100 | prelu | 85% | 82% | 2495 | 4012 |

*Note that the downsampling method matters. The A3C model was trained using `cv2.resize(... interpolation=cv2.INTER_LINEAR)`. Using `scipy.misc.imresize(... mode='nearest')` resulted in significantly worse performance for `Alien-v0`. With 500,680  samples (30 episodes) and a3c prelu featurization, the agent achieved 67% train, 29% test, and 335 reward. These results are also included in `results/Alien-v0_prelu_30.txt`.

## Convolutional Featurizations

Below are a number of different convolutional featurizations.

Here, we use randomly-selected patches from the provided data.

| Game | # Fr | # Ep | Feat | Train Acc | Test Acc | Perf | A3C Perf |
|------|------|------|------|-----------|----------|------|----------|
| Breakout-v0 | 15,394 | 1 | conv | 48% | -- | -- | 727 |
| SpaceInvaders-v0 | 10,000 | 1 | conv | 29% | -- | -- | 4012 |

### `conv_tf` Hyperparameter Tuning for SpaceInvaders-v0

The following are training accuracies, using just one episode of SpaceInvaders-v0. `I` means that the matrix was ill-conditioned and non-invertible.

| Parameters | 1e-7 | 1e-5 | 1e-3 | 1e-1 | 1 | 1e1 | 1e2 | 1e3 | 1e5 |
|------------|------|------|------|------|---|-----|-----|-----|-----|
| Defaults | I | I | I | 24.54% | 24.54% | 24.54% | 24.51% | 24.06% | 29.54% |
| max | I | I | I | 24.54% | 24.54% | 24.54% | 24.51% | 24.06% | 29.54% |
| bias=5.0 | I | I | I | 28.98% | 28.98% | 28.97% | 28.93% | 28.94% | 29.45% |
| bias=10.0 | I | I | I | 28.59% | 28.59% | 28.6% | 28.52% | 28.24% | 27.94% |
