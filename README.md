# Least Squares

Featurizations (`featurizations/`):
- `a3c.py` -- Standalone script, uses A3C nn to featurize, can output any layer from policy to conv.
- `conv.py` -- Uses `conv_tf` to featurize, can use a number of filters including patches from original images
- `vgg16.py` -- Standalone script, uses VGG16 nn to featurize, can output any layer from policy to conv.

Below, we use 1000 episodes to gauge performance for both the least squares model and A3C. Statistics with a tilde ~ indicate the experiment is currently running. For comparison, ridge regression on the raw pixels (downsampled, `84x84x3`) achieves the following:

| Regularization | Train Acc |
|----------------|-----------|
| 1e-7 | 87.56% |
| 1e-5 | 87.24% |
| 1e-3 | 87.25% |
| 1e-1 | 86.94% |
| 1 | 87.05% |
| 1e1 | 86.49% |
| 1e2 | 86.75% |
| 1e3 | 86.75% |
| 1e5 | 79.48% |

With that said, the least squares agent does no better than guessing on the test set.

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

Below are a number of different convolutional featurizations. Here, we use randomly-selected patches from the provided data.

| Game | # Fr | # Ep | Feat | Train Acc | Test Acc | Perf | A3C Perf |
|------|------|------|------|-----------|----------|------|----------|
| Breakout-v0 | 15,394 | 1 | conv | 48% | -- | -- | 727 |
| Breakout-v0 | 15,394 | 1 | vgg16 | 17% | -- | -- | 727 |
| SpaceInvaders-v0 | 10,000 | 1 | conv | 29% | -- | -- | 4012 |

### `conv_tf` Hyperparameter Tuning for SpaceInvaders-v0

The following are training accuracies, using just one episode of SpaceInvaders-v0. `I` means that the matrix was ill-conditioned and non-invertible.

| Parameters | Conv | 1e-7 | 1e-5 | 1e-3 | 1e-1 | 1 | 1e1 | 1e2 | 1e3 | 1e5 |
|------------|------|------|------|------|------|---|-----|-----|-----|-----|
| Default | 1024x2x2 | I | I | I | 24.54% | 24.54% | 24.54% | 24.51% | 24.06% | 29.54% |
| max | 1024x2x2 | I | I | I | 24.54% | 24.54% | 24.54% | 24.51% | 24.06% | 29.54% |
| bias=5.0 | 1024x2x2 | I | I | I | 28.98% | 28.98% | 28.97% | 28.93% | 28.94% | 29.45% |
| bias=10.0 | 1024x2x2 | I | I | I | 28.59% | 28.59% | 28.6% | 28.52% | 28.24% | 27.94% |
| patch=7,pool=141 | 1024x2x2 | I | I | I | I | I | I | I | I | I |
| patch=8,pool=140 | 1024x2x2 | I | I | I | 30.7% | 30.71% | 30.87% | 30.57% | 30.92% | 30.88% |
| patch=9,pool=139 | 1024x2x2 | I | I | I | 24.47% | 23.98% | 22.76% | 30.56% | 29.44% | 29.65% |
| patch=10,pool=138 | 1024x2x2 | I | I | I | 28.62% | 28.62% | 28.47% | 28.29% | 30.83% | 29.41% |
| patch=12,pool=136 | 1024x2x2 | I | I | I | 30.4% | 30.4% | 30.42% | 30.43% | 30.73% | 29.71% |
| patch=14,pool=134 | 1024x2x2 | I | I | I | 27.43% | 27.32% | 27.22% | 22.35% | 30.52% | 27.17% |
| patch=16,pool=132 | 1024x2x2 | I | I | I | 30.92% | 30.91% | 30.93% | 30.87% | 30.75% | 28.18% |
| patch=18,pool=130 | 1024x2x2 | I | I | I | 28.23% | 28.23% | 28.23% | 28.17% | 27.86% | 25.55% |
| patch=20,pool=128 | 1024x2x2 | I | I | I | 34.73% | 34.73% | 34.81% | **35.06%** | 34.88% | 28.89% |
| patch=22,pool=126 | 1024x2x2 | I | I | I | 27.02% | 27.02% | 27.02% | 27.12% | 27.28% | 34.63% |
| patch=24,pool=124 | 1024x2x2 | I | I | I | 31.18% | 31.17% | 31.17% | 31.17% | 31.2% | 21.24% |
| patch=26,pool=122 | 1024x2x2 | I | I | I | 25.97% | 25.97% | 25.97% | 25.98% | 26.03% | 28.65% |
| patch=28,pool=120 | 1024x2x2 | I | I | I | I | 28.81% | 28.81% | 28.81% | 28.81% | 27.62% |
| patch=30,pool=118 | 1024x2x2 | I | I | I | I | 32.76% | 32.76% | 32.76% | 32.76% | 33.12% |
| patch=10,pool=138 | 2048x2x2 | I | I | I | 30.14% | 30.15% | 30.28% | 30.19% | 30.71% | 29.5% |
| patch=10,pool=138 | 4096x2x2 | I | I | I | 29.89% | 29.89% | 29.87% | 29.86% | 29.49% | 25.93% |
| patch=20,pool=128 | 2048x2x2 | I | I | I | 33.67% | 33.66% | 33.72% | 34.01% | 34.19% | 34.17% |
| patch=20,pool=128 | 4096x2x2 | I | I | I | 36.07% | 36.07% | 36.09% | 35.28% | 33.82% | 34.4% |

## BlobProst

### Blob

The below are all training accuracies.

| n | episodes | parameters | d | 1e-7 | 1e-5 | 1e-3 | 1e-1 | 1 | 1e1 | 1e2 | 1e3 | 1e5 |
|---|----------|------------|---|------|------|------|------|---|-----|-----|-----|-----|
| 17873 | 22 | bpc=5 | 36k | 71.02% | 71.02% | 71.02% | 70.98% | 70.64% | 69.91% | 67.24% | 62.08% | 51.42% |

