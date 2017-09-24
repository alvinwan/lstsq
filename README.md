# Least Squares

Featurizations (`featurizations/`):
- `a3c.py` -- Standalone script, uses A3C nn to featurize, can output any layer from policy to conv.
- `conv.py` -- Uses `conv_tf` to featurize, can use a number of filters including patches from original images
- `vgg16.py` -- Standalone script, uses VGG16 nn to featurize, can output any layer from policy to conv. (TODO)

Below, we use 1000 episodes to gauge performance for both the least squares model and A3C. Statistics with a tilde ~ indicate the experiment is currently running.

## A3C Featurizations

The least squares agents below are trained using OLS. No regularization has been added.

| Game | # Fr | # Ep | Feat | Train Acc | Test Acc | Perf | A3C Perf |
|------|------|------|------|-----------|----------|-------|-----------|
| Alien-v0 | 3,520 | 1 | prelu | 90% | 29% | 676 | 3123 |
| Alien-v0 | 564,794 | 30 | prelu | 81% | ~75% | ~2410 | 3123 |
| Breakout-v0 | 15,394 | 1 | prelu | 80% | ~68% | ~538 | 727 |
| Breakout-v0 | 712,054 | 30 | prelu | 77% | ~72% | ~622 | 727 |
| Centipede-v0 | 68,844 | 1 | prelu | 81% | 79% | 2723 | 2549 |
| Centipede-v0 | 705,146 | 30 | prelu | 80% | 79% | 2846 | 2549 |
| SpaceInvaders-v0 | 399,000 | 100 | fc5 | 65% | 42% | 824 | 4012 |
| SpaceInvaders-v0 | 428,000 | 100 | prelu | 85% | 82% | 2495 | 4012 |

*Note that the downsampling method matters. The A3C model was trained using `cv2.resize(... interpolation=cv2.INTER_LINEAR)`. Using `scipy.misc.imresize(... mode='nearest')` resulted in significantly worse performance for `Alien-v0`. With 500,680  samples (30 episodes) and a3c prelu featurization, the agent achieved 67% training, ~29% testing, and ~355 reward. These results are also included in the `results/` directory.

## Convolutional Featurizations

| Game | # Fr | # Ep | Feat | Train Acc | Test Acc | Perf | A3C Perf |
|------|------|------|------|-----------|----------|-------|-----------|
| Breakout-v0 | 15,394 | 1 | conv | 48% | -- | -- | 727 |
| SpaceInvaders-v0 | 10,000 | 1 | conv | 29% | -- | -- | 4012 |

### Hyperparameter Tuning for SpaceInvaders-v0

Use just 1 episode and train accuracy as proxy:
- defaults: 24.51%
- `max`: 24.51%
- `bias=5.0`: 
