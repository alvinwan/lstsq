# Least Squares

Featurizations (`featurizations/`):
- `a3c.py` -- Standalone script, uses A3C nn to featurize, can output any layer from policy to conv.
- `conv.py` -- Uses `conv_tf` to featurize, can use a number of filters including patches from original images
- `vgg16.py` -- Standalone script, uses VGG16 nn to featurize, can output any layer from policy to conv. (TODO)

Below, we use 1000 episodes to gauge performance for both the least squares model and A3C. Statistics with a tilde ~ indicate the experiment is currently running.

## A3C Featurizations

| Game | # Fr | # Ep | Feat | Train Acc | Test Acc | Perf | A3C Perf |
|------|------|------|------|-----------|----------|-------|-----------|
| Alien-v0 | 3,520 | 1 | prelu | 90% | 29% | ~693 | 3123 |
| Alien-v0 | 500,680 | 30 | prelu | -- | -- | -- | 3123 |
| Breakout-v0 | 15,394 | 1 | prelu | 80% | ~68% | ~538 | 727 |
| Breakout-v0 | 712,054 | 30 | prelu | 77% | ~72% | ~622 | 727 |
| Centipede-v0 | 68,844 | 1 | prelu | 81% | 79% | 2723 | 2549 |
| Centipede-v0 | 705,146 | 30 | prelu | 80% | 79% | 2846 | 2549 |
| SpaceInvaders-v0 | 399,000 | 100 | fc5 | 65% | 42% | 824 | 4012 |
| SpaceInvaders-v0 | 428,000 | 100 | prelu | 85% | 82% | 2495 | 4012 |


## Convolutional Featurizations

| Game | # Fr | # Ep | Feat | Train Acc | Test Acc | Perf | A3C Perf |
|------|------|------|------|-----------|----------|-------|-----------|
| Breakout-v0 | 15,394 | 1 | conv | 48% | -- | -- | 727 |
| SpaceInvaders-v0 | 10,000 | 1 | conv | 29% | -- | -- | 4012 |
