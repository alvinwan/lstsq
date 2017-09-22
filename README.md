# Least Squares

Featurizations:
- `a3c.py` -- Uses A3C nn to featurize, can output any layer from policy to conv.
- `vgg16.py` -- Uses VGG16 nn to featurize, can output any layer from policy to conv. (TODO)


| Game | # Samples | # Episodes | Layer | Train Acc | Test Acc | Performance* |
|------|-----------|------------|-------|-----------|----------|-------------|
| SpaceInvaders-v0 | 399,000 | 100 | fc5 | 65% | 42% | 824 |
| SpaceInvaders-v0 | 428,000 | 100 | prelu | 85% | ~83% | ~1814 |

*Above, we use 1000 episodes to gauge test time performance.
