# Least Squares

Featurizations:
- `a3c.py` -- Uses A3C nn to featurize, can output any layer from policy to conv.
- `vgg16.py` -- Uses VGG16 nn to featurize, can output any layer from policy to conv. (TODO)


| Game | # Fr. | # Ep. | Feat. | Train Acc | Test Acc | Performance* |
|------|-------|-------|-------|-----------|----------|--------------|
| Alien-v0 | 3,520 | 1 | prelu | 90% | -- | -- |
| Breakout-v0 | 15,394 | 1 | prelu | 80% | ~68% | ~566 |
| Breakout-v0 | 15,394 | 1 | conv | 48% | -- | -- |
| Breakout-v0 | 712,054 | 30 | prelu | -- | -- | -- |
| Centipede-v0 | 68,844 | 1 | prelu | 81% | 79% | 2723 |
| Centipede-v0 | 705,146 | 30 | prelu | -- | -- | -- |
| SpaceInvaders-v0 | 399,000 | 100 | fc5 | 65% | 42% | 824 |
| SpaceInvaders-v0 | 428,000 | 100 | prelu | 85% | 82% | 2495 |
| SpaceInvaders-v0 | 10,000 | 1 | conv | 29% | -- | -- |

*Above, we use 1000 episodes to gauge test time performance. Statistics with a tilde ~ indicate the experiment is currently running.
