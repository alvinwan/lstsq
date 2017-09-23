# Least Squares

Featurizations:
- `a3c.py` -- Uses A3C nn to featurize, can output any layer from policy to conv.
- `vgg16.py` -- Uses VGG16 nn to featurize, can output any layer from policy to conv. (TODO)


| Game | # Fr. | # Ep. | Feat. | Train Acc | Test Acc | Performance* |
|------|-------|-------|-------|-----------|----------|--------------|
| SpaceInvaders-v0 | 399,000 | 100 | fc5 | 65% | 42% | 824 |
| SpaceInvaders-v0 | 428,000 | 100 | prelu | 85% | 82% | 2495 |
| SpaceInvaders-v0 | 10,000 | 1 | conv | 29% | -- | -- |
| Centipede-v0 | 68,844 | 1 | prelu | 81% | ~80% | ~2773 |
| Breakout-v0 | 15,394 | 1 | prelu | 80% | ~67% | ~420 |

*Above, we use 1000 episodes to gauge test time performance. Statistics with a tilde ~ indicate the experiment is currently running.
