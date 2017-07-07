# Scratch

Mess of a few files. Turns out 51k episodes is the most havoc can take, so I needed to stream all computations.

The following three allow me to compute `(x-mu)^T(x-mu) = x^Tx - 2x^Tmu + mu^Tmu`. At some point, the latter, I may want to rewrite in the first form.

- xtx: `raw-atari-xtx-final`, `raw-atari-xty-final`
- mu: `raw-atari-mu-final`
- xtmu: `raw-atari-xtmu-final`

Once this is done, diagonalize `xtx` to get the right singular vectors of `x`, `V`. Take the projected `X` to be `D`. Then, we want `D^TD` and `D^TY`.

- pca: `raw-atari-v{1,2,3,4,5...20}`
- project: `raw-atari-xvtxv-final`, `raw-atari-xtv-final` (misnomer, should be `xv`), `raw-atari-xvty-final`

Compute your models.

- w: `raw-atari-w{1,2,3,4,5...20}`

Then, run the game and collect rewards in standard format. Ripped it off from the main script, but reading from `V` instead of a model.

```
python play.py
```
