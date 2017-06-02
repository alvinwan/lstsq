# Least squares

# Installation

```
pip install -r requirements.txt
```

# Usage

1. Encode your data. By default, the script will look for data at `./data/raw`.
Below, we will name our trial agent `cool`.

> Data is a set of compressed numpy arrays, that's n x (d + 2), where
d = w * h * c for the width, height, and channels of each image, respectively.
The extra 2 columns are for the action index and reward. n is the number of
images saved.

```
python lstsq.py encode --name=cool
```

By default, the above will use `downsample`. You may use other encoding
techniques with the `--featurize` flag.

2. Run a solver. By default, the script will run ordinary least squares.

```
python lstsq.py solve --name=cool
```

The result is effectively an agent.

3. Finally, play the game using your newly trained agent.

```
python lstsq.py play --name=cool
```
