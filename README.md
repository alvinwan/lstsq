# Least squares

# Installation

```
pip install -r requirements.txt
```

# Usage

1. Encode your data. By default, the script will look for data at `./data/raw`

```
python lstsq.py encode
```

By default, the above will use `downsample`. You may use other encoding
techniques with the `--featurize` flag.

2. Run a solver. By default, the script will run ordinary least squares.

```
python lstsq.py solve
```
