import numpy as np
from skimage import feature
from skimage import color


def featurize_density(state):
    c_height = c_width = 10
    n_rows = state.shape[0] // c_height
    n_cols = state.shape[1] // c_width

    final = np.zeros((n_rows, n_cols, 3))

    for c in range(state.shape[2]):
        edges = feature.canny(state[:, :, c])
        for i in range(n_cols):
            for j in range(n_rows):
                cell = edges[j * c_height:(j + 1) * c_height,
                       i * c_width:(i + 1) * c_width]
                fraction = np.sum(cell) / (c_height * c_width)
                final[j, i, c] = fraction
    return np.ravel(final)


def featurize_density_gs(state):
    state_gs = color.rgb2gray(state)
    return featurize_density(state_gs)


def featurize_threshold(state):
    return featurize_density(state) > 0.15
