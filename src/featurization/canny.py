import numpy as np
from skimage import feature
from skimage import color


def featurize(state):
    state_gs = color.rgb2gray(state)
    edges = feature.canny(state_gs)  # TODO: can be improved by canny per color
    
    c_height = c_width = 10
    n_rows = edges.shape[0] // c_height
    n_cols = edges.shape[1] // c_width

    final = np.zeros((n_rows, n_cols))
    for i in range(n_cols):
        for j in range(n_rows):
            cell = edges[j*c_height:(j+1)*c_height,i*c_width:(i+1)*c_width]
            fraction = np.sum(cell) / (c_height * c_width)
            final[j][i] = fraction > 0.15
    return np.ravel(final)
