import numpy as np

from photoelasticity.tools.matrix_tools import resize_matrix


def test_resize_matrix():
    matrix = np.array([[1, 2], [3, 4]])
    resized = resize_matrix(matrix, (4, 4))
    assert resized.shape == (4, 4)
