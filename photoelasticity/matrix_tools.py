import numpy as np
from scipy.ndimage import zoom


def resize_matrix(matrix, target_shape):
    zoom_factors = (
        target_shape[0] / matrix.shape[0],
        target_shape[1] / matrix.shape[1],
    )
    return zoom(matrix, zoom_factors, order=1)  # order=1 for bilinear interpolation


def find_center_strip(data):
    center_strip_size = 4 * len(data) // 100
    x = y = r = len(data) // 2
    cropped_center = data[y - center_strip_size:y + center_strip_size, x - r:x + r]
    oneDintensities = np.mean(cropped_center, 0)
    mean_brightness = np.mean(oneDintensities)
    numed_data = np.nan_to_num(oneDintensities, nan=mean_brightness)
    return numed_data
