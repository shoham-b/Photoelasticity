import numpy as np
from scipy.ndimage import zoom

center_strip_size = 30
percentage_of_circle_to_take = 90


def resize_matrix(matrix, target_shape):
    """
    Resize a 2D NumPy matrix to the target shape using interpolation.
    """
    zoom_factors = (
        target_shape[0] / matrix.shape[0],
        target_shape[1] / matrix.shape[1],
    )
    return zoom(matrix, zoom_factors, order=1)  # order=1 for bilinear interpolation


def find_center_strip(data):
    x = y = r = len(data) // 2
    prominent_r = r * percentage_of_circle_to_take // 100
    cropped_center = data[y - center_strip_size:y + center_strip_size, x - prominent_r:x + prominent_r]
    oneDintensities = np.mean(cropped_center, 0)
    mean_brightness = np.mean(oneDintensities)
    numed_data = np.nan_to_num(oneDintensities, nan=mean_brightness)
    return numed_data
