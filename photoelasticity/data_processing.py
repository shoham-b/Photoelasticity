import numpy as np


def process_data(data, intensity_threshold_percentage=0.34, center_max=False):
    data = strip_dark_boundaries(data, intensity_threshold_percentage)

    return data


def strip_dark_boundaries(data, intensity_threshold_percentage=0.44):
    max_intensity = np.max(data)
    threshold = max_intensity * intensity_threshold_percentage
    # Find the first index from the left where abs(value) > threshold
    left_idx = 0
    while left_idx < len(data) and abs(data[left_idx]) <= threshold:
        left_idx += 1
    # Find the first index from the right where abs(value) > threshold
    right_idx = len(data)
    while right_idx > 0 and abs(data[right_idx - 1]) <= threshold:
        right_idx -= 1
    largest_boundary = max(left_idx, len(data) - right_idx)
    data = data[largest_boundary:-largest_boundary - 1]

    return data
