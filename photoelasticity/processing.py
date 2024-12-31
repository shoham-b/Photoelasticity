import numpy as np
from scipy.signal import find_peaks, argrelmin


def process_data(data):
    max_intensity = np.max(data)
    threshold = max_intensity * 0.44
    # Find the first index from the left where abs(value) > threshold
    left_idx = 0
    while left_idx < len(data) and abs(data[left_idx]) <= threshold:
        left_idx += 1
    # Find the first index from the right where abs(value) > threshold
    right_idx = len(data)
    while right_idx > 0 and abs(data[right_idx - 1]) <= threshold:
        right_idx -= 1
    data = data[left_idx:right_idx]
    data = data - np.min(data)

    return data


def center_data_around_radius(data):
    minima_indices = argrelmin(data,order=50)[0]
    highest_minima_idx = minima_indices[np.argmax(data[minima_indices])]
    radius = min((len(data)) - highest_minima_idx, highest_minima_idx)
    data = data[highest_minima_idx - radius:highest_minima_idx + radius + 1]
    if len(data) % 2:  # we need a center
        data = data[:-1]
    return data
