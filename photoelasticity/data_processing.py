import numpy as np
from numpy.ma.core import argmax
from scipy.optimize import minimize
from scipy.signal import find_peaks, argrelmin, argrelmax

ARG_EXTREME_ORDER = 70


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def process_data(data, intensity_threshold_percentage=0.34, center_max=False):
    data = strip_dark_boundaries(data, intensity_threshold_percentage)
    # data = center_data_around_radius(data, center_max)

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


def center_data_around_radius(data, center_max=False):
    if center_max:
        averaging_amount = 200
        averaged_data = moving_average(data, averaging_amount)
        minima_indices = argrelmin(averaged_data, order=ARG_EXTREME_ORDER)[0]
        maxima = argmax(averaged_data)

        # Find two minima to the left and right of the max
        left_minima = minima_indices[minima_indices < maxima]
        right_minima = minima_indices[minima_indices > maxima]

        closest_left_min = left_minima[-1]
        closest_right_min = right_minima[0]
        center_idx = (closest_left_min + closest_right_min) // 2
        highest_minima_idx = center_idx + averaging_amount // 2


    else:
        averaging_amount = 20
        averaged_data = moving_average(data, averaging_amount)
        minima_indices = argrelmin(averaged_data, order=ARG_EXTREME_ORDER)[0]
        minima_indices_without_bounds = minima_indices[1:-1]

        higest_minima_averaging_index = minima_indices_without_bounds[
            np.argmax(averaged_data[minima_indices_without_bounds])]
        center_idx = higest_minima_averaging_index
        highest_minima_idx = center_idx + averaging_amount // 2

    data = center_around(data, highest_minima_idx)
    data = fit_to_find_center(data)
    data = fine_tune_center(data)
    return data


def fine_tune_center(data):
    averaging_amount = 150
    try:
        averaged_data = moving_average(data, averaging_amount)
    except ValueError:
        pass
    minima_indices = argrelmin(averaged_data, order=ARG_EXTREME_ORDER)[0]
    if not minima_indices.any():
        return data
    elif len(minima_indices) >= 5:
        minima_indices_without_bounds = minima_indices[1:-1]
    else:
        minima_indices_without_bounds = minima_indices
    minimas_center_averages = int(np.average(minima_indices_without_bounds))

    maxima_indices = argrelmax(averaged_data, order=ARG_EXTREME_ORDER)[0]
    maxima_indices_without_bounds = maxima_indices[1:-1]
    if not maxima_indices_without_bounds.any():
        return data
    maximas_center_averages = int(np.average(maxima_indices_without_bounds))
    center_averages = (minimas_center_averages + maximas_center_averages) // 2
    data = center_around(data, center_averages + averaging_amount // 2)
    return data


def center_around(data, highest_minima_idx):
    radius = min((len(data)) - highest_minima_idx, highest_minima_idx)
    data = data[highest_minima_idx - radius:highest_minima_idx + radius + 1]
    return data


def fit_to_find_center(data: np.ndarray):
    average_window_size = 100
    averaged_data = moving_average(data, average_window_size)
    result = minimize(compare_hand_sides, np.array(len(averaged_data) // 2), averaged_data, )
    optimal_center = int(result.x[0]) + average_window_size // 2
    return center_around(data, optimal_center)


def compare_hand_sides(center, data):
    center = int(center)
    assumed_data = center_around(data, center)
    if len(assumed_data) % 2:
        assumed_data = assumed_data[:-1]
    assumed_raduis = len(assumed_data) // 2
    lhs, rhs = assumed_data[:assumed_raduis], assumed_data[assumed_raduis:]
    loss = (rhs - lhs[::-1]) ** 2
    loss_threshold = np.percentile(loss, 90)

    # Filter the array for values in the top 90%
    bottom_losses = loss[loss < loss_threshold]
    return np.mean(bottom_losses)
