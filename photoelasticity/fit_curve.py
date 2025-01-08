import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.optimize import Bounds
from scipy.signal import argrelmax

from photoelasticity.matrix_tools import find_center_strip


class FitError(Exception):
    pass


def assumed_function_with_offset(x, I0, A, offset):
    radius = len(x) // 2
    zeta = x / radius
    zeta_stuff = (1 - zeta ** 2) / (1 + zeta ** 2) ** 2
    return I0 * np.sin(A * zeta_stuff + offset) ** 2


def make_assumed_function(radius):
    def assumed_function_without_offset(x, I0, A, offset=0):
        offset = int(offset)
        zeta = (x + offset) / radius
        zeta_stuff = (1 - zeta ** 2) / (1 + zeta ** 2) ** 2
        return I0 * np.sin(A * zeta_stuff) ** 2

    return assumed_function_without_offset


def find_fit_params(data: np.array, image_title: str, guessA=None, data_amount_percent=1):
    radius = len(data) // 2
    center_strip = find_center_strip(data)

    data_for_fit = center_strip[int(0.1 * len(center_strip)):int(0.9 * len(center_strip))]
    data_for_fit = data_for_fit - min(data_for_fit)

    print(f"For image title {image_title}")
    relative_indices = np.array(range(-len(data_for_fit) // 2, len(data_for_fit) // 2))

    data_amount = int(data_amount_percent * len(data_for_fit))
    data_for_fit = data_for_fit[:data_amount]
    relative_indices = relative_indices[:data_amount]
    if guessA is None:
        maximas_count = get_maximas_count(center_strip) - 1
        guessA = maximas_count * np.pi
    assumed_function_without_offset = make_assumed_function(radius)

    quarter_way = len(relative_indices) // 4

    min_offset = relative_indices[quarter_way]
    max_offset = max(relative_indices[int(0.75 * quarter_way)], relative_indices[-1]) + 1
    try:
        results, _ = scipy.optimize.curve_fit(assumed_function_without_offset,
                                              relative_indices,
                                              data_for_fit,
                                              p0=[50, guessA, 0],
                                              bounds=((30, 1, min_offset),
                                                      (70, guessA * 3, max_offset),),
                                              maxfev=5000)
    except ValueError:
        raise FitError()
    I0, A, *_ = results
    fitted_data = assumed_function_without_offset(relative_indices, *results)

    print(f"""
    fitted result is:
        I0:{I0}
        A:{A}
        guessed A: {guessA}
""")
    plot_figure(A, data_for_fit, image_title, relative_indices, fitted_data)


def get_maximas_count(data):
    maximas_count = len(argrelmax(data, order=40)[0]) // 2
    return maximas_count


def plot_figure(A, data, image_title, relative_indicies, fitted_data):
    data = data[:len(relative_indicies)]
    title = f"Brightness of {image_title}"
    plt.figure(dpi=400)
    plt.suptitle(title, fontsize=10)
    plt.plot(relative_indicies, data, label="Measured data", linewidth=0.5)
    plt.plot(relative_indicies, fitted_data, label=f"Fitted (A={A:.3f})", linewidth=0.5)
    plt.legend(loc='upper right')
    plt.xlabel("Distance from center [pixel]")
    plt.ylabel("Brightness percentage [unitless]")
    plt.show()
