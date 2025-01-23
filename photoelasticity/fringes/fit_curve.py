import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.optimize import Bounds
from scipy.signal import argrelmax

from photoelasticity.tools.matrix_tools import find_center_strip

TITLE_FONT_SIZE = 15
AXIS_TEXT_FONT_SIZE = 13


class FitError(Exception):
    pass


def make_assumed_function(radius):
    def assumed_function_without_offset(x, I0, A, offset=0):
        zeta = (x + offset) / radius
        zeta_stuff = (1 - zeta ** 2) / (1 + zeta ** 2) ** 2
        return I0 * np.sin(A * zeta_stuff) ** 2

    return assumed_function_without_offset


def find_fit_params(data: np.array, image_title: str, guess=[50, 10, 0], data_amount_percent=1):
    radius = len(data) // 2
    center_strip = find_center_strip(data)

    data_for_fit = center_strip[int(0.1 * len(center_strip)):int(0.9 * len(center_strip))]
    data_for_fit = data_for_fit - min(data_for_fit)

    print(f"For image title {image_title}")
    relative_indices = np.array(range(-len(data_for_fit) // 2, len(data_for_fit) // 2))

    data_amount = int(data_amount_percent * len(data_for_fit))
    data_for_fit = data_for_fit[:data_amount]
    relative_indices = relative_indices[:data_amount]
    assumed_function_without_offset = make_assumed_function(radius)
    quarter_way = len(relative_indices) // 4

    # min_offset = relative_indices[quarter_way]
    # max_offset = max(relative_indices[int(0.75 * quarter_way)], relative_indices[-1]) + 1
    # try:
    #     results, _ = scipy.optimize.curve_fit(assumed_function_without_offset,
    #                                           relative_indices,
    #                                           data_for_fit,
    #                                           p0=[50, guessA, 0],
    #                                           bounds=((30, 1, min_offset),
    #                                                   (70, guessA * 3, max_offset),),
    #                                           maxfev=5000)
    # except ValueError:
    #     raise FitError()
    try:
        results, pcov = scipy.optimize.curve_fit(assumed_function_without_offset,
                                                 relative_indices,
                                                 data_for_fit,
                                                 p0=guess,
                                                 bounds=((5, 0, guess[2] - 100),
                                                         (80, 100, guess[2] + 100),),
                                                 maxfev=5000)
    except ValueError:
        raise FitError()
    I0, A, offset_f = results
    fitted_data = assumed_function_without_offset(relative_indices, *results)
    errs = np.sqrt(np.diag(pcov))

    plot_figure(A, errs, data_for_fit, image_title, relative_indices, fitted_data)


def get_maxima_count(data):
    return len(argrelmax(data, order=40)[0]) // 2


def plot_figure(A, errs, data, image_title, relative_indices, fitted_data):
    I0_err, A_err, offset_f_err = errs
    data = data[:len(relative_indices)]
    title = f"Brightness of {image_title}"
    plt.figure(dpi=400)
    plt.suptitle(title, fontsize=TITLE_FONT_SIZE)
    plt.plot(relative_indices, data, label="Measured data", linewidth=0.5)
    plt.plot(relative_indices, fitted_data, label=f"Fitted (A={A:.3f})", linewidth=0.5)
    plt.legend(loc='upper left')
    plt.xlabel("Distance from center [pixel]", fontsize=AXIS_TEXT_FONT_SIZE)
    plt.ylabel("Brightness percentage", fontsize=AXIS_TEXT_FONT_SIZE)

    # plt.figtext(0.01, 0.01, f"Fit error: A:{A_err:.3f}, I0:{I0_err:.3f}, offset:{offset_f_err:.3f}")

    plt.savefig(fr"{__file__}/../../../graphs/{title.replace('.', ' ')}.png")
