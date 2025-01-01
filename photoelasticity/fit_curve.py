import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.signal import argrelmax

from photoelasticity.data_processing import moving_average, process_data


def assumed_function_with_offset(x, I0, A, offset):
    radius = len(x) // 2
    zeta = x / radius
    zeta_stuff = (1 - zeta ** 2) / (1 + zeta ** 2) ** 2
    return I0 * np.sin(A * zeta_stuff + offset) ** 2


def assumed_function_without_offset(x, I0, A):
    radius = len(x) // 2
    zeta = x / radius
    zeta_stuff = (1 - zeta ** 2) / (1 + zeta ** 2) ** 2
    return I0 * np.sin(A * zeta_stuff) ** 2


def find_fit_params(data: np.array, image_title: str, override_maximas_count=None):
    print(f"For image title {image_title}")
    relative_indicies = np.array(range(-len(data) // 2, len(data) // 2))
    assert len(relative_indicies) == len(data)

    if override_maximas_count is None:
        maximas_count = get_maximas_count(data)
        guess_ = [90, maximas_count]
        assumed_function = assumed_function_without_offset
    else:
        maximas_count = override_maximas_count
        offset_guess = 50
        guess_ = [90, maximas_count, offset_guess]
        assumed_function = assumed_function_with_offset

    results, _ = scipy.optimize.curve_fit(assumed_function, relative_indicies, data,
                                          p0=guess_)
    I0, A, *_ = results

    print(f"""
    fitted result is:
        I0:{I0}
        A:{A}
""")
    plot_figure(A, data, image_title, relative_indicies, results)


def get_maximas_count(data):
    averaged_data = moving_average(data, 30)
    minimas_count = len(argrelmax(averaged_data, order=50)[0]) + 5
    return minimas_count


def plot_figure(A, data, image_title, relative_indicies, results):
    title = f"Fit for {image_title.split('.')[0]}"
    plt.figure(dpi=400)
    plt.suptitle(title, fontsize=10)
    fitted_data = assumed_function_without_offset(relative_indicies, *results)
    plt.plot(relative_indicies, data, label="Measured data", linewidth=0.5)
    plt.plot(relative_indicies, fitted_data, label=f"Fitted (A={A:.3f})", linewidth=0.5)
    plt.legend(loc='upper right')
    plt.xlabel("Distance from center [pixel]")
    plt.ylabel("Brightness percentage [unitless]")
    plt.show()
