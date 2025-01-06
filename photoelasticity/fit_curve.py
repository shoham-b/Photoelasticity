import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.optimize import Bounds
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


def find_fit_params(data: np.array, image_title: str):
    print(f"For image title {image_title}")
    relative_indicies = np.array(range(-len(data) // 2, len(data) // 2))
    assert len(relative_indicies) == len(data)

    maximas_count = get_maximas_count(data)

    results, _ = scipy.optimize.curve_fit(assumed_function_without_offset,
                                          relative_indicies,
                                          data,
                                          p0=[90, maximas_count],
                                          bounds=((30, 1), (100, 20)),
                                          maxfev=5000)
    I0, A, *_ = results
    fitted_data = assumed_function_without_offset(relative_indicies, *results)

    print(f"""
    fitted result is:
        I0:{I0}
        A:{A}
""")
    plot_figure(A, data, image_title, relative_indicies, fitted_data)


def get_maximas_count(data):
    averaged_data = moving_average(data, 30)
    minimas_count = len(argrelmax(averaged_data, order=50)[0]) + 3
    return minimas_count


def plot_figure(A, data, image_title, relative_indicies, fitted_data):
    title = f"Fit for {image_title.split('.')[0]}"
    plt.figure(dpi=400)
    plt.suptitle(title, fontsize=10)
    plt.plot(relative_indicies, data, label="Measured data", linewidth=0.5)
    plt.plot(relative_indicies, fitted_data, label=f"Fitted (A={A:.3f})", linewidth=0.5)
    plt.legend(loc='upper right')
    plt.xlabel("Distance from center [pixel]")
    plt.ylabel("Brightness percentage [unitless]")
    plt.show()
