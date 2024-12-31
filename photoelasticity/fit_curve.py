import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.signal import argrelmin, argrelmax

from photoelasticity.processing import center_data_around_radius


def assumed_function(x, I0, A):
    radius = len(x) // 2
    zeta = x / radius
    zeta_stuff = (1 - zeta ** 2) / (1 + zeta ** 2) ** 2
    return I0 * np.sin(A * zeta_stuff) ** 2


def find_fit_params(data: np.array, image_title: str):
    print(f"For image title {image_title}")
    data = center_data_around_radius(data)
    relative_indicies = np.array(range(-len(data) // 2, len(data) // 2))
    assert len(relative_indicies) == len(data)
    minimas_count = len(argrelmax(data, order=50)[0])+5
    results, _ = scipy.optimize.curve_fit(assumed_function, relative_indicies, data, p0=[90, minimas_count])
    I0, A, = results

    print(f"""
    fitted result is:
        I0:{I0}
        A:{A}
""")
    title = f"fit for {image_title}"

    plt.figure(dpi=400)
    plt.suptitle(title, fontsize=10)

    fitted_data = assumed_function(relative_indicies, *results)

    plt.plot(relative_indicies, data, label="Measured data", linewidth=0.5)
    plt.plot(relative_indicies, fitted_data, label=f"Fitted (A={A:.3f})", linewidth=0.5)
    plt.legend(loc='upper right')
    plt.xlabel("Distance from center [pixel]")
    plt.ylabel("Intensity [volt]")
    plt.show()
