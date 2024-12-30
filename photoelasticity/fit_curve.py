import numpy as np
import scipy
from matplotlib import pyplot as plt


def create_assumed_function_by_radius(radius):
    def assumed_function(x, I0, A):
        zeta = x / radius
        zeta_stuff = (1 - zeta ** 2) / (1 + zeta ** 2) ** 2
        return I0 * np.sin(A * zeta_stuff)**2

    return assumed_function


def find_fit_params(data: np.array, radius: float, image_title: str):
    if len(data) % 2:  # we need a center
        data = data[:-1]
    assumed_intensity_function = create_assumed_function_by_radius(radius)
    relative_indicies = np.array(range(-len(data) // 2, len(data) // 2))
    assert len(relative_indicies) == len(data)
    results,_ = scipy.optimize.curve_fit(assumed_intensity_function, relative_indicies, data,p0=[60,15])

    title = f"fit for {image_title}"

    plt.figure(dpi=400)
    plt.suptitle(title, fontsize=10)

    fitted_data = assumed_intensity_function(relative_indicies, *results)

    plt.plot(relative_indicies, data, label="Measured data", linewidth=0.5)
    plt.plot(relative_indicies, fitted_data, label="Fitted data", linewidth=0.5)
    plt.legend(loc='upper right')
    plt.xlabel("Distance from center [pixel]")
    plt.ylabel("Intensity [volt]")
    plt.show()
