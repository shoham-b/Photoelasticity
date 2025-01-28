# see Eq.(21)--(26) of : 10.1103/PhysRevE.98.012905
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy

np.seterr(all='raise')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 13


def find_y(sig_sq):  # invert Eq.(24)
    def err(y):
        try:
            expr1 = np.exp(-y ** 2) / (np.sqrt(np.pi) * scipy.special.erfc(y))
            err = np.abs((y ** 2 + 0.5 - expr1 * y) / ((expr1 - y) ** 2) - 1 - sig_sq)
        except Exception as e:
            print("sigma squared =", sig_sq)
            print("Search for y(sigma) failed, the variance might be too big")
            raise
        return err

    root_result = scipy.optimize.root(err, [0])
    return root_result.x[0]  # return the value of y we found from the root search


def find_force_dist_coeffs(forces):
    # input: list of unitless forces (already divided by their average)
    # calculate lambda_a and lambda_b for either the normal or tangential forces
    # note: drawing sig(y) in Desmos, if the variance is greater than 1, then Eq.(24) suddenly becomes very noisy and possibly not one-to-one

    variance = np.var(forces)  # the standard deviation of the now-unitless forces
    y = find_y(variance)
    lambda_b = (-y + (np.exp(-y ** 2) / (np.sqrt(np.pi) * scipy.special.erfc(y)))) ** 2
    lambda_a = 2 * y * np.sqrt(lambda_b)
    return (lambda_a, lambda_b, variance)


def predicted_CDF(x, lambda_a, lambda_b):
    expr1 = lambda_a / (2 * np.sqrt(lambda_b))
    return ((scipy.special.erf((lambda_a + 2 * lambda_b * x) / (2 * np.sqrt(lambda_b))) - scipy.special.erf(expr1)) /
            (scipy.special.erfc(expr1)))


def draw_graphs(forces, title="", force_err=0.2):
    # forces = (forces - np.min(forces))
    # forces = forces/np.max(forces)
    forces = forces / np.mean(forces)
    z = forces.shape[0]
    data_xs = np.sort(np.array(forces))
    data_ys = np.array([float(i + 1) / z for i in range(z)])
    # plt.plot(data_xs,data_ys,'.',markersize=15)
    plt.errorbar(data_xs, data_ys, xerr=force_err * data_xs, linestyle='None', marker='.', markersize=15, capsize=3)

    xs = np.arange(0.0, 5.0, 0.02)
    (lambda_a, lambda_b, variance) = find_force_dist_coeffs(forces)
    plt.plot(xs, predicted_CDF(xs, lambda_a, lambda_b), 'r--', linewidth=3)
    plot_text = "σ²=" + str(round(variance, 2)) + "\nlambda_a=" + str(round(lambda_a, 2)) + "\nlambda_b=" + str(
        round(lambda_b, 2))
    plt.text(2, 0.1, plot_text)

    plt.xlabel('Unitless force')
    plt.ylabel('CDF')
    plt.title(title)
    plt.subplots_adjust(bottom=0.15, left=0.15)

    save_plot(title)


def save_plot(title):
    main_image_dir = Path(rf"{__file__}/../../../forces_graphs").resolve()
    main_image_dir.mkdir(parents=True, exist_ok=True)
    graph_path = main_image_dir / f"{title}.png"
    plt.savefig(str(graph_path))
    plt.close()
