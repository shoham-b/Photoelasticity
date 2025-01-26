# see Eq.(21)--(26) of : 10.1103/PhysRevE.98.012905
import logging
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
            exit(3)
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


def draw_graphs(forces, title=""):
    forces = forces / np.average(forces)  # make the forces unitless
    if np.var(forces) == 0:
        logging.warn(f"Variance of forces is 0, skipping drawing graphs for {title}")
        return
    z = forces.shape[0]
    (lambda_a, lambda_b, variance) = find_force_dist_coeffs(forces)

    # draw the cumulative density functions of the unitless forces:
    data_xs = np.sort(np.array(forces))
    data_ys = np.array([float(i + 1) / z for i in range(z)])

    print(data_ys / predicted_CDF(data_xs, lambda_a, lambda_b))
    # delta=np.sqrt(np.average((data_ys/predicted_CDF(data_xs,lambda_a,lambda_b) - 1)**2))

    plt.plot(data_xs, data_ys, '.')
    xs = np.arange(0.0, 5.0, 0.02)
    plt.plot(xs, predicted_CDF(xs, lambda_a, lambda_b), 'r--')
    plt.xlabel('Unitless force')
    plt.ylabel('Cumulative distribution function')
    plt.title(title)
    plt.text(3.5, 0.1, "σ²=" + str(round(variance, 2)), fontsize=15)  # +"\n"+"δ="+str(round(delta,2)))
    main_image_dir = Path("../../../forces_graphs").resolve()
    main_image_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(main_image_dir / f"{title}.png"))
