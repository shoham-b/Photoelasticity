# see Eq.(21)--(26) of : 10.1103/PhysRevE.98.012905
import matplotlib.pyplot as plt
import numpy as np
import scipy

np.seterr(all='raise')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 13


def _find_y(sig_sq):  # invert the ridiculous looking Eq.(24)
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
    y = _find_y(variance)
    lambda_b = (-y + (np.exp(-y ** 2) / (np.sqrt(np.pi) * scipy.special.erfc(y)))) ** 2
    lambda_a = 2 * y * np.sqrt(lambda_b)
    return (lambda_a, lambda_b, variance)


def predicted_CDF(x, lambda_a, lambda_b):
    expr1 = lambda_a / (2 * np.sqrt(lambda_b))
    return ((scipy.special.erf((lambda_a + 2 * lambda_b * x) / (2 * np.sqrt(lambda_b))) - scipy.special.erf(expr1)) /
            (scipy.special.erfc(expr1)))


def draw_graphs(forces, title=""):
    forces = forces / np.average(forces)  # make the forces unitless
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
    plt.show()


arr = [np.float64(2.048219007168245), np.float64(2.7896630316369686), np.float64(4.64716394735462),
       np.float64(1.8299456213711385), np.float64(4.0493553231907935), np.float64(2.7977939737457502),
       np.float64(3.5614622652055568), np.float64(1.491668491676309), np.float64(0.4364208967397615),
       np.float64(3.1181980479685367), np.float64(2.3287719272817924), np.float64(3.940831790721589),
       np.float64(2.122365005996994), np.float64(2.7026907761238417), np.float64(3.6945258501514555),
       np.float64(2.822152483030359), np.float64(2.3282125929144875), np.float64(2.2316385216888706),
       np.float64(2.8737743084248555), np.float64(1.559443865083308)]
draw_graphs(np.array(arr))
