import matlab.engine
import numpy as np

from photoelasticity.tools.matlab import start_matlab

pi = np.pi
fsigma = matlab.double(6265)

forces0 = matlab.double([50, 50, 30])
rm = matlab.double(0.03)
beta = matlab.double([0.6 * pi / 2, pi + 0.2 * pi / 2, 3 * pi / 2])
z = 3


def solve_disk(image, forces_guess, beta, fsigma, rm, z):
    with start_matlab() as eng:
        (forces, alphas, img_final) = eng.customDiskSolver(forces_guess, beta, fsigma, rm, z, image, nargout=3)
    img_final = np.array(img_final)
