import matlab.engine
import numpy as np

from photoelasticity.tools.matlab import start_matlab


# pi = np.pi
# fsigma = matlab.double(6265)
#
# forces0 = matlab.double([50, 50, 30])
# rm = matlab.double(0.03)
# beta = matlab.double([0.6 * pi / 2, pi + 0.2 * pi / 2, 3 * pi / 2])
# z = 3


def solve_disk(image_path, forces_guess, angles, fsigma, radius):
    z = len(angles)
    if z == 0:
        return
    with start_matlab() as eng:
        matlabed_forces_guess = matlab.double(forces_guess)
        matlabed_angles = matlab.double(angles.tolist())
        matlabed_radius = matlab.double(float(radius))
        matlabed_fsigma = matlab.double(float(fsigma))
        try:
            (forces, alphas, img_final) = eng.customDiskSolver(matlabed_forces_guess,
                                                               matlabed_angles, matlabed_fsigma,
                                                               matlabed_radius, z,
                                                               image_path, nargout=3)
        except matlab.engine.MatlabExecutionError:
            return
    img_final = np.array(img_final)
