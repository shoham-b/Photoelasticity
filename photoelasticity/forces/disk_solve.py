from pathlib import Path

import matlab.engine
import numpy as np
from imageio.v2 import imwrite

from photoelasticity.tools.matlab import start_matlab


# pi = np.pi
# fsigma = matlab.double(6265)
#
# forces0 = matlab.double([50, 50, 30])
# rm = matlab.double(0.03)
# beta = matlab.double([0.6 * pi / 2, pi + 0.2 * pi / 2, 3 * pi / 2])
# z = 3


def solve_disk(image_path, forces_guess, angles, fsigma, radius ):
    z = len(angles)
    if z == 0:
        return
    with start_matlab() as eng:
        matlabed_forces_guess = matlab.double(forces_guess)
        matlabed_angles = matlab.double(angles.tolist())
        matlabed_radius = matlab.double(float(radius))
        matlabed_fsigma = matlab.double(float(fsigma))

        (forces, alphas, img_final) = eng.customDiskSolver(matlabed_forces_guess,
                                                           matlabed_angles, matlabed_fsigma,
                                                           matlabed_radius, z,
                                                           image_path, nargout=3)
    img_final = np.array(img_final)
    return forces, alphas, img_final


def solve_multiple_disks(circles_image_paths, circle_radiuses, neighbour_circles_angle):
    for i, image_path in enumerate(circles_image_paths):
        angles = neighbour_circles_angle[i]
        angles = angles[~np.isnan(angles)] + np.pi
        if angles.any():
            (forces, alphas, img_final) = solve_disk(image_path, [150 / len(angles)] * len(angles), angles, 10.0,
                                                     circle_radiuses[i])
            finals_dir = Path(fr"{__file__}/../../../finals/{image_path.stem}").resolve()
            imwrite(str(finals_dir / f"{i}.jpg"), img_final)
