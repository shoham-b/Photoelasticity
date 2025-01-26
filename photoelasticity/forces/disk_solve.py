from pathlib import Path

import cv2
import matlab.engine
import numpy as np

from photoelasticity.forces.force_statistics import draw_graphs
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

        matlabed_path = str(image_path)
        (forces, alphas, img_final) = eng.customDiskSolver(matlabed_forces_guess,
                                                           matlabed_angles, matlabed_fsigma,
                                                           matlabed_radius, z,
                                                           matlabed_path, nargout=3)
    img_final = np.array(img_final)
    return forces, alphas, img_final


def solve_multiple_disks(circles_image_paths, circle_radiuses, neighbour_circles_angle):
    for i, image_path in enumerate(circles_image_paths):
        image_path = Path(image_path)
        angles = neighbour_circles_angle[i]
        angles = angles[~np.isnan(angles)] % (2 * np.pi)
        if angles.any():
            (forces, alphas, img_final) = solve_disk(image_path, [150 / len(angles)] * len(angles), angles, 100.0,
                                                     circle_radiuses[i])
            full_image_path = image_path.parent
            finals_dir = Path(fr"{__file__}/../../../finals/{full_image_path.name}").resolve()
            finals_dir.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(finals_dir / f"{i}.jpg"), img_final)
            draw_graphs(forces, f"Forces map for {image_path.stem} of {full_image_path.name}")
