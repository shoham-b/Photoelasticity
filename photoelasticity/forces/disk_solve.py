import logging
from pathlib import Path

import cv2
import matlab.engine
import numpy as np

from photoelasticity.forces.force_statistics import draw_graphs
from photoelasticity.tools.matlab import start_matlab

pi = np.pi
fsigma = 6265


#
# forces0 = matlab.double([50, 50, 30])
# rm = matlab.double(0.03)
# beta = matlab.double([0.6 * pi / 2, pi + 0.2 * pi / 2, 3 * pi / 2])
# z = 3


def solve_disk(image_path, forces_guess, angles, radius):
    z = len(angles)
    if z == 0:
        return
    with start_matlab() as eng:
        matlabed_forces_guess = matlab.double(forces_guess)
        matlabed_angles = matlab.double(angles.tolist())
        matlabed_radius = matlab.double(float(radius) / 100.0)
        matlabed_fsigma = matlab.double(float(fsigma))

        matlabed_path = str(image_path)
        try:
            (forces, alphas, img_final) = eng.customDiskSolver(matlabed_forces_guess,
                                                               matlabed_angles, matlabed_fsigma,
                                                               matlabed_radius, z,
                                                               matlabed_path, nargout=3)
        except matlab.engine.MatlabExecutionError:
            logging.warn(f"Matlab failed to solve disk for {image_path}")
            return (None, None, None)
    img_final = np.array(img_final)
    return forces, alphas, img_final


def solve_multiple_disks(circles_image_paths, circle_radiuses, neighbour_circles_angle):
    for i, image_path in enumerate(set(circles_image_paths)):
        image_path = Path(image_path)
        angles = neighbour_circles_angle[i]
        angles = (angles[~np.isnan(angles)] + np.pi) % (2 * np.pi)
        if len(angles) > 1:
            (forces, alphas, img_final) = solve_disk(image_path, [4500] * len(angles), angles,
                                                     circle_radiuses[i])
            if forces is None:
                continue
            full_image_path = image_path.parent
            finals_dir = Path(fr"{__file__}/../../../finals/{full_image_path.name}").resolve()
            finals_dir.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(finals_dir / f"{i}.jpg"), img_final)
            logging.info(f"""For image {image_path.parent.name} of number {image_path.stem}:
            Forces: {forces}
            Alphas: {alphas}
            """)
            draw_graphs(forces, f"Forces map for {i} of {full_image_path.name}")
