import logging
from pathlib import Path

import diskcache
import matlab.engine
import numpy as np
from PIL import Image

from photoelasticity.forces.force_statistics import draw_graphs
from photoelasticity.tools.matlab import start_matlab

pi = np.pi
fsigma = 6265

#
# forces0 = matlab.double([50, 50, 30])
# rm = matlab.double(0.03)
# beta = matlab.double([0.6 * pi / 2, pi + 0.2 * pi / 2, 3 * pi / 2])
# z = 3
force_cache = diskcache.Cache("../../../force_cache")


@force_cache.memoize()
def solve_disk(image_path, forces_guess, angles, radius):
    z = len(angles)
    if z == 0:
        return
    with start_matlab() as eng:
        matlabed_forces_guess = matlab.double(forces_guess)
        matlabed_angles = matlab.double(angles)
        matlabed_radius = matlab.double(float(radius) / 100.0)
        matlabed_fsigma = matlab.double(float(fsigma))

        matlabed_path = str(image_path)
        try:
            (forces, alphas, image) = eng.customDiskSolver(matlabed_forces_guess,
                                                           matlabed_angles, matlabed_fsigma,
                                                           matlabed_radius, z,
                                                           matlabed_path, nargout=3)
        except matlab.engine.MatlabExecutionError:
            logging.warn(f"Matlab failed to solve disk for {image_path}")
            return (None, None, None)
    image = np.array(image)
    pythoned_forces = np.array(forces)[0]
    pythoned_angles = np.array(alphas)[0]

    return image, pythoned_forces, pythoned_angles


def solve_multiple_disks(circles_image_paths, circle_radiuses, neighbour_circles_angle, ignore_images):
    if not circles_image_paths:
        return
    full_image_path = Path(circles_image_paths[0]).parent
    name = full_image_path.name
    final_images_dir = Path(rf"{__file__}/../../../final_images/{name}").resolve()
    final_images_dir.mkdir(exist_ok=True, parents=True)

    all_forces = []
    all_angles = []
    for i, image_path in enumerate(circles_image_paths):
        if i in ignore_images:
            continue
        image_path = Path(image_path)
        angles = neighbour_circles_angle[i]

        logging.info(f"""Solving disk for image {image_path.stem}
    with angles {[np.round(angle / np.pi, 4) for angle in angles]}""")
        if len(angles) > 1:
            (img_final, forces, alphas) = solve_disk(image_path, [4500] * len(angles), angles,
                                                     circle_radiuses[i])
            if forces is None:
                continue
            all_forces.append(forces)
            all_angles.append(alphas)
            Image.fromarray(img_final).convert("L").save(str(final_images_dir / f"{i}.jpeg"))

    tangent_forces = [force * np.sin(angle) for force, angle in zip(all_forces, all_angles)]
    normal_forces = [force * np.cos(angle) for force, angle in zip(all_forces, all_angles)]


    flat_normal_forces =np.abs(np.array([force for sublist in normal_forces for force in sublist]))
    flat_tangent_forces =np.abs( np.array([force for sublist in tangent_forces for force in sublist]))

    top_normal_forces = flat_normal_forces[flat_tangent_forces>np.percentile(flat_tangent_forces, 0)]
    top_tangent_forces = flat_tangent_forces[flat_tangent_forces>np.percentile(flat_tangent_forces, 0)]

    logging.info(f"Normal forces: {flat_normal_forces.tolist()}")
    logging.info(f"Tangent forces: {flat_tangent_forces.tolist()}")

    title = f"{name} normal forces"
    draw_graphs(top_normal_forces, title)
    title = f"{name} tangent forces"
    draw_graphs(top_tangent_forces, title)
    return
