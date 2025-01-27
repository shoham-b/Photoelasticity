import logging
from pathlib import Path

import diskcache
import matlab.engine
import numpy as np
from PIL import Image

from photoelasticity.forces.force_statistics import draw_graphs
from photoelasticity.tools.matlab import start_matlab

np.set_printoptions(precision=3)

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

    all_angles, all_forces = find_all_disk_forces(circle_radiuses, circles_image_paths, name,
                                                  neighbour_circles_angle, ignore_images)

    tangent_forces = [force * np.sin(angle) for force, angle in zip(all_forces, all_angles)]
    normal_forces = [force * np.cos(angle) for force, angle in zip(all_forces, all_angles)]

    print(f"Normal forces: {normal_forces}")
    print(f"Tangent forces: {tangent_forces}")

    flat_normal_forces = np.abs(np.array([force for sublist in normal_forces for force in sublist]))
    flat_normal_forces = flat_normal_forces[~np.isclose(flat_normal_forces, 0.0)]  # filter out the forces that are close to 0
    flat_tangent_forces = np.abs(np.array([force for sublist in tangent_forces for force in sublist]))
    flat_tangent_forces = flat_tangent_forces[~np.isclose(flat_tangent_forces, 0.0)]  # filter out the forces that are close to 0

    normal_forces_threshold = np.average(flat_normal_forces) * 0.1
    tangent_forces_threshold = np.average(flat_tangent_forces) * 0.2
    top_normal_forces = flat_normal_forces[flat_normal_forces > normal_forces_threshold]
    top_tangent_forces = flat_tangent_forces[flat_tangent_forces > tangent_forces_threshold]

    logging.info(f"Normal forces: {flat_normal_forces.tolist()}")
    logging.info(f"Tangent forces: {flat_tangent_forces.tolist()}")

    title = f"{name} normal forces"
    draw_graphs(top_normal_forces, title)
    title = f"{name} tangent forces"
    draw_graphs(top_tangent_forces, title)
    return


def find_all_disk_forces(circle_radiuses, circles_image_paths, name, neighbour_circles_angle, ignore_images):
    final_images_dir = Path(rf"{__file__}/../../../final_images/{name}").resolve()
    final_images_dir.mkdir(exist_ok=True, parents=True)

    all_forces = []
    all_angles = []
    for i, image_path in enumerate(circles_image_paths):
        if i in ignore_images:
            continue
        angles = neighbour_circles_angle[i]
        if not angles:
            continue
        alphas, forces = find_single_disk(angles, final_images_dir / f"{i}.jpeg", Path(image_path),
                                          circle_radiuses[i])

        all_forces.append(forces)
        all_angles.append(alphas)

    return all_angles, all_forces


def find_single_disk(angles, final_image_path, image_path, radius):
    logging.info(f"""Solving disk for image {image_path.stem}
    with angles {[np.round(angle / np.pi, 4).tolist() for angle in angles]}""")
    (img_final, forces, alphas) = solve_disk(image_path, [1700] * len(angles), angles,
                                             radius)
    Image.fromarray(img_final).convert("L").save(str(final_image_path))
    return alphas, forces
