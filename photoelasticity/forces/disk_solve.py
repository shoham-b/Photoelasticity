import logging
import math
from pathlib import Path
from pprint import pprint

import diskcache
import matlab.engine
import matplotlib.pyplot as plt
import numpy as np

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
force_cache = diskcache.Cache(fr"{__file__}/../../../force_cache")


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
    image = np.array(image).tolist()
    pythoned_forces = np.array(forces)[0].tolist()
    pythoned_angles = np.array(alphas)[0].tolist()

    return image, pythoned_forces, pythoned_angles


def solve_multiple_disks(circles_image_paths, circle_radiuses, neighbour_circles_angle, ignore_images):
    if not circles_image_paths:
        return
    full_image_path = Path(circles_image_paths[0]).parent
    name = full_image_path.name

    print(f"neighbour_circles_angle: {[[angle.tolist() for angle in angles] for angles in neighbour_circles_angle]}")
    print(f"image 11 angles {neighbour_circles_angle[11]}")
    all_forces, all_angles, all_images = find_all_disk_forces(circle_radiuses, circles_image_paths, name,
                                                              neighbour_circles_angle, ignore_images)
    print(f"format - force,angle,radius")
    pprint(list(zip(all_forces, all_angles, circle_radiuses)))
    flat_forces = [force for sublist in all_forces for force in sublist]
    flat_angles = [angle for sublist in all_angles for angle in sublist]
    normal_forces = np.abs(np.array([force * math.sin(angle) for force, angle in zip(flat_forces, flat_angles)]))
    tangent_forces = np.abs(np.array([force * math.cos(angle) for force, angle in zip(flat_forces, flat_angles)]))

    logging.info(f"Unfiltered flat forces: {flat_forces}")
    logging.info(f"Unfiltered flat normal forces: {normal_forces}")
    logging.info(f"Unfiltered flat tangent forces: {tangent_forces}")

    # force_threshold = np.percentile(flat_forces, 10)
    #
    # filtered_normal_forces = normal_forces[flat_forces > force_threshold]
    # filtered_tangent_forces = tangent_forces[flat_forces > force_threshold]
    #
    # logging.info(f"Normal forces: {filtered_normal_forces.tolist()}")
    # logging.info(f"Tangent forces: {filtered_tangent_forces.tolist()}")

    try:
        title = f"Normal forces"
        draw_graphs(normal_forces, title)
    except FloatingPointError:
        logging.error(f"Failed to draw graph for {title}")
    try:
        title = f"Tangent forces"
        draw_graphs(tangent_forces, title)
    except FloatingPointError:
        logging.error(f"Failed to draw graph for {title}")
    return


def find_all_disk_forces(circle_radiuses, circles_image_paths, name, neighbour_circles_angle, ignore_images):
    final_images_dir = Path(rf"{__file__}/../../../final_images/{name}").resolve()
    final_images_dir.mkdir(exist_ok=True, parents=True)

    valid_indices = [i for i in range(len(circles_image_paths)) if
                     i not in ignore_images and neighbour_circles_angle[i]]
    valid_angles = [neighbour_circles_angle[i] for i in valid_indices]
    valid_image_paths = [circles_image_paths[i] for i in valid_indices]
    valid_radiuses = [circle_radiuses[i] for i in valid_indices]

    results = [find_single_disk(angles, final_images_dir / f"{i}.png", Path(image_path), radius)
               for i, angles, image_path, radius in zip(valid_indices, valid_angles, valid_image_paths, valid_radiuses)]

    all_forces, all_angles, all_images = zip(*results)
    return all_forces, all_angles, all_images


def find_single_disk(angles, final_image_path, image_path, radius):
    logging.info(f"""Solving disk for image {image_path.stem}
    with angles {[np.round(angle / np.pi, 4).tolist() for angle in angles]}""")
    (img_final, forces, alphas) = solve_disk(image_path, [1700] * len(angles), angles,
                                             radius)
    plt.imsave(final_image_path, img_final)
    return alphas, forces, img_final
