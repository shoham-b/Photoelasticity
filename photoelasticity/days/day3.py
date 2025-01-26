import numpy as np

from photoelasticity.days.data import get_day_data
from photoelasticity.forces.disk_solve import solve_disk
from photoelasticity.image_detection.image_detection import extract_multiple_circles_and_count_stripes
from photoelasticity.tools.multiprocessing import with_pool


def do_day_3():
    day_data = get_day_data(3)
    column_data, box_data = day_data[:9], day_data[9:]
    use_cache = True
    dp = 2.3

    with with_pool() as pool:
        # pool.starmap(run_column, [(data_path, use_cache, dp) for data_path in column_data])
        pool.starmap(run_box, [(data_path, use_cache, dp) for data_path in box_data])


def run_box(data_path, use_cache, dp):
    circles_images, circle_radiuses, neighbour_circles_angle = extract_multiple_circles_and_count_stripes(data_path,
                                                                                                          0.125, 0.34,
                                                                                                          use_cache=use_cache,
                                                                                                          dp=dp)
    for i, image in enumerate(circles_images):
        angles = neighbour_circles_angle[i]
        angles = angles[~np.isnan(angles)]
        if angles.any():
            solve_disk(image, [150 / len(angles)] * len(angles), angles, 10.0, circle_radiuses[i])
    return


def run_column(data_path, use_cache, dp):
    images, radius, angles = extract_multiple_circles_and_count_stripes(data_path, 0.41, 0.45,
                                                                        use_cache=use_cache, dp=dp)


if __name__ == '__main__':
    do_day_3()
