import logging
from pathlib import Path

from photoelasticity.days.data import get_day_data
from photoelasticity.forces.disk_solve import solve_multiple_disks
from photoelasticity.image_detection.image_detection import extract_multiple_circles_and_count_stripes
from photoelasticity.tools.multiprocessing import with_pool


def do_day_4(use_cache=True):
    day_data = get_day_data(4)
    with with_pool() as pool:
        pool.starmap(run_image_detection, [(data_path, use_cache) for data_path in day_data])


def do_special_day_4(use_cache=True):
    logging.basicConfig(level=logging.INFO)
    data_path = Path(r"C:\Users\shoha\PycharmProjects\Photoelasticity\data\day4\DSC_0021.jpg")

    ignore_disks = {3, 13}
    ignore_negibhors = ((11, 15))
    images, radius, angles = extract_multiple_circles_and_count_stripes(data_path,
                                                                        0.15, 0.31,
                                                                        use_cache=use_cache, dp=1.5,
                                                                        ignore_disks=ignore_disks)
    ignore_disks = {0, 1, 7, 14, 15, 17}
    logging.info(f"Solving disk for image {data_path.stem}")
    logging.info(f"all radiuses: {radius}")
    # solve_multiple_disks(images, radius, angles, ignore_disks)


def regenerate_day4_cache():
    day_data = get_day_data(4)
    with with_pool() as pool:
        pool.starmap(extract_multiple_circles_and_count_stripes,
                     [(data_path, 0.15, 0.31, False, 1.5) for data_path in day_data])


def run_image_detection(data_path, use_cache):
    images, radius, angles = extract_multiple_circles_and_count_stripes(data_path, 0.15, 0.31,
                                                                        use_cache=use_cache, dp=1.5)

    solve_multiple_disks(images, radius, angles)


if __name__ == '__main__':
    do_special_day_4()
