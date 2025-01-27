from photoelasticity.days.data import get_day_data
from photoelasticity.forces.disk_solve import solve_multiple_disks
from photoelasticity.image_detection.image_detection import extract_multiple_circles_and_count_stripes
from photoelasticity.tools.multiprocessing import with_pool


def do_day_3(use_cache=True):
    day_data = get_day_data(3)
    column_data, box_data = day_data[:9], day_data[9:]

    dp = 2.3
    with with_pool() as pool:
        pool.starmap(run_column, [(data_path, use_cache, dp) for data_path in column_data])
        pool.starmap(run_box, [(data_path, use_cache, dp) for data_path in box_data])


def regenerate_day3_cache():
    day_data = get_day_data(3)
    column_data, box_data = day_data[:9], day_data[9:]

    dp = 3
    with with_pool() as pool:
        # pool.starmap(extract_multiple_circles_and_count_stripes,
        #              [(data_path, 0.43, 0.5, False, dp) for data_path in column_data])
        pool.starmap(extract_multiple_circles_and_count_stripes,
                     [(data_path, 0.125, 0.34, False, dp) for data_path in box_data])


def run_box(data_path, use_cache, dp):
    images, radius, angles = extract_multiple_circles_and_count_stripes(data_path,
                                                                        0.125, 0.38,
                                                                        use_cache=use_cache,
                                                                        dp=dp)
    solve_multiple_disks(images, radius, angles)
    return


def run_column(data_path, use_cache, dp):
    images, radius, angles = extract_multiple_circles_and_count_stripes(data_path, 0.41, 0.47,
                                                                        use_cache=use_cache, dp=dp)

    solve_multiple_disks(images, radius, angles)


if __name__ == '__main__':
    do_day_3()
