from photoelasticity.days.data import get_day_data
from photoelasticity.image_detection.image_detection import extract_multiple_circles_and_count_stripes
from photoelasticity.tools.multiprocessing import with_pool


def do_day_4():
    day_data = get_day_data(4)
    with with_pool() as pool:
        pool.map(run_image_detection, day_data)


def run_image_detection(data_path):
    return extract_multiple_circles_and_count_stripes(data_path, 0.15, 0.31,
                                                      use_cache=False, dp=1.5)


if __name__ == '__main__':
    do_day_4()
