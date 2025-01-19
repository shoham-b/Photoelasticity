import multiprocessing

from photoelasticity.days.data import get_day_data
from photoelasticity.image_detection.image_detection import extract_multiple_circles_and_count_stripes


def do_day_3():
    day_data = get_day_data(3)
    column_data, box_data = day_data[:9], day_data[9:]
    canny_threshold = 29
    should_cache = False

    with multiprocessing.Pool() as pool:
        pool.starmap(run_column, [(data_path, should_cache) for data_path in column_data])
        pool.starmap(run_box, [(data_path, canny_threshold, should_cache) for data_path in box_data])


def run_box(data_path, canny_threshold, should_cache):
    return extract_multiple_circles_and_count_stripes(
        data_path, 0.125, 0.31,
        should_cache=should_cache, canny_threshold=canny_threshold)


def run_column(data_path, should_cache):
    return extract_multiple_circles_and_count_stripes(
        data_path, 0.355, 0.46,
        should_cache=should_cache, canny_threshold=data_path)


if __name__ == '__main__':
    do_day_3()
