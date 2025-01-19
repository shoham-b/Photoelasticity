from photoelasticity.days.data import get_day_data
from photoelasticity.image_detection.image_detection import extract_multiple_circles_and_count_stripes


def do_day_3():
    day_data = get_day_data(3)
    column_data, box_data = day_data[:9], day_data[9:]
    related_column_data = [extract_multiple_circles_and_count_stripes(
        image_path, 0.225, 0.37,
        should_cache=True)
        for image_path in column_data]
    related_box_data = [extract_multiple_circles_and_count_stripes(
        image_path, 0.125, 0.31,
        should_cache=True)
        for image_path in box_data]
