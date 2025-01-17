from photoelasticity.days.data import get_day_data
from photoelasticity.image_detection.image_detection import extract_multiple_circles_and_count_stripes


def do_day_3():
    day_data = get_day_data(3)
    related_box_data = [extract_multiple_circles_and_count_stripes(
        image_path, 0.125, 0.31,
        should_cache=False)
        for image_path in day_data]