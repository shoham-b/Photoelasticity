from photoelasticity.days.data import get_day_data
from photoelasticity.image_detection.image_detection import extract_multiple_circles_and_count_stripes


def do_day_3():
    stresses = [4.9, 9.8, 14.7, 19.6, 24.5, 29.4, 34.3]
    day_data = get_day_data(3)
    related_data = [extract_multiple_circles_and_count_stripes(
        image_path, 0.15, 0.4,
        should_cache=False)
        for image_path in day_data]
