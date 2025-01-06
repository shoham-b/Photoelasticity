from photoelasticity.data import get_day_data
from photoelasticity.fit_curve import find_fit_params
from photoelasticity.image_detection import extract_circle_and_count_stripes
from photoelasticity.matrix_tools import find_center_strip


def do_day_2():
    related_data = range(7, 15)
    interesting_images = [
        f"DSC_{str(image_num).zfill(4)}.jpg"
        for image_num in related_data
    ]
    related_data = [extract_circle_and_count_stripes(image_path, 0.425, 0.44)
                    for image_path in get_day_data(2, interesting_images)]

    for i, data in enumerate(related_data):
        middle_strip = find_center_strip(data)
        find_fit_params(middle_strip, f"data of {i}")
