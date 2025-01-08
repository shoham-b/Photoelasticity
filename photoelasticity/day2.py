from photoelasticity.data import get_day_data
from photoelasticity.fit_curve import find_fit_params
from photoelasticity.image_detection import extract_circle_and_count_stripes


def do_day_2():
    related_data = range(8, 15)
    interesting_images = [
        f"DSC_{str(image_num).zfill(4)}.jpg"
        for image_num in related_data
    ]
    stresses = [4.9, 9.8, 14.7, 19.6, 24.5, 29.4, 34.3]
    related_data = [extract_circle_and_count_stripes(image_path, 0.425, 0.44)
                    for image_path in get_day_data(2, interesting_images)]
    guesses = [7.28, 9.82, 8.12, 9.45, 12.51, 14.65, 12.63, 13.27]
    for i, data in enumerate(related_data):
        find_fit_params(data, f"Circular polariscope with stress {stresses[i]}N", guesses[i], 0.6)
