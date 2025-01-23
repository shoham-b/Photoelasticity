from photoelasticity.days.data import get_day_data
from photoelasticity.fringes.fit_curve import find_fit_params
from photoelasticity.image_detection.image_detection import extract_circle_and_count_stripes


def do_day_2():
    related_data = range(8, 15)
    interesting_images = [
        f"DSC_{str(image_num).zfill(4)}.jpg"
        for image_num in related_data
    ]
    force = [523, 663, 807, 922, 1045, 1152, 1221]
    related_data = [extract_circle_and_count_stripes(image_path, 0.425, 0.44)
                    for image_path in get_day_data(2, interesting_images)]
    guess_I = [50, 50, 40, 45, 40, 46, 50, 50]
    guess_A = [7.28, 9.6, 11, 12, 15, 12.1, 12, 11]
    guess_offset = [30, 15, 22, 25, 0, 36, -25, 0]
    images_num = len(guess_A)
    guesses = [[guess_I[i], guess_A[i], guess_offset[i]] for i in range(images_num)]
    for i, data in enumerate(related_data):
        find_fit_params(data, f"Circular polariscope with force {force[i]}N", guesses[i], 0.6)


if __name__ == '__main__':
    do_day_2()
