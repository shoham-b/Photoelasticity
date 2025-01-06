import numpy as np

from photoelasticity.data import get_day_data
from photoelasticity.fit_curve import find_fit_params
from photoelasticity.image_detection import extract_circle_and_count_stripes
from photoelasticity.matrix_tools import resize_matrix, find_center_strip


def do_day_1():
    related_data = [
        # [247, 259, 263, 275, 276],
        [248, 258, 264, 274, 277],
        [249, 257, 265, 273, 278],
        [250, 256, 266, 272, 279],
        [251, 255, 267, 271, 280],
        [252, 254, 268, 270, 281]
    ]
    interesting_images = [
        [f"V_0{str(image_num).zfill(3)}.jpg"
         for image_num in bundle]
        for bundle in related_data
    ]
    related_data = [
        [extract_circle_and_count_stripes(image_path, 0.9, 0.98)
         for image_path in get_day_data(1, interesting_images_bundle)]
        for interesting_images_bundle in interesting_images
    ]
    for i, data in enumerate(related_data):
        max_r = max(len(subdata) for subdata in data)
        resized_data = [resize_matrix(subdata, (max_r, max_r)) for subdata in data]
        averaged_data = np.mean(resized_data, axis=0)
        middle_strip = find_center_strip(averaged_data)
        find_fit_params(middle_strip, f"data of {i}")
