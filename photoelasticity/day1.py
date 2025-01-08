import numpy as np

from photoelasticity.data import get_day_data
from photoelasticity.fit_curve import find_fit_params
from photoelasticity.image_detection import extract_circle_and_count_stripes
from photoelasticity.matrix_tools import resize_matrix


def do_day_1():
    related_data = [
        # [247, 259, 263, 275, 276],
        [248, 258, 264, 274, 277],
        [249, 257, 265, 273, 278],
        [250, 256, 266, 272, 279],
        [251, 255, 267, 271, 280],
        [252, 254, 268, 270, 281]
    ]
    stresses = [14.7, 19.6, 24.5, 29.4, 34.3]
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
    guessess = [4.98E+00, 7.01E+00, 16, 18, 24 ]
    for i, data in enumerate(related_data):
        max_r = max(len(subdata) for subdata in data)
        resized_data = [resize_matrix(subdata, (max_r, max_r)) for subdata in data]
        center_1D = np.mean(resized_data, axis=0)
        find_fit_params(center_1D, f"Plane polariscope with stress {stresses[i]}N", guessess[i])
