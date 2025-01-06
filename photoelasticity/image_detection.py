import os
from pathlib import WindowsPath

import cv2
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.distance_measures import radius

resulted_circles = r"C:\Users\shoha\PycharmProjects\Photoelasticity\drawn_circles"
resulted_stripe = r"C:\Users\shoha\PycharmProjects\Photoelasticity\stripes"
center_strip_size = 30
percentage_of_circle_to_take = 90


def extract_circle_and_count_stripes(image_path: WindowsPath) -> np.array:
    # load the image, clone it for output, and then convert it to grayscale
    image = cv2.imread(image_path)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_height, image_width = gray.shape
    max_radius = int(min(image_height, image_width) * 0.98) // 2  # Maximum radius
    min_radius = int(max_radius * 0.9)
    circles = cv2.HoughCircles(gray,
                               cv2.HOUGH_GRADIENT,
                               15, 30,
                               param1=30, param2=50,
                               minRadius=min_radius, maxRadius=max_radius)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        (x, y, r) = circles[0]  # Get the first circle in the list

        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)

        # show the output image
        save_circle_image(image_path, output)
        # plt.imshow(output)

        prominent_r = r * percentage_of_circle_to_take // 100

        cropped_center = gray[y - center_strip_size:y + center_strip_size, x - prominent_r:x + prominent_r]
        save_strip_image(image_path, cropped_center)
        oneDintensities = np.mean(cropped_center, 0)
        mean_brightness = np.mean(oneDintensities)
        numed_data = np.nan_to_num(oneDintensities, nan=mean_brightness)
        return numed_data


def save_circle_image(image_path, output):
    output_path = get_output_path(image_path)
    assert cv2.imwrite(output_path, np.hstack([output]), )


def get_output_path(image_path):
    return os.path.join(resulted_circles, image_path.name)


def save_strip_image(image_path, cropped_center):
    assert cv2.imwrite(os.path.join(resulted_stripe, image_path.name), np.hstack([cropped_center]), )
