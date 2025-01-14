from pathlib import WindowsPath, Path

import cv2
import diskcache
import numpy as np


class ImageError(Exception):
    pass


def extract_circle_and_count_stripes(image_path: WindowsPath, min_rad_percent, max_rad_percent) -> np.array:
    # load the image, clone it for output, and then convert it to grayscale
    cache = diskcache.Cache("../image_cache")
    cached = cache.get(image_path)
    if cached is not None:
        return cached

    image = cv2.imread(image_path)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 20, 20)
    image_height, image_width = gray.shape
    max_fitting_radius = min(image_height, image_width) // 2
    max_radius = int(max_fitting_radius * max_rad_percent)
    min_radius = int(max_fitting_radius * min_rad_percent)
    circles = cv2.HoughCircles(canny,
                               cv2.HOUGH_GRADIENT,
                               16, 30,
                               param1=30, param2=60,
                               minRadius=min_radius, maxRadius=max_radius)

    # ensure at least some circles were found
    if circles is None:
        raise ImageError("No circles detected")

    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    (x, y, r) = circles[0]  # Get the first circle in the list

    # draw the circle in the output image, then draw a rectangle
    # corresponding to the center of the circle
    cv2.circle(output, (x, y), r, (0, 255, 0), 4)

    # show the output image
    save_circle_image(image_path, output)
    # plt.imshow(output)

    # numed_data = find_center_strip(gray, image_path, x, y)
    result = gray[y - r:y + r, x - r:x + r]

    cache[image_path] = result

    return result


def save_circle_image(image_path, output):
    output_path = get_output_path(image_path, 'circle')
    assert cv2.imwrite(output_path, np.hstack([output]))


def get_output_path(image_path, output_type):
    output_dir = Path(__file__).parent.parent / f"drawn_{output_type}"
    output_dir.mkdir(exist_ok=True, parents=True)
    return (output_dir / image_path.name).absolute()


def save_strip_image(image_path, cropped_center):
    output_path = get_output_path(image_path, 'strip')
    assert cv2.imwrite(output_path, np.hstack([cropped_center]))
