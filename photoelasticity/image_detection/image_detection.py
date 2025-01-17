from pathlib import WindowsPath, Path

import cv2
import diskcache
import numpy as np
from imageio.v2 import imwrite


class ImageError(Exception):
    pass


cache = diskcache.Cache("../image_cache")

allowed_circle_collision = 10
prominent_circles_num = 20
rough_canny_params = 60, 70


def extract_circle_and_count_stripes(image_path: WindowsPath, min_rad_percent, max_rad_percent) -> np.array:
    if (cached := cache.get(image_path)) is not None:
        return cached

    circles, gray, output = _find_circles(image_path, max_rad_percent, min_rad_percent)

    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    (x, y, r) = circles[0]  # Get the first circle in the list

    _draw_circle(image_path, output, r, x, y)

    result = gray[y - r:y + r, x - r:x + r]

    cache[image_path] = result
    return result


def extract_multiple_circles_and_count_stripes(image_path: WindowsPath, min_rad_percent, max_rad_percent,
                                               should_cache=True) -> np.array:
    if should_cache and ((cached := cache.get(image_path)) is not None):
        return cached

    circles, gray, output = _find_circles(image_path, max_rad_percent, min_rad_percent)

    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    prominent_circles = circles[:prominent_circles_num]
    filtered_circles = _filter_colliding_circles(prominent_circles)
    neighbour_circles = _find_neighbour_circles_matrix(filtered_circles)

    for (x, y, r) in circles:
        _draw_circle(image_path, output, r, x, y)

    if should_cache:
        cache[image_path] = filtered_circles

    return filtered_circles, neighbour_circles


def _filter_colliding_circles(circles):
    collision_mask = _find_neighbour_circles_matrix(circles)
    np.fill_diagonal(collision_mask, False)  # Ignore self-collisions

    tril = np.tril(collision_mask)
    collided_before = collision_mask & tril
    circles_mask = np.any(collided_before, axis=1)

    filtered_circles = circles[circles_mask]

    return filtered_circles


def _find_neighbour_circles_matrix(circles):
    circle_centers = circles[:, 1:3]
    circle_radii = circles[:, 0]
    dist_matrix = np.sqrt(
        np.sum((circle_centers[:, np.newaxis] - circle_centers[np.newaxis, :]) ** 2, axis=2))
    radius_sum_matrix = circle_radii[:, np.newaxis] + circle_radii[np.newaxis, :]
    # Create a mask to filter out colliding circles
    collision_mask = dist_matrix < radius_sum_matrix - allowed_circle_collision
    np.fill_diagonal(collision_mask, False)  # Ignore self-collisions
    return collision_mask


def _draw_circle(image_path, output, r, x, y):
    # draw the circle in the output image, then draw a rectangle
    # corresponding to the center of the circle
    cv2.circle(output, (x, y), r, (0, 255, 0), 4)
    # show the output image
    _save_circle_image(image_path, output)


def _find_circles(image_path, max_rad_percent, min_rad_percent):
    # load the image, clone it for output, and then convert it to grayscale
    image = cv2.imread(image_path)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 30, 30)
    imwrite(fr"../canny/{image_path.name}canny.jpg", canny)
    image_height, image_width = gray.shape
    max_fitting_radius = min(image_height, image_width) // 2
    max_radius = int(max_fitting_radius * max_rad_percent)
    min_radius = int(max_fitting_radius * min_rad_percent)
    circles = cv2.HoughCircles(canny,
                               cv2.HOUGH_GRADIENT,
                               2.1, min_radius,
                               param1=45, param2=55,
                               minRadius=min_radius, maxRadius=max_radius)
    # ensure at least some circles were found
    if circles is None:
        raise ImageError("No circles detected")
    return circles, gray, output


def _save_circle_image(image_path, output):
    output_path = get_output_path(image_path, 'circle')
    assert cv2.imwrite(output_path, np.hstack([output]))


def get_output_path(image_path, output_type):
    output_dir = Path(__file__).parent.parent.parent / f"drawn_{output_type}"
    output_dir.mkdir(exist_ok=True, parents=True)
    return (output_dir / image_path.name).absolute()


def save_strip_image(image_path, cropped_center):
    output_path = get_output_path(image_path, 'strip')
    assert cv2.imwrite(output_path, np.hstack([cropped_center]))
