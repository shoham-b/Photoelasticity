from pathlib import WindowsPath, Path

import cv2
import diskcache
import numpy as np
from imageio.v2 import imwrite


class ImageError(Exception):
    pass


cache = diskcache.Cache("../image_cache")

allowed_circle_collision = 0.85
allowed_neigbhor_distance = 1.2
prominent_circles_num = 40


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

    neighbour_circles = _find_neighbour_circles_matrix(filtered_circles, allowed_neigbhor_distance)
    centers_angles = _find_circle_center_angles(filtered_circles)
    neighbour_circles_angle = np.where(neighbour_circles, centers_angles, np.nan)

    circles_images = [gray[y - r:y + r, x - r:x + r] for (x, y, r) in filtered_circles]

    for (x, y, r) in filtered_circles:
        _draw_circle(image_path, output, r, x, y)

    if should_cache:
        cache[image_path] = (circles_images, neighbour_circles_angle)

    return circles_images, neighbour_circles_angle


def _find_circle_center_angles(circles):
    circle_centers = circles[:, 0:2]
    circle_centers_diff = circle_centers[:, np.newaxis, :] - circle_centers[np.newaxis, :, :]
    circle_center_angle = np.arctan2(circle_centers_diff[:, :, 1], circle_centers_diff[:, :, 0])
    return circle_center_angle


def _filter_colliding_circles(circles):
    collision_mask = _find_neighbour_circles_matrix(circles, allowed_circle_collision)
    np.fill_diagonal(collision_mask, False)  # Ignore self-collisions

    tril = np.tril(collision_mask)
    collided_before = collision_mask & tril
    circles_mask = np.any(collided_before, axis=1)

    filtered_circles = circles[~circles_mask]

    return filtered_circles


def _find_neighbour_circles_matrix(circles, allowed_collision):
    circle_centers = circles[:, 0:2]
    circle_radii = circles[:, 2]
    circle_centers_diff = circle_centers[:, np.newaxis, :] - circle_centers[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(circle_centers_diff, axis=-1)
    radius_sum_matrix = circle_radii[:, np.newaxis] + circle_radii[np.newaxis, :]
    # Create a mask to filter out colliding circles
    collision_mask = dist_matrix < radius_sum_matrix * allowed_collision
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
    canny_threshold_upper = np.percentile(gray, 50, method="weibull")
    canny_threshold_lower = np.percentile(gray, 10, method="weibull")
    canny = cv2.Canny(gray, canny_threshold_lower, canny_threshold_upper, 11)
    imwrite(fr"{__file__}/../../../canny/{image_path.name}.canny.jpg", canny)
    image_height, image_width = gray.shape
    max_fitting_radius = min(image_height, image_width) // 2
    max_radius = int(max_fitting_radius * max_rad_percent)
    min_radius = int(max_fitting_radius * min_rad_percent)
    circles = cv2.HoughCircles(canny,
                               cv2.HOUGH_GRADIENT,
                               1.3, min_radius,
                               param1=10, param2=45,
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
