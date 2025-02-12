import logging
from pathlib import WindowsPath, Path

import cv2
import diskcache
import numpy as np
from imageio.v2 import imwrite

CONNECT_CIRCLES_COLOR = (240, 184, 43)
FOUND_CIRCLES_COLOR = (240, 155, 90)


class ImageError(Exception):
    pass


cache = diskcache.Cache("../image_cache")

allowed_circle_collision = 0.88
allowed_neigbhor_distance = 1.03
prominent_circles_num = 40


@cache.memoize()
def extract_circle_and_count_stripes(image_path: WindowsPath, min_rad_percent, max_rad_percent) -> np.array:
    circles, gray, output = _find_circles(image_path, max_rad_percent, min_rad_percent, dp=1.4)

    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    (x, y, r) = circles[0]  # Get the first circle in the list

    _draw_circle(output, r, x, y)

    return gray[y - r:y + r, x - r:x + r]


def extract_multiple_circles_and_count_stripes(image_path: WindowsPath, min_rad_percent, max_rad_percent,
                                               use_cache, dp, ignore_disks=(), neigbhors_to_ignore=()) -> np.array:
    if use_cache and ((cached := cache.get(image_path)) is not None):
        return cached

    gray, output, photoelastic_circles = _find_prominent_circles(dp, image_path, max_rad_percent, min_rad_percent)
    _write_all_circles_numbers(output, photoelastic_circles)
    small_blue_circles = _find_small_blue_circles(image_path)
    all_circles = np.vstack(
        [photoelastic_circles, small_blue_circles]) if small_blue_circles is not None else photoelastic_circles
    neighbour_circles = _find_neighbour_circles_matrix(all_circles)
    neighbour_circles_angle = _find_circle_center_angles(all_circles)
    for to_ignore in ignore_disks:
        neighbour_circles_angle[to_ignore, :] = np.nan
        neighbour_circles_angle[:, to_ignore] = np.nan
    for circle_a, circle_b in neigbhors_to_ignore:
        neighbour_circles[circle_a, circle_b] = np.nan
        neighbour_circles[circle_b, circle_a] = np.nan
    angle_or_none = np.where(neighbour_circles, neighbour_circles_angle, np.nan) + np.pi

    angles_per_photoelastic_circle = [[angle for angle in neigh if not np.isnan(angle)] for neigh in
                                      angle_or_none[:len(photoelastic_circles)]]

    circles_dir = Path(fr"{__file__}/../../../circles/{image_path.stem}").resolve()
    circles_dir.mkdir(exist_ok=True, parents=True)
    circles_images = []
    for i, (x, y, r) in enumerate(photoelastic_circles):
        cropped_center = _get_cropped_circle(gray, r, x, y)
        cropped_center_path = str(circles_dir / f"{i}.jpg")
        cv2.imwrite(cropped_center_path, cropped_center)
        circles_images.append(cropped_center_path)

    for (x, y, r) in all_circles:
        _draw_circle(output, r, x, y)
    _connect_neighbohr_circle_centers(all_circles, neighbour_circles, output)
    _save_circle_image(image_path, output)
    circle_radiuses = photoelastic_circles[::, 2]

    results = circles_images, circle_radiuses, angles_per_photoelastic_circle
    cache[image_path] = results
    logging.info(f"Extracted circles from {image_path}")
    return results


def _get_cropped_circle(gray, r, x, y):
    image_copy = gray.copy()
    circular_mask = create_circular_mask(image_copy.shape, (x, y), r)
    image_copy[~circular_mask] = 0
    xs, ys = np.where(image_copy)
    left_boundary = np.min(xs)
    right_boundary = np.max(xs)
    top_boundary = np.min(ys)
    bottom_boundary = np.max(ys)
    cropped_center = image_copy[left_boundary:right_boundary, top_boundary:bottom_boundary]
    return cropped_center


def _find_prominent_circles(dp, image_path, max_rad_percent, min_rad_percent):
    circles, gray, output = _find_circles(image_path, max_rad_percent, min_rad_percent, dp)
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    prominent_circles = circles[:prominent_circles_num]
    filtered_circles = _filter_colliding_circles(prominent_circles)
    distances = np.sqrt(filtered_circles[:, 0] ** 2 + filtered_circles[:, 1] ** 2)
    sorted_circles = filtered_circles[np.argsort(distances)]
    return gray, output, sorted_circles


def create_circular_mask(shape, center, radius):
    h, w = shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def _find_circle_center_angles(circles):
    circle_centers = circles[:, 0:2]
    circle_centers_diff = circle_centers[:, np.newaxis] - circle_centers[np.newaxis, :]
    circle_center_angle = np.arctan2(circle_centers_diff[:, :, 1], circle_centers_diff[:, :, 0])
    return circle_center_angle


def _filter_colliding_circles(circles):
    collision_mask = _find_collision_circles_matrix(circles, )
    np.fill_diagonal(collision_mask, False)  # Ignore self-collisions

    tril = np.tril(collision_mask)
    collided_before = collision_mask & tril
    circles_mask = np.any(collided_before, axis=1)

    filtered_circles = circles[~circles_mask]

    return filtered_circles


def _find_neighbour_circles_matrix(circles):
    dist_matrix, radius_sum_matrix = _get_dist_and_rad_sum(circles)
    # Create a mask to filter out colliding circles
    collision_mask = dist_matrix < np.clip(radius_sum_matrix * allowed_neigbhor_distance, radius_sum_matrix + 20, None)
    np.fill_diagonal(collision_mask, False)  # Ignore self-collisions

    return collision_mask


def _find_collision_circles_matrix(circles):
    dist_matrix, radius_sum_matrix = _get_dist_and_rad_sum(circles)
    # Create a mask to filter out colliding circles
    collision_mask = dist_matrix < radius_sum_matrix * allowed_circle_collision
    np.fill_diagonal(collision_mask, False)  # Ignore self-collisions

    return collision_mask


def _find_circle_in_circle(circles):
    circle_centers = circles[:, 0:2]
    circle_radii = circles[:, 2]
    circle_centers_diff = circle_centers[:, np.newaxis, :] - circle_centers[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(circle_centers_diff, axis=-1)

    # Create a mask to filter out colliding circles
    collision_mask = dist_matrix < np.maximum(circle_radii[:, np.newaxis], circle_radii[np.newaxis, :])
    np.fill_diagonal(collision_mask, False)  # Ignore self-collisions

    return collision_mask


def _get_dist_and_rad_sum(circles):
    circle_centers = circles[:, 0:2]
    circle_radii = circles[:, 2]
    circle_centers_diff = circle_centers[:, np.newaxis, :] - circle_centers[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(circle_centers_diff, axis=-1)
    radius_sum_matrix = circle_radii[:, np.newaxis] + circle_radii[np.newaxis, :]
    return dist_matrix, radius_sum_matrix


def _draw_circle(output, r, x, y):
    # draw the circle in the output image, then draw a rectangle
    # corresponding to the center of the circle
    cv2.circle(output, (x, y), r, FOUND_CIRCLES_COLOR, 4)


def _write_all_circles_numbers(output, circles):
    for i, (x, y, r) in enumerate(circles):
        _write_circle_number(output, i, x, y)


def _write_circle_number(output, i, x, y):
    cv2.putText(output, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 4, FOUND_CIRCLES_COLOR, 3)


def _connect_neighbohr_circle_centers(circles, neighbour_circles, output):
    for i, (x, y, r) in enumerate(circles):
        for j, (x2, y2, r2) in enumerate(circles):
            if neighbour_circles[i, j]:
                cv2.line(output, (x, y), (x2, y2), CONNECT_CIRCLES_COLOR, 2)


def _find_circles(image_path, max_rad_percent, min_rad_percent, dp):
    # load the image, clone it for output, and then convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_threshold_upper = np.percentile(gray, 50, method="weibull")
    canny_threshold_lower = np.percentile(gray, 10, method="weibull")
    canny = cv2.Canny(gray, canny_threshold_lower, canny_threshold_upper, 20)
    imwrite(fr"{__file__}/../../../canny/{image_path.name}.canny.jpg", canny)
    image_height, image_width = gray.shape
    max_fitting_radius = min(image_height, image_width) // 2
    max_radius = int(max_fitting_radius * max_rad_percent)
    min_radius = int(max_fitting_radius * min_rad_percent)
    circles = cv2.HoughCircles(canny,
                               cv2.HOUGH_GRADIENT,
                               dp, min_radius,
                               param1=canny_threshold_upper, param2=45,
                               minRadius=min_radius, maxRadius=max_radius)
    # ensure at least some circles were found
    if circles is None:
        raise ImageError("No circles detected")
    return circles, gray, image.copy()


def _find_small_blue_circles(image_path, dp=1):
    image = cv2.imread(image_path)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.where(blue_mask, gray, 0)

    blue_dir = Path(fr"{__file__}/../../../blue").resolve()
    blue_dir.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(blue_dir / image_path.name), gray)

    image_height, image_width = gray.shape
    max_fitting_radius = min(image_height, image_width) // 2
    max_radius = int(max_fitting_radius * 0.1)
    min_radius = int(max_fitting_radius * 0.01)
    circles = cv2.HoughCircles(gray,
                               cv2.HOUGH_GRADIENT,
                               dp, min_radius,
                               param2=15,
                               minRadius=min_radius, maxRadius=max_radius)

    if circles is None:
        return None
    filtered_circles = _filter_colliding_circles(circles.astype(int)[0])
    return filtered_circles


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
