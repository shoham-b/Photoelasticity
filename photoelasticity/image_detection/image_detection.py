from pathlib import WindowsPath, Path

import cv2
import diskcache
import numpy as np
from imageio.v2 import imwrite


class ImageError(Exception):
    pass


cache = diskcache.Cache("../image_cache")

allowed_circle_collision = 0.88
allowed_neigbhor_distance = 1.1
prominent_circles_num = 40


def extract_circle_and_count_stripes(image_path: WindowsPath, min_rad_percent, max_rad_percent) -> np.array:
    if (cached := cache.get(image_path)) is not None:
        return cached

    circles, gray, output = _find_circles(image_path, max_rad_percent, min_rad_percent, dp=1.4)

    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    (x, y, r) = circles[0]  # Get the first circle in the list

    _draw_circle(output, r, x, y)

    result = gray[y - r:y + r, x - r:x + r]

    cache[image_path] = result
    return result


def extract_multiple_circles_and_count_stripes(image_path: WindowsPath, min_rad_percent, max_rad_percent,
                                               use_cache, dp) -> np.array:
    if use_cache and ((cached := cache.get(image_path)) is not None):
        return cached

    gray, output, circles = _find_prominent_circles(dp, image_path, max_rad_percent, min_rad_percent)

    centers_angles = _find_circle_center_angles(circles)

    neighbour_circles = _find_neighbour_circles_matrix(circles, allowed_neigbhor_distance)
    neighbour_circles_angle = np.where(neighbour_circles, centers_angles, np.nan)

    circles_dir = Path(fr"{__file__}/../../../circles/{image_path.stem}").resolve()
    circles_dir.mkdir(exist_ok=True, parents=True)
    circles_images = []
    for i, (x, y, r) in enumerate(circles):
        cropped_center = _get_cropped_circle(gray, r, x, y)
        cropped_center_path = str(circles_dir / f"{i}.jpg")
        cv2.imwrite(cropped_center_path, cropped_center)
        circles_images.append(cropped_center_path)
        _draw_circle(output, r, x, y)
    _connect_neighbohr_circle_centers(circles, neighbour_circles, output)
    _save_circle_image(image_path, output)
    circle_radiuses = circles[::, 2]

    results = circles_images, circle_radiuses, neighbour_circles_angle
    cache[image_path] = results
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


def _draw_circle(output, r, x, y):
    # draw the circle in the output image, then draw a rectangle
    # corresponding to the center of the circle
    cv2.circle(output, (x, y), r, (255, 0, 0), 4)


def _connect_neighbohr_circle_centers(circles, neighbour_circles, output):
    for i, (x, y, r) in enumerate(circles):
        for j, (x2, y2, r2) in enumerate(circles):
            if neighbour_circles[i, j]:
                cv2.line(output, (x, y), (x2, y2), (0, 0, 255), 2)


def _find_circles(image_path, max_rad_percent, min_rad_percent, dp):
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
                               dp, min_radius,
                               param1=canny_threshold_upper, param2=45,
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
