import numpy as np
from skimage import io, feature, transform, draw


def extract_circle_and_count_stripes(image_path):
    # Read the image
    img = io.imread(image_path, as_gray=True)

    # Create a circular mask
    height, width = img.shape
    center = (width // 2, height // 2)
    radius = min(width, height) // 2
    mask = np.zeros(img.shape, dtype=bool)
    rr, cc = draw.circle(center[1], center[0], radius)
    mask[rr, cc] = True

    # Apply the mask
    masked_img = img * mask

    # Edge detection
    edges = feature.canny(masked_img, sigma=3)

    # Hough transform for circle detection
    hough_radii = np.arange(10, 50, 2)
    hough_res = transform.hough_circle(edges, hough_radii)

    # Find peaks in Hough space
    accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    # Count stripes (simplified approach)
    num_stripes = len(feature.peak_local_max(masked_img, min_distance=10))

    return num_stripes, masked_img
