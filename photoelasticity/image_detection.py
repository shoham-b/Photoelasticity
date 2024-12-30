import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_circle_and_count_stripes(image_path):
    # Load the image

    # load the image, clone it for output, and then convert it to grayscale
    image = cv2.imread(image_path)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_height, image_width = gray.shape
    min_radius = max(10, min(image_height, image_width) // 4)  # Minimum radius
    max_radius = min(image_height, image_width)  # Maximum radius
    circles = cv2.HoughCircles(gray,
                               cv2.HOUGH_GRADIENT,
                               2, 10,
                               param1=30,param2=50,
                               minRadius=min_radius, maxRadius=0)

    print(circles)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # count = count+1
        print(circles)

        # print(count)

        # loop over the (x, y) coordinates and radius of the circles
        (x, y, r) = circles[0]  # Get the first circle in the list

        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)

        # show the output image
        # cv2.imshow("output", np.hstack([output]))
        cv2.imwrite('output.jpg', np.hstack([output]), [cv2.IMWRITE_JPEG_QUALITY, 70])
        plt.imshow(image)

        # dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
    # mask = dist_from_center <= radius

    # Apply the mask
    # masked_image = np.copy(image)
    # masked_image[~mask] = 0  # Set pixels outside the mask to black

