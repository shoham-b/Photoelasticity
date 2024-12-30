import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_circle_and_count_stripes(image_path):
    # Load the image
    image = cv2.imread(image_path)  # Replace 'your_image.jpg' with your file path
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    cimg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Apply Gaussian blur to reduce noise
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using Hough Circle Transform
    image_shape = image.shape
    maxRadius = max(image_shape)
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        param1=10,  # Upper threshold for the internal Canny edge detector
        param2=30,  # Threshold for center detection
        minDist=int(maxRadius),
        minRadius=maxRadius // 4,  # Minimum radius of the circles
        maxRadius=maxRadius  # Maximum radius of the circles
    )

    # If circles are detected, draw them
    if circles is not None:
        circles = np.uint16(np.around(circles))  # Round the float values to integers
        centerX,centerY,radius = circles[0, :][0]
        cv2.circle(image, (centerX,centerY),radius, (0, 255, 0), 2)

        Y, X = np.ogrid[:image_shape[0], :image_shape[1]]

        dist_from_center = np.sqrt((X - centerX) ** 2 + (Y - centerY) ** 2)
        mask = dist_from_center <= radius

        # Apply the mask
        masked_image = np.copy(image)
        masked_image[~mask] = 0  # Set pixels outside the mask to black

        cv2.imshow('detected circles', cimg)
        cv2.imshow('masked circles', masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    exit()
