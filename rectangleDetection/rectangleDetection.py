# TODO figure how to use the function from common location
# from ../util/cvUtil import auto_canny

import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import scipy.misc
import os


# This method is useful for applying Canny filter and automatically determining the thresholds.
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def transform_to_edge_images():
    image_paths = glob.glob("images/*.jpg")

    for imagePath in image_paths:
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edgeImage = auto_canny(blurred)

        image_name = os.path.split(imagePath)[-1]
        scipy.misc.imsave("edgeImages/" + image_name, edgeImage)


        # plt.figure()
        # plt.imshow(gray, cmap="gray")


# transform_to_edge_images()
