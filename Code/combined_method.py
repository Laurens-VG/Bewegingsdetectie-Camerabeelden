from difference_method import diff_color, diff_color_3
from Edge_based_method import edge_based_method, edge_based_method_3
import numpy as np
import cv2


def combined_method(image1, image2):
    """
    combineer diff_color en edge_based_method via and-operatie
    :param image1: frame 1
    :param image2: frame 2
    :return: segmentatie van frame 2
    """
    result1 = diff_color(image1, image2)
    result2 = edge_based_method(image1, image2)
    combine = np.bitwise_and(result1, result2)
    return combine


def combined_method_3(image1, image2, image3):
    """
    combineer diff_color en edge_based_method via and-operatie
    :param image1: frame 1
    :param image2: frame 2
    :param image3: frame 2
    :return: segmentatie van frame 2
    """
    result1 = diff_color_3(image1, image2, image3)
    result2 = edge_based_method_3(image1, image2, image3)
    combine = np.bitwise_and(result1, result2)
    return combine


if __name__ == "__main__":
    im1 = cv2.imread("Images/pedestrians/input/in000600.jpg")
    im2 = cv2.imread("Images/pedestrians/input/in000610.jpg")
    combined_method(im1, im2)
    cv2.waitKey()
