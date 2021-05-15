import cv2
import numpy as np
from skimage import filters


def find_threshhold(diff_image):
    """
    automatisch goede threshold waarde vinden gebaseerd op
    Image difference threshold strategies andshadow detection (Paul L. Rosin, Tim Ellis)
    (https://pdfs.semanticscholar.org/ccbd/40c5d670f4532ad1a9f0003a8b8157388aa0.pdf)
    :param diff_image: verschilbeeld dat moet worden gethreshold
    :return: threshold value
    """
    med = np.median(diff_image.ravel())
    mad = np.median(np.abs(diff_image - med).ravel())
    return med + 6 * mad


def diff_color_3(image1, image2, image3):
    """
    difference methode op basis van drie frames
    ghosts worden verwijderd door de and operatie te nemen
    :param image1: frame 1
    :param image2: frame 2
    :param image3: frame 3
    :return: segmentatie van frame 2
    """
    res12 = diff_color(image1, image2)
    res32 = diff_color(image2, image3)
    return np.bitwise_and(res12, res32)


def diff_color(image1, image2):
    """
    difference methode in de LAB kleurruimte
    :param image1: frame 1
    :param image2: frame 2
    :return: segmentatie van frame 2
    """
    # euclidische afstand in LAB werkt beter dan in RGB
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB).astype(np.float32)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB).astype(np.float32)

    # verschilbeeld nemen
    im = np.sqrt((image1[:, :, 1] - image2[:, :, 1]) ** 2 + (image1[:, :, 0] - image2[:, :, 0]) ** 2 + (
            image1[:, :, 2] - image2[:, :, 2]) ** 2)
    # blur toepassen

    im = cv2.GaussianBlur(im, (5, 5), 0)

    # hysteresis thresholding
    threshhold = find_threshhold(im)
    im = (filters.apply_hysteresis_threshold(im, threshhold, threshhold * 6) * 255).astype(np.uint8)

    # opvulling van gaten in blobs
    contours, _ = cv2.findContours(im, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        cv2.drawContours(im, [c], 0, 255, -1)

    return im


def diff_remove_ghost(image1, image2):
    """
    difference methode die ghosts verwijderd op basis van randsimilariteit.
    Werkt enkel goed als de blobs van objecten en ghosts niet overlappen.
    zie Real-time ghost removal for foreground segmentation methods (Fei Yin)
    :param image1: frame 1
    :param image2: frame 2
    :return: segmentatie van frame 2
    """
    im_diff = cv2.absdiff(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY),
                          cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY))
    im_diff = cv2.GaussianBlur(im_diff, (5, 5), 0)
    threshold = find_threshhold(im_diff)
    res = (filters.apply_hysteresis_threshold(im_diff, threshold, threshold * 6) * 255).astype(np.uint8)

    # opvulling van gaten in blobs
    contours, _ = cv2.findContours(res, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        cv2.drawContours(res, [c], 0, 255, -1)

    diff_edges = cv2.Canny(im_diff, 25, 50)
    im1_edges = cv2.Canny(image1, 25, 50)
    im2_edges = cv2.Canny(image2, 25, 50)

    # cv2.imshow("edges", im2_edges)
    # cv2.imshow("masked diff edges", res / 255 * diff_edges)
    # cv2.imshow("masked edges ", res / 255 * im2_edges)
    # cv2.imshow("before", res)

    # verwijder blobs met slechte randsimilariteit
    contours, _ = cv2.findContours(res, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        mask = np.zeros_like(im1_edges)
        mask = cv2.drawContours(mask, contours, i, 1, thickness=cv2.FILLED)
        masked_edges_im1 = im1_edges * mask
        masked_edges_im2 = im2_edges * mask
        masked_diff_edges = diff_edges * mask
        n_union = np.sum(np.logical_and(masked_edges_im2, masked_diff_edges))
        n_intersection = np.sum(np.logical_or(masked_edges_im2, masked_diff_edges))
        if (n_union / n_intersection) < 0.3:
            cv2.drawContours(res, contours, i, 0, -1)
        # extra idee: slic ipv contours

    return res


if __name__ == "__main__":
    im1 = cv2.imread("Images/pedestrians/input/in000470.jpg")
    im2 = cv2.imread("Images/pedestrians/input/in000475.jpg")
    im3 = cv2.imread("Images/pedestrians/input/in000480.jpg")
    # im1 = cv2.imread("Images/highway/input/in000110.jpg")
    # im2 = cv2.imread("Images/highway/input/in000120.jpg")
    cv2.imshow("three frame diff", diff_color_3(im1, im2, im3))
    # cv2.imshow("remove ghost", diff_remove_ghost(im1, im2))
    # cv2.imshow("origineel", im2)
    cv2.waitKey()
