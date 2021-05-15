import cv2
import numpy as np
from skimage import data, filters

stepByStepImages = False

lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=0,
                      qualityLevel=0.01,
                      minDistance=5)


def edge_based_method():
    image1 = cv2.imread("Images/pedestrians/input/in000470.jpg")
    image2 = cv2.imread("Images/pedestrians/input/in000475.jpg")
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    image1, image2 = pre_processing_gray_images(image1, image2)  # Slechtere resultaten worden verkregen

    # Bepaal de randen van beide startafbeeldingen (de magnitude)
    SobelImage1 = SobelOperator(image1)
    SobelImage2 = SobelOperator(image2)

    # Achtergrond subtractie waardoor bewegende elemeneten van afbeelding 2 tevoorschijn komen
    difference = background_subtraction(SobelImage1, SobelImage2)

    threshold = find_threshold(difference)
    hyst = hysteresis_threshold(difference, threshold, threshold * 3)

    # WORK ON OPTICAL FLOW
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    point_mask = cv2.dilate(hyst, kernel, iterations=1)
    p1 = cv2.goodFeaturesToTrack(image2, mask=point_mask, **feature_params)
    p0, st, err = cv2.calcOpticalFlowPyrLK(image2, image1, p1, None, **lk_params)

    # Select good points
    good1 = p1[st == 1]
    good0 = p0[st == 1]
    frame = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    # draw the tracks
    for i, (new, old) in enumerate(zip(good0, good1)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv2.line(frame, (a, b), (c, d), (255, 255, 0), 2)
        # frame = cv2.circle(frame, (c, d), 2, (255, 0, 0), -1)
    err = err/np.max(err)
    for p in p1[err < 0.4]:
        frame = cv2.circle(frame, (p[0], p[1]), 2, (0, 255, 0), -1)
    for p in p1[err > 0.4].reshape(-1, 2):
        frame = cv2.circle(frame, (p[0], p[1]), 2, (0, 0, 255), -1)

    print()
    print(len(p1[st == 1]))
    print(len(p1[st == 0]))
    cv2.imshow("edges", frame * cv2.cvtColor(hyst // 255, cv2.COLOR_GRAY2BGR))
    cv2.imshow('frame', frame * cv2.cvtColor(point_mask // 255, cv2.COLOR_GRAY2BGR))
    cv2.imshow("frameline", frame)
    cv2.waitKey()

    # END WORK ON OPTICAL FLOW

    out = extra_processing(hyst)

    return out


def background_subtraction(SobelImage1, SobelImage2):
    global stepByStepImages

    out = (SobelImage2 - SobelImage1)
    out[out < 0] = 0

    if stepByStepImages:
        norm1 = cv2.normalize(SobelImage1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        norm2 = cv2.normalize(SobelImage2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        norm3 = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow("start1 - achtergrond subtractie", norm1)
        cv2.imshow("start2", norm2)
        cv2.imshow("resultaat achtergrond subtractie", norm3)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return out


def find_threshold(image):
    # https://pdfs.semanticscholar.org/ccbd/40c5d670f4532ad1a9f0003a8b8157388aa0.pdf
    no_zero = image[image != 0]
    med = np.median(no_zero.ravel())
    mad = np.median(np.abs(no_zero - med).ravel())
    return med + 6 * mad


def hysteresis_threshold(image, low, high):
    # max = cv2.minMaxLoc(image)[1]
    # low = np.max([max / 17, 0.03])  # 0.05
    # high = np.max([max / 6, 0.1])  # 0.15
    out = filters.apply_hysteresis_threshold(image, low, high).astype(int)
    out = (out * 255).astype(np.uint8)

    if stepByStepImages:
        image = cv2.normalize(image, -1, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow("input - hysteresis", image)
        cv2.imshow("output - hysteresis", out)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return out


def extra_processing(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    if stepByStepImages:
        norm1 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        norm2 = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow("input extra processing", norm1)
        cv2.imshow("resultaat extra processing", norm2)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return result


def pre_processing_gray_images(image1, image2):
    return cv2.GaussianBlur(image1, (3, 3), -1), cv2.GaussianBlur(image2, (3, 3), -1)


# Bepaal de X en Y Sobel operator
def SobelOperator(inputImage):
    edges_image1_y = cv2.Sobel(inputImage, cv2.CV_32F, 0, 1, -1, -1)
    edges_image1_x = cv2.Sobel(inputImage, cv2.CV_32F, 1, 0, -1, -1)

    magnitude = cv2.magnitude(edges_image1_x, edges_image1_y)
    # normalized = cv2.normalize(magnitude, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return magnitude


edge_based_method()
