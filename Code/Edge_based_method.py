import cv2
import numpy as np
from skimage import filters
from segmentatie import segmentatie

# Deze waarde op True = Laat de tussenstapafbeelding zien
# Maakt het makkelijk om de verschillende stappen te begrijpen
stepByStepImages = False


def main():
    # image1 = cv2.imread("Images/highway/input/in000800.jpg")
    # image2 = cv2.imread("Images/highway/input/in000805.jpg")
    #
    # output = edge_based_method(image1, image2)
    #
    # cv2.imshow("Start afbeelding 1", image1)
    # cv2.imshow("Start afbeelding 2", image2)
    # cv2.imshow("globaal resultaat1", output)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    image1 = cv2.imread("Images/pedestrians/input/in000470.jpg")
    image2 = cv2.imread("Images/pedestrians/input/in000475.jpg")
    image3 = cv2.imread("Images/pedestrians/input/in000480.jpg")
    output = edge_based_method_3(image1, image2, image3)

    cv2.imshow("Start afbeelding 1", image1)
    cv2.imshow("Start afbeelding 2", image2)
    cv2.imshow("Start afbeelding 3", image3)
    cv2.imshow("globaal resultaat1", output)
    cv2.waitKey()
    cv2.destroyAllWindows()


def edge_based_method_3(image1, image2, image3):
    """
    uitbreiding van edge_based_method naar dre frames
    ghostranden worden verwijderd door and-operatie van de randen
    :param image1: frame 1
    :param image2: frame 2
    :param image3: frame 3
    :return: segmentatie van frame 2
    """
    # omzetting naar grijswaarden
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
    # blurring
    image1 = pre_processing_gray_images(image1)
    image2 = pre_processing_gray_images(image2)
    image3 = pre_processing_gray_images(image3)
    # randen via sobel
    SobelImage1 = SobelOperator(image1)
    SobelImage2 = SobelOperator(image2)
    SobelImage3 = SobelOperator(image3)
    # verschilbeeld van randen
    difference12 = background_subtraction(SobelImage1, SobelImage2)
    difference32 = background_subtraction(SobelImage3, SobelImage2)
    # hysteresis thresholding
    threshold12 = find_threshold(difference12)
    hyst12 = hysteresis_threshold(difference12, threshold12, threshold12 * 3)
    threshold32 = find_threshold(difference32)
    hyst32 = hysteresis_threshold(difference32, threshold32, threshold32 * 3)
    # closing
    out12 = extra_processing(hyst12)
    out32 = extra_processing(hyst32)
    # interior filling van de and-operatie
    return segmentatie(np.bitwise_and(out12, out32))


def edge_based_method(image1, image2):
    """
    randgebaseerd methode voor bewegingsdetectie
    :param image1: frame 1
    :param image2: frame 2
    :return: segmentatie van frame 2
    """
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    image1 = pre_processing_gray_images(image1)
    image2 = pre_processing_gray_images(image2)

    # Bepaal de randen van beide startafbeeldingen (de magnitude)
    SobelImage1 = SobelOperator(image1)
    SobelImage2 = SobelOperator(image2)

    # Achtergrond subtractie waardoor bewegende randen van afbeelding 2 tevoorschijn komen
    difference = background_subtraction(SobelImage1, SobelImage2)

    # hysteresis thresholding
    threshold = find_threshold(difference)
    hyst = hysteresis_threshold(difference, threshold, threshold * 3)

    # closing
    out = extra_processing(hyst)

    # interior filling
    out = segmentatie(out)

    return out


def pre_processing_gray_images(image1):
    """
    blurring
    """
    return cv2.GaussianBlur(image1, (3, 3), -1)


def SobelOperator(inputImage):
    """
    Bepaal de gradient via Sobel operator
    """
    edges_image1_y = cv2.Sobel(inputImage, cv2.CV_32F, 0, 1, -1, -1)
    edges_image1_x = cv2.Sobel(inputImage, cv2.CV_32F, 1, 0, -1, -1)

    magnitude = cv2.magnitude(edges_image1_x, edges_image1_y)
    return magnitude


def background_subtraction(SobelImage1, SobelImage2):
    """
    achtergrond subtractie
    enkel randen aanwezig in het tweede beeld en niet in het eerste beeld worden behouden
    """
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
    """
    automatische thresholdwaarde op basis van
    Image difference threshold strategies andshadow detection (Paul L. Rosin, Tim Ellis)
    (https://pdfs.semanticscholar.org/ccbd/40c5d670f4532ad1a9f0003a8b8157388aa0.pdf)
    """
    no_zero = image[image != 0]
    med = np.median(no_zero.ravel())
    mad = np.median(np.abs(no_zero - med).ravel())
    return med + 6 * mad


def hysteresis_threshold(image, low, high):
    """
    hysteresis thresholding met low threshold en high threshold
    """
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
    """
    closing toegepast op image
    """
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


if __name__ == "__main__":
    stepByStepImages = False
    main()
