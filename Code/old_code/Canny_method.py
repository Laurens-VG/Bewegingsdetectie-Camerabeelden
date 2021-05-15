import cv2
import numpy as np


def ImageDiff2(image1, image2):
    image1_grey = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_grey = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    edges_image1_1 = cv2.Sobel(image1_grey, cv2.CV_32F, 0, 1)
    edges_image1_2 = cv2.Sobel(image1_grey, cv2.CV_32F, 1, 0)
    result1 = cv2.magnitude(edges_image1_1, edges_image1_2)

    edges_image2_1 = cv2.Sobel(image2_grey, cv2.CV_32F, 0, 1)
    edges_image2_2 = cv2.Sobel(image2_grey, cv2.CV_32F, 1, 0)
    result2 = cv2.magnitude(edges_image2_1, edges_image2_2)

    result = result1 - cv2.bitwise_and(result1, result2)

    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    x, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY)

    # Filters
    kernel = np.ones((3, 3), np.uint8)
    # img = cv2.GaussianBlur(result, (3, 3), 0)
    # img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    # img = cv2.erode(img, kernel, iterations=1)
    # img = cv2.dilate(img, kernel, iterations=1)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    # Contour
    # contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (255, 255, 0), cv2.FILLED)

    # Resultaat
    cv2.imshow("Image Differencing", result)
    cv2.waitKey()
    cv2.destroyAllWindows()


def Canny(image1, image2):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    image1 = cv2.GaussianBlur(image1, (3, 3), -1)
    image2 = cv2.GaussianBlur(image2, (3, 3), -1)

    image1 = cv2.Canny(image1, 150, 200)
    image2 = cv2.Canny(image2, 150, 200)
    result = image1 - (np.bitwise_and(image1, image2))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    contours = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    draw = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    draw = cv2.drawContours(draw, contours, -1, (0, 0, 255), 1)
    cv2.imshow("Canny", np.hstack((cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), draw)))
    cv2.waitKey()
    cv2.destroyAllWindows()


def Groundtruth(image1, image2):
    contours1 = cv2.findContours(image1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image1 = cv2.drawContours(image1, contours1, -1, (0, 0, 255), 1)

    contours2 = cv2.findContours(image2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image2 = cv2.drawContours(image2, contours2, -1, (0, 0, 255), 1)

    add = cv2.add(image1, image2)

    cv2.imshow("Groundtruth", np.hstack((image1, image2, add)))
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    image1 = cv2.imread("Images/highway/groundtruth/gt000848.png")
    image2 = cv2.imread("Images/highway/groundtruth/gt000873.png")
    image3 = cv2.imread("Images/in000968.jpg")
    image4 = cv2.imread("Images/in000843.jpg")

    # ImageDiff2(image3, image4)
    # Canny(image3, image4)
    Groundtruth(image1, image2)


if __name__ == "__main__":
    main()
