import cv2
import numpy as np
from segmentatie import segmentatie

image1 = cv2.imread("Images/highway/input/in000200.jpg")
image2 = cv2.imread("Images/highway/input/in000210.jpg")
image3 = cv2.imread("Images/highway/input/in000220.jpg")

image1 = cv2.GaussianBlur(image1, (3, 3), 1)
image2 = cv2.GaussianBlur(image2, (3, 3), 1)
image3 = cv2.GaussianBlur(image3, (3, 3), 1)

# cv2.imshow("test", np.hstack((image1, image2, image3)))

dif1 = np.abs(image2.astype(np.int32) - image1.astype(np.int32)).astype(np.uint8)
dif2 = np.abs(image2.astype(np.int32) - image3.astype(np.int32)).astype(np.uint8)

dif1 = cv2.GaussianBlur(dif1, (3, 3), 1)
dif2 = cv2.GaussianBlur(dif2, (3, 3), 1)
zeta1 = cv2.Canny(dif1, 70, 150)
zeta2 = cv2.Canny(dif2, 70, 150)
theta = np.bitwise_or(zeta1, zeta2)

image1 = cv2.Canny(image1, 70, 200)
image2 = cv2.Canny(image2, 70, 200)
image3 = cv2.Canny(image3, 70, 200)

DEl = np.bitwise_and(image2, np.bitwise_not(image1))
DEr = np.bitwise_and(image2, np.bitwise_not(image3))
ME = np.bitwise_and(DEl, DEr)

result = np.bitwise_and(ME, theta)

# cv2.imshow("test1", image1)
# cv2.imshow("test2", image2)
# cv2.imshow("ME", ME)
# cv2.imshow("DEl", DEl)
# cv2.imshow("DEr", DEr)
# cv2.imshow("zeta1", zeta1)
# cv2.imshow("zeta2", zeta2)
# cv2.imshow("theta", theta)
cv2.imshow("result", result)

cv2.imshow("segmentatie", segmentatie(result))

# kernel = np.ones((3, 3), np.uint8)
# result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
#
# contours = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
# draw = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
# draw = cv2.drawContours(draw, contours, -1, (0, 0, 255), 1)
# cv2.imshow("Canny", np.hstack((cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), draw)))
# cv2.imshow("result", result)


cv2.waitKey()
