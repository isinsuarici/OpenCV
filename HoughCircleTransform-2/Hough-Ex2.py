import cv2 as cv
import numpy as np
import argparse

# konsoldan çalıştırmak için:
# python Hough-Ex2.py -i ../input_pictures/input_bobin.png

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path of image")
args = vars(ap.parse_args())

img = cv.imread(args["image"])
img2 = img.copy()
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.GaussianBlur(img, (7, 7), 1.5)
img = cv.Canny(img, 50, 120)

cv.imshow("image with blur and grayscale", img)
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

# param1 : Upper threshold for the internal Canny edge detector.
# param2 : Threshold for center detection.


# dp 0 ile 2 arasında double sayılar olmalı
# param2 küçükse yanlış daireler de algılıyor
circles = np.uint16(np.around(circles))

for c in circles[0, :]:
    print(c)  # x, y, r
    cv.circle(img2, (c[0], c[1]), c[2], (0, 255, 0), 2)
    cv.circle(img2, (c[0], c[1]), 1, (0, 0, 255), 3)

cv.imshow(" output img", img2)
# cv.imshow("img", np.hstack([img, img2]))
cv.waitKey(0)
cv.imwrite("houghcircle-ex2.png", img2)
