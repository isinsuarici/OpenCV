import cv2 as cv
import numpy as np
import argparse

# konsoldan çalıştırmak için:
# python Hough-Ex2.py -i ../input_pictures/input_multiplecircles4.png

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path of image")
args = vars(ap.parse_args())

img = cv.imread(args["image"])
img2 = img.copy()
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
kernel = np.ones((3, 3), np.uint8)
x, img = cv.threshold(img, 240, 255, cv.THRESH_BINARY_INV)
kernel = np.ones((3, 3), np.uint8)
img = cv.erode(img, kernel, iterations=2)
kernel = np.ones((3, 3), np.uint8)
img = cv.dilate(img, kernel, iterations=3)

cv.imwrite('img_morph7.png', img)
print(img.shape)
h, w = img.shape[:]

cv.imshow("image", img)
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1.9, int(w / 50), param1=40, param2=20, minRadius=int(w / 300),
                          maxRadius=int(w / 30))


circles = np.uint16(np.around(circles))

for c in circles[0, :]:
    print(c)
    cv.circle(img2, (c[0], c[1]), c[2], (0, 255, 0), 2)
    cv.circle(img2, (c[0], c[1]), 1, (0, 0, 255), 1)

cv.imshow(" output img", img2)
cv.waitKey(0)
cv.imwrite("houghcircle-ex7.png", img2)
