import cv2 as cv
import numpy as np
import argparse

# konsoldan çalıştırmak için:
# python Hough-Ex2.py -i ../input_pictures/input_multiplecircles5.png

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path of image")
args = vars(ap.parse_args())

img = cv.imread(args["image"])
img2 = img.copy()
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.GaussianBlur(img, (5, 5), 1.5)
kernel = np.ones((3, 3), np.uint8)

img = cv.Canny(img, 50, 120)
cv.imwrite('img_morph4.png', img)
print(img.shape)
h, w = img.shape[:]

cv.imshow("image", img)
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1.4, int(w / 25), param1=40, param2=25, minRadius=int(w / 30),
                          maxRadius=int(w / 15))


circles = np.uint16(np.around(circles))

for c in circles[0, :]:
    print(c)
    cv.circle(img2, (c[0], c[1]), c[2], (0, 255, 0), 2)
    cv.circle(img2, (c[0], c[1]), 1, (0, 0, 255), 3)

cv.imshow(" output img", img2)
cv.waitKey(0)
cv.imwrite("houghcircle-ex6.png", img2)
