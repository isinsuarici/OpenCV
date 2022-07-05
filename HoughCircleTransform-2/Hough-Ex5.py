import cv2 as cv
import numpy as np
import argparse

# konsoldan çalıştırmak için:
# python Hough-Ex2.py -i ../input_pictures/input_multiplecircles3.png

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path of image")
args = vars(ap.parse_args())

img = cv.imread(args["image"])
img2 = img.copy()
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.GaussianBlur(img, (9, 9), 1.5)
kernel = np.ones((2, 2), np.uint8)
img = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

cv.imwrite('img_morph3.png', img)
print(img.shape)
h, w = img.shape[:]

cv.imshow("image", img)
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1.3, int(w / 60), param1=100, param2=30, minRadius=int(w / 200),
                          maxRadius=int(w / 60))
# circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1.8, int(w/60), param1=50, param2=40, minRadius=int(w/120),
#                           maxRadius=int(w/100))


circles = np.uint16(np.around(circles))

for c in circles[0, :]:
    print(c)  # x, y, r
    cv.circle(img2, (c[0], c[1]), c[2], (0, 255, 0), 2)
    cv.circle(img2, (c[0], c[1]), 1, (0, 0, 255), 3)

cv.imshow(" output img", img2)
# cv.imshow("img", np.hstack([img, img2]))
cv.waitKey(0)
cv.imwrite("houghcircle-ex5.png", img2)
