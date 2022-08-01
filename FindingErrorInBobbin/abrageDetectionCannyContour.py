import cv2
import numpy as np
#  gray scale -> canny -> contour -> HCT
img = cv2.imread('../input_bobins/pure/3.png')

cv2.imshow("Original image", img)
cv2.waitKey(0)

h, w, b = img.shape[:]

gray2 = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
cannyEdge2 = cv2.Canny(gray2, 60, 70)
contours2, hierarchy2 = cv2.findContours(cannyEdge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours2, -1, (0, 255, 255), 3)
cv2.imshow("contoursWith_CHAIN_APPROX_SIMPLE.png", img)
cv2.waitKey(0)
'''
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, int(h), param1=50, param2=60, minRadius=int(h / 9),
                           maxRadius=int(h / 1.5))

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(img, (x, y), r, (36, 255, 12), 3)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.imwrite("bobin-1.png", img)
'''