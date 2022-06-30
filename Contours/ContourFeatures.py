import cv2 as cv
import numpy as np

img = cv.imread("../input_pictures/input_altigen.png", 0)
cv.imshow("img", img)
cv.waitKey(0)
ret, thresh = cv.threshold(img, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv.moments(cnt)
print(M)

# The function cv.moments() gives a dictionary of all moment values calculated.
# merkezi hesaplamak için:

cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
print("cx: " + str(cx) + " cy: " + str(cy))

# alan
area = cv.contourArea(cnt)
print("area: " + str(area))

perimeter = cv.arcLength(cnt, True)
print("perimeter: " + str(perimeter))

# contour approximation
epsilon = 0.1 * cv.arcLength(cnt, True)
approx = cv.approxPolyDP(cnt, epsilon, True)

# konveks olup olmadığını kontrol etmek için, true false dönüyor.
k = cv.isContourConvex(cnt)
print(k)

# straight bounding rectangle
# nesnenin çevresine dikdörtgen çizmek için:
x, y, w, h = cv.boundingRect(cnt)
cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv.imshow("result", img)
cv.waitKey(0)
cv.destroyAllWindows()

# rotated rectangle ( dönüş yönünü de dikkate alır )
# nesnenin çevresine nesneyi kapsayan en küçük dikdörtgeni çizmek için:
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img, [box], 0, (0, 0, 255), 7)
cv.imshow("result2", img)
cv.waitKey(0)
cv.destroyAllWindows()

# minimum enclosing circle
(x, y), radius = cv.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
cv.circle(img, center, radius, (0, 0, 255), 2)
cv.imshow("result3", img)
cv.waitKey(0)
cv.destroyAllWindows()

# fitting an ellipse
ellipse = cv.fitEllipse(cnt)
cv.ellipse(img, ellipse, (0, 255, 0), 2)
cv.imshow("result4", img)
cv.waitKey(0)
cv.destroyAllWindows()

# fitting a line
rows, cols = img.shape[:2]
[vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)
cv.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
cv.imshow("result5", img)
cv.waitKey(0)
cv.destroyAllWindows()
