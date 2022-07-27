import cv2 as cv
import numpy as np

# bobinleri HCT ile kesiyoruz:
img = cv.imread("../input_bobins/with/1.png")

print('Original Dimensions : ', img.shape)
if img.shape[0] > 3000 or img.shape[1] > 3000:
    scale_percent = 10
elif img.shape[0] > 1500 or img.shape[1] > 1500:
    scale_percent = 20
else:
    scale_percent = 60
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

print('Resized Dimensions : ', img.shape)

cv.imshow("Resized image", img)
cv.waitKey(0)
cv.destroyAllWindows()

h, w, b = img.shape[:]

img2 = img.copy()
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.GaussianBlur(img, (3, 3), 1.5)  # 9,9 ?
cv.imshow("gri ve blurlu img", img)
cv.waitKey(0)

circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1.3,
                          int(2*h), param1=90, param2=40, minRadius=int(h / 2.4),
                          maxRadius=int(h/1.8))

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv.circle(img2, (x, y), r, (36, 255, 12), 3)

cv.imshow("img", img2)
cv.waitKey(0)
cv.imwrite("bobin-1.png", img2)
