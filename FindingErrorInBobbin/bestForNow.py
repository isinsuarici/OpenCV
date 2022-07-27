import cv2
import numpy as np

# en son 27.07

img = cv2.imread('../input_bobins/with/1.png')
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

img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

print('Resized Dimensions : ', img.shape)

cv2.imshow("Resized image", img)
cv2.waitKey(0)
hh, ww = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.GaussianBlur(img, (3, 3), 1.5)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.3,
                           int(2 * height), param1=90, param2=40, minRadius=int(height / 2.4),
                           maxRadius=int(height / 1.8))
print("circles:", circles)

img_circle = img.copy()
mask = np.zeros_like(gray)
for circle in circles[0]:
    (x, y, r) = circle
    x = int(x)
    y = int(y)
    r = int(r)
    cv2.circle(img_circle, (x, y), r, (0, 0, 255), 2)
    cv2.circle(mask, (x, y), r, 255, -1)

# get average color with mask
ave_color = cv2.mean(img, mask=mask)[:3]
print("average circle color:", ave_color)  # BGR

# show images
cv2.imshow('circle', img_circle)
cv2.imshow('mask', mask)
cv2.waitKey(0)
b, g, r = ave_color[:3]

x, thresh_simple = cv2.threshold(gray, 71.17, 255, cv2.THRESH_BINARY)

cv2.imshow('circle result', thresh_simple)
cv2.waitKey(0)

kernel = np.ones((5, 5), np.uint8)
gradient = cv2.morphologyEx(thresh_simple, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('gradient', gradient)
cv2.waitKey(0)
kernel = np.ones((9, 9), np.uint8)
erosion = cv2.erode(gradient, kernel, iterations=1)
cv2.imshow('erosion', erosion)
cv2.waitKey(0)

ms = cv2.subtract(thresh_simple, gradient)
cv2.imshow('ms', ms)
cv2.waitKey(0)

kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(ms, kernel, iterations=1)
cv2.imshow('ress', dilation)
cv2.waitKey(0)
