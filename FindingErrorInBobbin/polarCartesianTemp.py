import cv2 as cv
import math

img = cv.imread("../input_bobins/pure/9.png")
cv.imshow("Original image", img)
'''
# resize:
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
'''
# polara çevir:
h, w = img.shape[:2]
img_center = (h / 2, w / 2)
img_radius = math.hypot(h / 2, w / 2)

cart_2_polar_flag = cv.WARP_FILL_OUTLIERS
img_forth = cv.linearPolar(img, img_center, img_radius, cart_2_polar_flag)
cv.imshow("linear image", img_forth)


'''
# düz basit thresholding
img_blur = cv.medianBlur(img_forth, 3)
# cv.imshow('img_blur.png', img_blur)
x, thresh_binary = cv.threshold(img_blur, 127, 255, cv.THRESH_BINARY)
cv.imshow('thresh_binary.png', thresh_binary)
cv.waitKey(0)
'''
# adaptive thresholding - gaussian
img_forth = cv.cvtColor(img_forth, cv.COLOR_BGR2GRAY)
img_forth = cv.medianBlur(img_forth, 3)  # noiseu azaltmak için
thresh_adaptive_gaussian = cv.adaptiveThreshold(img_forth, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 29, 2)
cv.imshow('thresh_adaptive_gaussian.png', thresh_adaptive_gaussian)


# adaptive thresholding - mean
thresh_adaptive_mean = cv.adaptiveThreshold(img_forth, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 29, 2)
cv.imshow('thresh_adaptive_mean.png', thresh_adaptive_mean)


# kartezyene geri çevir:
polar_2_cart_flag = cv.WARP_INVERSE_MAP
img_back = cv.linearPolar(thresh_adaptive_mean, img_center, img_radius, polar_2_cart_flag)
cv.imshow("back to cartesian image", img_back)
cv.waitKey(0)