import cv2 as cv
import math

img = cv.imread("../input_bobins/with/1.png")
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

# polara çevir:
h, w = img.shape[:2]
img_center = (h/2, w/2)
img_radius = math.hypot(h/2, w/2)

cart_2_polar_flag = cv.WARP_FILL_OUTLIERS
img_forth = cv.linearPolar(img, img_center, img_radius, cart_2_polar_flag)
cv.imshow("linear image", img_forth)
cv.waitKey(0)

# kartezyene geri çevir:
polar_2_cart_flag = cv.WARP_INVERSE_MAP
img_back = cv.linearPolar(img_forth, img_center, img_radius, polar_2_cart_flag)
cv.imshow("back to cartesian image", img_back)
cv.waitKey(0)
