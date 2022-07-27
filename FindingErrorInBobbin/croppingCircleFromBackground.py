import cv2
import numpy as np
from matplotlib import pyplot as plt
# sondan 2. 27.07 öncesi
# bobinler HCT ile kesip arka planı siyah yapıyoruz ardından kontrast ve parlaklık ayarları.

img = cv2.imread("../input_bobins/with/1.png")
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
cv2.destroyAllWindows()

height, width, x = img.shape
mask = np.zeros((height, width), np.uint8)

img2 = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 1.5)  # 9,9 ?
cv2.imshow("gri ve blurlu img", img)
cv2.waitKey(0)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.3,
                           int(2 * height), param1=90, param2=40, minRadius=int(height / 2.4),
                           maxRadius=int(height / 1.8))

for i in circles[0, :]:
    i[2] = i[2] + 4
    # Draw on mask
    cv2.circle(mask, (int(i[0]), int(i[1])), int(i[2]), (255, 255, 255), thickness=-1)

# Copy that image using that mask
masked_data = cv2.bitwise_and(img, img, mask=mask)

# Apply threshold
_, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

# Find contour
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # [-2:]
x, y, w, h = cv2.boundingRect(contours[0])

# Crop masked_data
crop = masked_data[y:y + h, x:x + w]

cv2.imshow('Cropped img', crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

####
height, width = crop.shape
print(crop.shape)
mask2 = np.zeros((height, width), np.uint8)

circles2 = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, 1.3,
                            int(2 * height), param1=30, param2=40, minRadius=int(height / 20),
                            maxRadius=int(height / 1.8))

for i in circles2[0, :]:
    i[2] = i[2] + 4
    cv2.circle(mask2, (int(i[0]), int(i[1])), int(i[2]), (255, 255, 255), thickness=-1)

masked_data2 = cv2.bitwise_and(crop, crop, mask=mask2)

_, thresh2 = cv2.threshold(mask2, 1, 255, cv2.THRESH_BINARY)

contours2, hier2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # [-2:]
x, y, w, h = cv2.boundingRect(contours2[0])

crop2 = masked_data2[y:y + h, x:x + w]

cv2.imshow('Cropped img2', crop2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# kontrast ve parlaklık ayarı:
# alpha = 1.5 # Contrast control (1.0-3.0)
# beta = 10 # Brightness control (0-100)
#
# adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
#
# cv2.imshow('original', img)
# cv2.imshow('adjusted', adjusted)
# cv2.waitKey()

crop2 = cv2.cvtColor(crop2, cv2.COLOR_GRAY2BGR)
imghsv = cv2.cvtColor(crop2, cv2.COLOR_BGR2HSV)

imghsv[:, :, 2] = [[max(pixel - 25, 0) if pixel < 190 else min(pixel + 25, 255) for pixel in row] for row in
                   imghsv[:, :, 2]]

img = cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)
cv2.imshow('contrast', img)
cv2.waitKey()

'''
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("imggray",img_gray)
cv2.waitKey(0)
mask_spec = np.zeros_like(img_gray)
# get average color with mask
ave_color = cv2.mean(img_gray, mask=mask_spec)[:3]
print("average circle color:", ave_color)
'''


