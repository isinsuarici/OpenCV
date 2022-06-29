import cv2 as cv

img = cv.imread('../input_pictures/input_rende.png')


""" BLURRING/ SMOOTHING **************************************************************** """
# 1. Averaging
blur_averaging = cv.blur(img, (5, 5))
cv.imwrite('blur_averaging.png', blur_averaging)
# 2. Gaussian Blurring
blur_gaussian = cv.GaussianBlur(img, (5, 5), 0)
cv.imwrite('blur_gaussian.png', blur_gaussian)
# 3. Median Blurring
blur_median = cv.medianBlur(img, 5)
cv.imwrite('blur_median.png', blur_median)
# 4. Bilateral Filtering
blur_bilateral = cv.bilateralFilter(img, 9, 75, 75)
cv.imwrite('blur_bilateral.png', blur_bilateral)