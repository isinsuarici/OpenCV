import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

""" Color changing - Grayscale ****************************************************************"""
img = cv.imread('input_interstellar.png')
gray2 = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)  # color to color
cv.imwrite('deneme.png', gray2)
# veya görseli alırken parametrede 0 vererek direkt grayscale yapabiliriz:
gray = cv.imread("input_interstellar.png", 0)

""" SCALING **************************************************************** """

print(img.shape)
img_resize = cv.resize(img, None, 2, 2, cv.INTER_CUBIC)  # INTER_CUBIC => zoom
cv.imwrite('img_resize.png', img_resize)
print(img_resize.shape)

print(img.shape)
height, width = img.shape[:2]
dec = cv.resize(img, (int(50 * width / 100), int(50 * height / 100)), interpolation=cv.INTER_AREA)
# interpolasyon parametre olarak verilmezse default INTER_LINEAR kullanıyor.
# görüntü büyütülürken arada boş pikseller kalıyor o kısımlara ne konulacağına interpolasyon ile karar veriliyor.
cv.imwrite('img_resize2.png', dec)
print(dec.shape)

# cv.imshow("img",res)
# cv.waitKey(0)
# cv.destroyAllWindows()

change_height = cv.resize(img, (width, int(50 * height / 100)), interpolation=cv.INTER_AREA)
# cv.imshow("img",change_height)
# cv.waitKey(0)
# cv.destroyAllWindows()

change_weight = cv.resize(img, (int(50 * width / 100), height), interpolation=cv.INTER_AREA)
# cv.imshow("img",change_weight)
# cv.waitKey(0)
# cv.destroyAllWindows()

change_withSpecs = cv.resize(img, (100, 100), interpolation=cv.INTER_AREA)
# cv.imshow("img",change_withSpecs)
# cv.waitKey(0)
# cv.destroyAllWindows()

""" TRANSLATION **************************************************************** """
M = np.float32([[1, 0, 50], [0, 1, 150]])
dist = cv.warpAffine(img, M, (width, height))  # parametreler = image, kaydırma miktarı, image boyutu
# cv.imshow("img",dist)
# cv.waitKey(0)
# cv.destroyAllWindows()

""" ROTATION **************************************************************** """
M = cv.getRotationMatrix2D(((width - 1) / 2.0, (width - 1) / 2.0), 90, 1)
# getRotationMatrix2D parametreleri = döndürme merkezi, döndürülecek açı ( pozitifse saat yönünün tersinde ), scale
dist_rotate = cv.warpAffine(img, M, (height, width))
# cv.imshow("img",dist_rotate)
# cv.waitKey(0)
# cv.destroyAllWindows()


""" THRESHOLDING **************************************************************** """
# 1.Simple Thresholding / Global Thresholding
x, thresh_simple = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
thresh_simpleLowThresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
# threshold parametreleri =
# image, thresh değeri (bu değer aşılmazsa pixel 0 olacak), threshi geçerse 255 olacak, thresh types.
# Thresholding types
# cv.THRESH_BINARY
# cv.THRESH_BINARY_INV
# cv.THRESH_TRUNC
# cv.THRESH_TOZERO
# cv.THRESH_TOZERO_INV
cv.imwrite('thresh_simple.png', thresh_simple)

titles = ['Input Image', 'THRESH_BINARY', "THRESH_BINARY Low Threshold"]
images = [gray, thresh_simple, thresh_simpleLowThresh]
for i in range(3):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# 2.Adaptive Thresholding
# simple thresholdingten farklı olarak, adaptive thresholding değişen ortam ışıklarına uyum sağlayabilir.
# örn fotonun solunda ışık var ama sağında yoksa soldaki threshold değeri sağdakinden farklı olacak.
img_blur = cv.medianBlur(gray, 5)
cv.imwrite('img_blur.png', img_blur)
x, thresh_binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
cv.imwrite('thresh_binary.png', thresh_binary)
thresh_adaptive_mean = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 23, 2)
cv.imwrite('thresh_adaptive_mean.png', thresh_adaptive_mean)
thresh_adaptive_gaussian = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 23, 2)
cv.imwrite('thresh_adaptive_gaussian.png', thresh_adaptive_gaussian)

titles = ['Input Image', 'thresh adaptive', "thresh adaptive gaussian"]
images = [gray, thresh_adaptive_mean, thresh_adaptive_gaussian]
for i in range(3):
    plt.subplot(1, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# adaptive threshold parametreleri:
# source, dest, max value ( koşulun sağlandığı piksele atanan değer ),
# adaptive metod, blockSize = piksel için threshold değeri belirlemek için kaç komşusuna bakılacağı(3,5,7 gibi),
# C= ortalamadan çıkarılan sabit? ( genelde pozitif )

# 3.Otsu's Binarization
ret2, thresh_otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imwrite('thresh_otsu.png', thresh_otsu)

# gürültüyü azaltmak için önce blurlayıp ardından threshleyebiliriz.
blur = cv.GaussianBlur(gray, (3, 3), 0)
# gaussianblur, gauss filtresi kullanarak blurlar
ret3, thresh_otsu_blur = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imwrite('thresh_otsu_blur.png', thresh_otsu_blur)

titles = ['Input Image', 'thresh otsu', "thresh otsu blur"]
images = [gray, thresh_otsu, thresh_otsu_blur]
for i in range(3):
    plt.subplot(1, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

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

# MORPHOLOGICAL TRANSFORMATIONS
