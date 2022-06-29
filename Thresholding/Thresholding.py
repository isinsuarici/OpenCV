import cv2 as cv

from matplotlib import pyplot as plt

gray = cv.imread("../input_pictures/input_rende.png", 0)
cv.imwrite('gray.png', gray)

""" THRESHOLDING **************************************************************** """
# 1.Simple Thresholding / Global Thresholding
x, thresh_simple = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
y, thresh_simpleLowThresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
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
    plt.subplot(1, 3, i+1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
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
