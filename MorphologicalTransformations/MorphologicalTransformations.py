import cv2 as cv
import numpy as np

gray = cv.imread("../input_pictures/input_rende.png", 0)
cv.imwrite('gray.png', gray)

"""  MORPHOLOGICAL TRANSFORMATIONS **************************************************************** """
# 1.Erosion
# Convolution layere benziyor, bir filtre seçiyoruz görsel üzerinde dolaşıyor.
# Orijinal görüntüdeki bir piksel, yalnızca kernel altındaki tüm pikseller 1 ise 1 yapılır.
# ( Yani beyaz=1 yapma ihtimalimiz düşük. )
# Aksi taktirde 0 yapılır yani aşınır.
# Böylece beyaz gürültüden kurtuluruz ama görüntünün boyutu küçülmüş olur.
# ( Ön planda beyaz olan görüntüler için daha uygun. )

kernel = np.ones((9, 9), np.uint8)
erosion = cv.erode(gray, kernel, iterations=1)
cv.imwrite('erosion.png', erosion)

# 2.Dilation
# Erozyonun tam tersidir. Kernel altındaki en az bir piksel 1 ise değer 1 yapılır.
# Böylece görüntüdeki beyaz bölge artar veya ön plandaki nesnenin boyutu artar.
# Gürültü giderme yapılırken önce erozyon sonra dilation yapılır.
# Çünkü erozyon beyaz noiseleri kaldırır ama nesnenin boyutu da küçülmüş olur.
# Ardından dilation kullanarak nesneyi genişletiriz.

dilation = cv.dilate(gray, kernel, iterations=1)
cv.imwrite('dilation.png', dilation)

dilation_it3 = cv.dilate(gray, kernel, iterations=3)
cv.imwrite('dilation_it3.png', dilation_it3)

# 3.Opening
# erozyon ardından dilation
# noise gidermede faydalı.

opening = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
cv.imwrite('opening.png', opening)

# 4.Closing
# openingin tersidir.
# dilation ardından erozyon
# nesnenin içinde delik varsa onları kapatmada faydalı.

closing = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
cv.imwrite('closing.png', closing)

# 5.Morphological Gradient                                          !!!
# görüntüdeki dilation ve erozyon arasındaki farktır.
# output, objenin ana hatları olacaktır!

gradient = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel)
cv.imwrite('gradient.png', gradient)

# 6.Top Hat
# input image ile opening arasındaki farktır.
tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
cv.imwrite('tophat.png', tophat)

# 7.Black Hat
# input image ile closing arasındaki farktır.
blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
cv.imwrite("blackhat.png", blackhat)

""" Numpy ile kernel oluşturmaktansa kendimiz de istediğimiz şekillerde kernel oluşturabiliriz. """
# MORPH_ELLIPSE, MORPH_RECT, MORPH_CROSS
kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
gradient_ellipsekernel = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel_ellipse)
cv.imwrite('gradient_ellipsekernel.png', gradient_ellipsekernel)

# rect ile numpydan oluşturmak arasında fark yok.
kernel_rect = cv.getStructuringElement(cv.MORPH_RECT, (9,9))
gradient_rectkernel = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel_rect)
cv.imwrite('gradient_rectkernel.png', gradient_rectkernel)

kernel_cross = cv.getStructuringElement(cv.MORPH_CROSS, (4, 4))
gradient_crosskernel = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel_cross)
cv.imwrite('gradient_crosskernel.png', gradient_crosskernel)
