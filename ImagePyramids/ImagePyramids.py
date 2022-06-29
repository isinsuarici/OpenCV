import cv2 as cv

# Normalde sabit boyutlu bir görüntü ile çalışıyorduk.
# Ancak bazı durumlarda aynı görüntünün farklı çözünürlükleriyle çalışmamız gerekir.
# Örneğin, bir görüntüde yüz arıyorsak o görüntüdeki nesnenin hangi boyutta olacağından
# emin olamayız. Bu durumda, aynı görüntüden farklı çözünürlüklerde bir set oluşturup
# hepsinde nesne aramamız gerekir. Bu görüntü kümesine Görüntü Piramitleri denir.
# ( en yüksek çözünürlüklü görüntü altta, en düşük çözünürlüklü görüntü üstte. )

""" 1. Gaussian Pyramids *********************************"""

gray = cv.imread("../input_pictures/input_agac.png")
lower_reso = cv.pyrDown(gray)  # level_1
cv.imwrite("lower_reso.png", lower_reso)

lower_lower_reso = cv.pyrDown(lower_reso)  # level_2
cv.imwrite("lower_lower_reso.png", lower_lower_reso)

high_reso = cv.pyrUp(lower_lower_reso)
cv.imwrite("high_reso.png", high_reso)

# high_reso ile baştaki image aynı değil çünkü çözünürlüğü bir kez düşürdüğümüzde bilgiyi kaybederiz.

""" 2. Laplace Pyramids *********************************"""
# Laplacian piramitleri, gauss piramitlerinden oluşur. Bunun için özel bir fonksiyon yok.
# Laplacian piramit görüntüleri, yalnızca kenar görüntüleri gibidir.
# Öğelerinin çoğu sıfırdır.
# Görüntü sıkıştırmada kullanılır.
# Laplacian'daki bir seviye, Gauss'daki bu seviye ile Gauss'daki üst
# seviyesinin genişletilmiş versiyonu arasındaki farktan oluşur.
