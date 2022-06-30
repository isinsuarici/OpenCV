import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Histogram, bir görüntünün yoğunluk dağılımı hakkında bilgi veren bir grafiktir.
# X ekseninde piksel değerleri vardır. (Genellikle 0-255 arasındadır.)
# y ekseninde görüntüde karşılık gelen piksel sayısı vardır.
# Bir görüntünün histogramına bakarak, görüntünün kontrastı, parlaklığı,
# yoğunluk dağılımı vs hakkında az çok bilgiye sahip oluruz.
# 0-> siyah, 255-> beyaz
# histogram, renkli görüntü için değil, grayscale için çizilir.


# Histogram terminolojisi:
# BINS = her bir piksel için piksel sayısını gösterir. 0'dan 255'e kadar.
# Pikselleri belirli şekilde kümelemek istiyorsak örn 15-15-15-15...
# şeklinde göstersin hepsini tek tek göstermek yerine.
# Bu şekilde sadece 16 değer ile temsil edebiliriz.
# Her bir değerin içinde, içindeki tüm piksellerin değerinin toplamı var.
# Bu her alt parçaya BIN denir. İlk durumda bin sayısı 256, 2.durumda 16
# OpenCV'de BINS'e histSize denir.
# DIMS = Verilerini topladığımız parametre sayısı. Bu durumda,
# yalnızca yoğunluk değeriyle ilgili verileri topluyoruz yani 1
# RANGE = Yoğunluk değerleri aralığı. Normalde [0,256]

# cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
# images = source köşeli parantez içinde verilmeli
# channels = kanalın indeksi. Köşeli parantez içinde verilmeli.
# channels değeri grayscale için 0, mavi, yeşil, kırmızı için 0, 1, 2
# mask = tam görüntünün histogramını bulmak için None olarak verilir.
# Görüntünün belirli bir bölgesinin histogramını bulmak için kendimiz mask oluşturmalıyız.
# histSize = BIN sayısını gösterir. Köşeli parantez içinde verilmeli.
# ranges = range. Normalde [0,256] olur.

# histogram calculation in openCV
img = cv.imread("../input_pictures/input_interstellar.png", 0)
hist = cv.calcHist([img], [0], None, [256], [0, 256])

# histogram calculation in numpy
hist2, bins = np.histogram(img.ravel(), 256, [0, 256])
# numpyda 257 bins var.( Numpy, aralık olarak hesapladığından 1 fazla çıkıyor.)

# Histogram çizmek için kısa yol matplotlib, uzun yol opencv

# matplotlib -  matplotlib.pyplot.hist()
img = cv.imread("../input_pictures/input_golgeAgac.png", 0)
plt.hist(img.ravel(), 256, [0, 256])
plt.show()
# veya manuel olarak şu yöntemi deneyebiliriz:
img2 = cv.imread("../input_pictures/input_interstellar.png")
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv.calcHist([img2], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()

# openCV ile yaparsak:
# tam görüntünün histogramını en başta cv.calcHist ile bulmuştuk.
# görüntünün bazı bölgelerinin histogramını bulmak istersek,
# histogramını bulmak istediğimiz bölgede beyaz renkli, diğer taraflarda
# siyah olan bir maske oluşturmak gerekiyor. Sonra bunu mask olarak kullanıcaz.

img = cv.imread("../input_pictures/input_golgeAgac.png", 0)

# mask yaratmak için:
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv.bitwise_and(img, img, mask=mask)
# histogramı mask ile ve masksız hesaplayalım:
hist_full = cv.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv.calcHist([img], [0], mask, [256], [0, 256])

# mavi çizgi tüm hist için, turuncu belirlediğimiz bölge için.
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0, 256])
plt.show()

#devam et
