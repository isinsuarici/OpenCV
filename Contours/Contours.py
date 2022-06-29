import cv2 as cv

gray = cv.imread("../input_pictures/input_agac.png")

# Konturlar, aynı renk veya yoğunluğa sahip tüm sürekli noktaları sınır boyunca birleştiren bir eğridir.
# Şekil analizi, nesne algılama, tanımada kullanılır.
# Binary görüntü kullanılırsa daha doğru sonuçlar elde edilir.
# Bu yüzden konturları bulmadan önce threshold veya canny edge detection kullanılır.
# Bulunacak nesne beyaz, arka plan siyah olmalı.