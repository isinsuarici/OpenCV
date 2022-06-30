import cv2 as cv

# findContours, drawContours
img = cv.imread("../input_pictures/input_rect.png")
img2 = img.copy()
# Konturlar, aynı renk veya yoğunluğa sahip tüm sürekli noktaları sınır boyunca birleştiren bir eğridir.
# Şekil analizi, nesne algılama, tanımada kullanılır.
# Binary görüntü kullanılırsa daha doğru sonuçlar elde edilir.
# Bu yüzden konturları bulmadan önce threshold veya canny edge detection kullanılır.
# Bulunacak nesne beyaz, arka plan siyah olmalı.

"""findContours"""
# findContours with CHAIN_APPROX_NONE:
gray = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
cannyEdge = cv.Canny(gray, 30, 200)
contours, hierarchy = cv.findContours(cannyEdge, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)


# findContours with CHAIN_APPROX_SIMPLE:
gray2 = cv.cvtColor(img2, cv.COLOR_RGBA2GRAY)
cannyEdge2 = cv.Canny(gray2, 30, 200)
contours2, hierarchy2 = cv.findContours(cannyEdge2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img2, contours2, -1, (0, 255, 0), 3)
cv.imwrite("contoursWith_CHAIN_APPROX_SIMPLE.png", img2)

# findContours'un üçüncü argümanı olan contour approximation:
# bu parametrede seçilen değere göre kontur yani şeklin sınırları farklı şekillerde tutulur.
# (Kontur aynı yoğunluktaki bir şeklin sınırlarıdır.)
# bir şekili tanımak için tüm sınırlarına ihtiyacımız yoktur.
# Örn dikdörtgen için düşünürsek sadece köşedeki dört noktayı bilsek yeter.
# bu köşeleri birleştiren her noktaya ihtiyacımız yoktur.
# bunun için CHAIN_APPROX_SIMPLE kullanabiliriz.
# CHAIN_APPROX_SIMPLE, tüm gereksiz noktaları kaldırarak konturu sıkıştırır.
# Böylece bellekten tasarruf sağlar.

"""drawContours"""
# drawContours argumanları:
# kaynak görüntü
# list şeklinde konturlar
# konturun indeksi ( Eğer hepsini çizeceksek -1 )
# renk
# kalınlık

# sadece belirli bir bölgeyi konturlamak için:
cv.drawContours(img, contours, 3, (0, 255, 0), 9)
cv.imshow('Contours', img)
cv.waitKey(0)

# veya bu kod da aynı işi yapar, daha kullanışlı
cnt = contours[9]
cv.drawContours(img, [cnt], 0, (0, 255, 0), 3)
cv.imshow('Contours', img)
cv.waitKey(0)

# tamamını konturlamak için
cv.drawContours(img, contours, -1, (0, 255, 0), 3)
cv.imshow('Contours', img)
cv.waitKey(0)
cv.imwrite("contoursWith_CHAIN_APPROX_NONE.png", img)
# hep orijinal img'i değiştiriyoruz bu yüzden img.copy() yapıp onun
# üzerinde çalışmak daha mantıklı.

