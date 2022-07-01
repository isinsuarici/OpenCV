import cv2 as cv
import numpy as np

# circleları bulmak için kullanılır.
# çemberin formülü (x-x0)^2 + (y-y0)^2 = r^2 şeklinde olduğu için 3 parametreye ihtiyacımız olacak.

# ÖNEMLİ!!! =>
# Genelde dairelerin merkezini iyi algılamasına rağmen, yarıçapı doğru bulamayabiliyor.
# Bu yüzden, eğer biliyorsak yarıçap aralığını minRadius ve maxRadius ile belirtmeliyiz!
# Yarıçap aratmadan sadece merkezleri döndürmek için maxRadius'u negatif yap.


img = cv.imread("../input_pictures/input_money.png")  # read yaparken parama 0 verip sonra blurlarsam false
# negativeler oldu.
img = cv.GaussianBlur(img, (7, 7), 1.5)  # Bu kısmı idedeki dökümanda özellikle belirtmiş GaussianBlur() with 7x7
# kernel and 1.5 sigma !!!!! False negatifi baya azalttı.
img = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1.3, 30, param1=150, param2=80, minRadius=0, maxRadius=0)
# ide dökümanında param1 için 300 civarı iyi diyor. 60 gibi düşük değerler verdiğimde bazılarını algılayamamıştı.
# param2 için küçük değerler verince false positive çok artıyor.

# HoughCircles params=>
# img
# method => HOUGH_STANDARD, HOUGH_PROBABILISTIC, HOUGH_MULTI_SCALE, HOUGH_GRADIENT
# ama gradient dışındakileri deneyince hata veriyor. Diğerleri desteklenmiyormuş.
# dp = akümülatör çözünürlüğünün görüntü çözünürlüğüne ters oranı.
# dp=1 ise akümülatör giriş görüntüsüyle aynı.
# dp=2 is akümülatör genişliği ve yüksekliği yarısı kadar.
# minDist => tespit edilen dairelerin merkezler arasındaki min mesafe.
# minDist eğer çok küçükse false pozitifler oluyor. Çok büyükse de false negatifler oluyor.

circles = np.uint16(np.around(circles))  # circleları yuvarlayıp 16 bitte sakla

for c in circles[0, :]:
    cv.circle(img, (c[0], c[1]), c[2], (0, 255, 0), 3)
    cv.circle(img, (c[0], c[1]), 1, (0, 0, 255), 5)  # 3.parametre merkeze çizeceğimiz noktanın yarıçapı

cv.imshow("img", img)
cv.waitKey(0)
