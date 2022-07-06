import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# watershed algoritması su havzası tarzı bir yaklaşıma dayanıyor.
# çukurlarda su old düşünelim. Bir çukurdaki diğeriyle karışmasın diye bariyerler
# koymamız gerekiyor. Bunu  img proc açısından düşünürsek,
# birbirine değen şekillerimiz var, bunların sınırlarını belirliyoruz.
# bunu yaparken nesnelerimize etiketler vermemiz gerkeiyor.
# ön planda ya da nesne olduğundan emin olduğumuz bölgelere bir renk
# arka planda veya nesne olmadığından emin olduğumuz bölgelere başka bir renk vereceğiz.
# hakkında hiçbir şeyden emin olmadığımız bölgeleri ise 0 ile etiketleyeceğiz.
# aardından watershed alg'ını uygulayacağız.
# markerımız, verdiğimiz etiketlerle güncellenecek ne nesnelerin sınırları
# -1 olacaktır.


img = cv.imread("../input_pictures/input_money4.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
cv.imwrite("th1.png", th)

# beyaz gürültüyü kaldırmak için opening uyguladık:
# (nesnedeki delikleri kaldırmak için closing uygulayabilirdik)
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(th, cv.MORPH_OPEN, kernel, iterations=2)

# nesnenin merkezine yakın olan gölgeler = ön plan
# burada emin olmadığımız tek bölge madeni paraların sınırları.
# bu yüzden madeni para olduğuna emin olduğumuz alanı çıkaracağız.
# Erozyon ile sınır piksellerini kaldırırsak,
# geriye kalan alanın madeni para old. emin oluruz.


# dilation yaparsak geriye kalan alanın kesinlikle arka plan old. biliriz:
sure_bg = cv.dilate(opening, kernel, iterations=3)

# Nesneler birbirine dokunduğu için onları ayırmak için distanceTransform uyguluyoruz?
# Eğer bunu yapmaya çalışmak yerine sadece ön plan alanını çıkarmak istiyor olsaydık,
# sadece erosion uygulamamız yeterli olabilirdi.
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
# Geriye kalan bölgeler, hiçbir fikrimizin olmadığı bölgeler olacaktır(sınır):
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

cv.imwrite("distanceTransform.png", dist_transform)
cv.imwrite("threshold.png", sure_fg)
# Artık hangi bölgenin ne olduğunu bildiğimiz için,
# marker ile bölgeleri etiketliyoruz:
# Ön plan ve arka planı kesin olarak bildiğimiz için
# onları farklı pozitif tamsayılar ile etiketleyeceğiz.
# kesin olarak bilmediğimiz(sınır) alanları da 0 ile etiketleyeceğiz.
# bunun için cv.connectedComponents()'i kullanacağız. Bu metod,
# görüntünün arka planını 0 ile etiketler, ardından diğer nesnelere 1'den
# başlayarak tam sayılar vererek etiketler.

ret, markers = cv.connectedComponents(sure_fg)
# Ancak biz Watershed metodunu kullanacağımız için sadece bilinmeyen bölgelerin
# 0 değeri ile etiketlenmesini istiyoruz.
markers = markers + 1
# bilinmeyen bölgelere 0 verelim:
markers[unknown == 255] = 0
cv.imwrite("markers.png", markers)
# Watershed'i uygulayalım ve sınırlara -1 değerini verelim:
markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]
cv.imwrite("markers.png", markers)
cv.imwrite("img.png", img)
plt.imshow(markers), plt.colorbar(), plt.show()