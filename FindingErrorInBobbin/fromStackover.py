import cv2
import numpy as np
import sys
# kendi kodum değil deneme amaçlı burada bıraktım, yanlış çalışıyor

# initialize global H, S, V values
min_global_h = 179
min_global_s = 255
min_global_v = 255

max_global_h = 0
max_global_s = 0
max_global_v = 0

# load input image from the cmd-line
filename = "../input_bobins/with/1.png"
img = cv2.imread("../input_bobins/with/1.png")
if (img is None):
    print('!!! Failed imread')
    sys.exit(-1)

# create an auxiliary image for debugging purposes
dbg_img = img.copy()

# initiailize a list of Regions of Interest that need to be scanned to identify good HSV values to threhsold by color
w = img.shape[1]
h = img.shape[0]
roi_w = int(w * 0.10)
roi_h = int(h * 0.10)
roi_list = []
roi_list.append( (int(w*0.25), int(h*0.15), roi_w, roi_h) )
roi_list.append( (int(w*0.25), int(h*0.60), roi_w, roi_h) )

# convert image to HSV color space
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# iterate through the ROIs to determine the min/max HSV color of the reel
for rect in roi_list:
    x, y, w, h = rect
    x2 = x + w
    y2 = y + h
    print('ROI rect=', rect)

    cropped_hsv_img = hsv_img[y:y+h, x:x+w]

    h, s, v = cv2.split(cropped_hsv_img)
    min_h  = np.min(h)
    min_s  = np.min(s)
    min_v  = np.min(v)

    if (min_h < min_global_h):
        min_global_h = min_h

    if (min_s < min_global_s):
        min_global_s = min_s

    if (min_v < min_global_v):
        min_global_v = min_v

    max_h  = np.max(h)
    max_s  = np.max(s)
    max_v  = np.max(v)

    if (max_h > max_global_h):
        max_global_h = max_h

    if (max_s > max_global_s):
        max_global_s = max_s

    if (max_v > max_global_v):
        max_global_v = max_v

    # debug: draw ROI in original image
    cv2.rectangle(dbg_img, (x, y), (x2, y2), (255,165,0), 4) # red


cv2.imshow('ROIs', cv2.resize(dbg_img, dsize=(0, 0), fx=0.5, fy=0.5))
#cv2.waitKey(0)
cv2.imwrite(filename[:-4] + '_rois.png', dbg_img)

# define min/max color for threshold
low_hsv = np.array([min_h, min_s, min_v])
max_hsv = np.array([max_h, max_s, max_v])
#print('low_hsv=', low_hsv)
#print('max_hsv=', max_hsv)

# threshold image by color
img_bin = cv2.inRange(hsv_img, low_hsv, max_hsv)
cv2.imshow('binary', cv2.resize(img_bin, dsize=(0, 0), fx=0.5, fy=0.5))
cv2.imwrite(filename[:-4] + '_binary.png', img_bin)

#cv2.imshow('img_bin', cv2.resize(img_bin, dsize=(0, 0), fx=0.5, fy=0.5))
#cv2.waitKey(0)

# create a mask to store the contour of the reel (hopefully)
mask = np.zeros((img_bin.shape[0], img_bin.shape[1]), np.uint8)
crop_x, crop_y, crop_w, crop_h = (0, 0, 0, 0)

# iterate throw all the contours in the binary image:
#   assume that the first contour with an area larger than 100k belongs to the reel
contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for contourIdx, cnt in enumerate(contours):
    area = cv2.contourArea(contours[contourIdx])
    print('contourIdx=', contourIdx, 'area=', area)

    # draw potential reel blob on the mask (in white)
    if (area > 100000):
        crop_x, crop_y, crop_w, crop_h = cv2.boundingRect(cnt)
        centers, radius = cv2.minEnclosingCircle(cnt)

        cv2.circle(mask, (int(centers[0]), int(centers[1])), int(radius), (255), -1) # fill with white
        break

cv2.imshow('mask', cv2.resize(mask, dsize=(0, 0), fx=0.5, fy=0.5))
cv2.imwrite(filename[:-4] + '_mask.png', mask)

# copy just the reel area into its own image
reel_img = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('reel_img', cv2.resize(reel_img, dsize=(0, 0), fx=0.5, fy=0.5))
cv2.imwrite(filename[:-4] + '_reel.png', reel_img)

# crop the reel to a smaller image
if (crop_w != 0 and crop_h != 0):
    cropped_reel_img = reel_img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    cv2.imshow('cropped_reel_img', cv2.resize(cropped_reel_img, dsize=(0, 0), fx=0.5, fy=0.5))

    output_filename = filename[:-4] + '_crop.png'
    cv2.imwrite(output_filename, cropped_reel_img)

cv2.waitKey(0)