import cv2 as cv

cap = cv.VideoCapture(0)  # 1'de ikinci kamerayı açar
if not cap.isOpened():
    print("Cant open the camera")
    exit()
while True:
    ret, frame = cap.read()  # frameleri döner
    if not ret:
        print("Cant receive frame")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey() == ord('s'):
        break
cap.release()
cv.destroyAllWindows()
