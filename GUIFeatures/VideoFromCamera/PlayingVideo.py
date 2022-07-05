import cv2 as cv

cap = cv.VideoCapture('output.avi')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive the frame")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()