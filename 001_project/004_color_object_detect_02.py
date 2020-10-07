import numpy as np
import cv2

cap = cv2.VideoCapture("./001_project/video.mp4")

while True:
    ret, frame = cap.read()

# convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# set a range
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([70, 255, 255])

# apply threshold
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    res = cv2.bitwise_and(frame, frame, mask=mask_green)

    cv2.imshow('original', frame)
    cv2.imshow('Green', res)

    if cv2.waitKey(1) > 0 :
        break

cv2.destroyAllWindows()


