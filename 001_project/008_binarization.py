import sys
import cv2
import numpy as np

cap = cv2.VideoCapture('./001_project/data/cut_video.avi')

while True:
    ret, frame = cap.read()
    newframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img_binary = cv2.adaptiveThreshold(newframe, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 29, 20)

    cv2.imshow('newframe', img_binary)
    if cv2.waitKey(1) > 0 : 
        break

cap.release()
cv2.destroyAllWindows()



