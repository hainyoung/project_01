import sys

import numpy as np
import cv2

im = cv2.imread('./speed_model/1022_speed.png')
#im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(3,3),0)
# thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

lower_white = np.array([180])
upper_white = np.array([255])
mask_white = cv2.inRange(blur,lower_white, upper_white)
mask_white = cv2.bitwise_and(blur, blur, mask = mask_white)

#_, thresh = cv2.threshold(mask_green,135,255, cv2.THRESH_OTSU)

contours,hierarchy = cv2.findContours(mask_white,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow("mask", mask_white)
# cv2.imshow("blur", blur)
samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

for cnt in contours:
    if cv2.contourArea(cnt)>10:
        [x,y,w,h] = cv2.boundingRect(cnt)

        if  h>25:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = mask_white[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print("training complete")

np.savetxt('./mask_speed_generalsamples.data',samples)
np.savetxt('./mask_speed_generalresponses.data',responses)
