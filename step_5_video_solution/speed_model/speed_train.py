import sys

import numpy as np
import cv2

im = cv2.imread('final_resized_speed_img.png')
#im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray,(5,5),0)
# thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

# lower_green = np.array([100])
# upper_green = np.array([200])
# mask_green = cv2.inRange(blur,lower_green, upper_green)
# mask_green = cv2.bitwise_and(blur, blur, mask = mask_green)

_, thresh = cv2.threshold(gray,135,255, cv2.THRESH_OTSU)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# cv2.imshow("blur", blur)
cv2.imshow("thresh", thresh)
#cv2.imshow("mask", mask_green)

samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)

        if  h>28:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
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

np.savetxt('44_generalsamples.data',samples)
np.savetxt('44_generalresponses.data',responses)