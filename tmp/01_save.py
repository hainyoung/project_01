# import sys
# import numpy as np
# import cv2

# image = cv2.imread('./213-tile.jpg')
# img = image.copy()

# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# blur = cv2.GaussianBlur(gray,(5,5),0)
# thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

# contours, hierachy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# samples = np.empty((0, 100))
# responses = []
# keys = [i for i in range(48, 58)]

# for cnt in contours:
#     if cv2.contourArea(cnt) > 50 :
#         [x,y,w,h] = cv2.boundingRect(cnt)

#         if h>28:
#             cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
#             roi = thresh[y:y+h, x:x+w]
#             roismall = cv2.resize(roi, (10,10))
#             cv2.imshow('norm', img)
#             key = cv2.waitKey(0)

#             if key == 27:
#                 sys.exit()
#             elif key in keys:
#                 responses.append(int(chr(key)))
#                 sample = roismall.reshape(1,100)
#                 smaples = np.append(samples, sample, 0)

# responses = np.array(responses, np.float32)
# responses = responses.reshape((responses.size, 1))
# print("training complete")

# np.savetxt('./generalsamples.data', samples)
# np.savetxt('./generalresponses.data', responses)

# # samples = np.loadtxt('./generalsamples.data', np.float32)
# # responses = np.loadtxt('./generalresponses.data', np.float32)
# # responses = responses.reshape((responses.size, 1))

# # model = cv2.ml.KNearest_create()
# # model.train(samples, responses)


import sys

import numpy as np
import cv2

im = cv2.imread('./date_time_imgs.png')
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

#################      Now finding Contours         ###################

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

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

np.savetxt('./etc/date_time_generalsamples.data',samples)
np.savetxt('./etc/date_time_generalresponses.data',responses)