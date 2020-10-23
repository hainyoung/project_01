import numpy as np
import cv2
import pandas as pd
import sys

def mask_speed_digit_recognizer(image,x,y,h,w):
	    #     #######   training part    ############### 
	    
	    # BINARY MODEL
        # samples_speed = np.loadtxt('./speed_generalsamples.data',np.float32)
        # responses_speed = np.loadtxt('./speed_generalresponses.data',np.float32)

        # # # #OTSU MODEL
        # samples_speed = np.loadtxt('./models_OTSU/generalsamples.data',np.float32)
        # responses_speed = np.loadtxt('./models_OTSU/generalresponses.data',np.float32)
        
        # # #OTSU MODEL_newimage
        samples_speed = np.loadtxt('./speed_model/speed_generalsamples.data',np.float32)
        responses_speed = np.loadtxt('./speed_model/speed_generalresponses.data',np.float32)


        responses_speed = responses_speed.reshape((responses_speed.size,1))
        model = cv2.ml.KNearest_create()
        model.train(samples_speed, cv2.ml.ROW_SAMPLE, responses_speed)
        roi_speed = image[y:y+h, x:x+w]
        out_speed = np.zeros(roi_speed.shape,np.uint8)
        gray_speed = cv2.cvtColor(roi_speed,cv2.COLOR_BGR2GRAY)
        blur_speed = cv2.GaussianBlur(gray_speed,(3,3),0)
        #_, thresh_speed = cv2.threshold(blur_speed, 127,255, cv2.THRESH_BINARY)
        # _, thresh_speed = cv2.threshold(blur_speed, 0,255, cv2.THRESH_OTSU)

        lower_white = np.array([180])
        upper_white = np.array([255])
        mask_white = cv2.inRange(blur_speed, lower_white, upper_white)
        mask_white = cv2.bitwise_and(blur_speed, blur_speed, mask = mask_white)

        contours_speed,hierarchy_speed = cv2.findContours(mask_white,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        speed = []
        for cnt_speed in contours_speed:
            if cv2.contourArea(cnt_speed) > 10:           
                [x, y, w, h] = cv2.boundingRect(cnt_speed)
                if  h>28:
                    cv2.rectangle(roi_speed,(x,y),(x+w,y+h),(0,255,0),2)
                    roi_sp = mask_white[y:y+h,x:x+w]
                    roismall = cv2.resize(roi_sp,(10,10))
                    roismall = roismall.reshape((1,100))
                    roismall = np.float32(roismall)
                    # for each in roismall:
                    retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                    string_speed= str(int((results[0][0])))
                    # string_dt = str((results[0][0]))
                    cv2.putText(out_speed,string_speed,(x,y+h),0,1,(0,255,0))
                    # print(string_speed)
                    speed.append(string_speed)
        return speed