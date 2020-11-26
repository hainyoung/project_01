import numpy as np
import cv2
import pandas as pd
import sys


#Functions
def converter(roi):
    # roi_h = roi.shape[0]
    # roi_w = roi.shape[1]
    # resize_roi = cv2.resize(roi,(int(roi_w*4),int(roi_h*4)))  

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # threshold the image using Otsus method to preprocess for tesseract
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # perform a median blur to smooth image slightly
    blur = cv2.medianBlur(thresh, 3)
    # resize image to double the original size as tesseract does better with certain text size
    blur = cv2.resize(blur, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
    # run tesseract and convert image text to string
    return blur

def hsv_green(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 30, 90])
    upper_green = np.array([93, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_green = cv2.bitwise_and(roi, roi, mask = mask_green)
    return mask_green

def hsv_white(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,120])
    upper_white = np.array([30,255,255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_white = cv2.bitwise_and(roi, roi, mask = mask_white)
    return mask_white 

def draw_rec(frame,name,x,y,h,w):
    rec = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    rec = cv2.putText(rec, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 1)
    return rec


def date_time_recognizer(image,x,y,h,w):
        samples_dt = np.loadtxt('./date_time_model/date_time_generalsamples.data', np.float32)
        responses_dt = np.loadtxt('./date_time_model/date_time_generalresponses.data', np.float32)
        responses_dt = responses_dt.reshape((responses_dt.size, 1))
        model_dt = cv2.ml.KNearest_create()
        model_dt.train(samples_dt, cv2.ml.ROW_SAMPLE, responses_dt)
        roi_dt = image[y:y+h, x:x+w]
        out_dt = np.zeros(roi_dt.shape,np.uint8)
        gray_dt = cv2.cvtColor(roi_dt,cv2.COLOR_BGR2GRAY)
        blur_dt = cv2.GaussianBlur(gray_dt,(5,5),0)
        _, thresh_dt = cv2.threshold(blur_dt, 127,255, cv2.THRESH_OTSU)
        contours_dt,hierarchy_dt = cv2.findContours(thresh_dt,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        dt = []
        for cnt_dt in contours_dt:
            if cv2.contourArea(cnt_dt)>300:
                #x, y, w, h = 28, 51, 873-28, 131-51
                [x, y, w, h] = cv2.boundingRect(cnt_dt)
                if  h>32:
                    cv2.rectangle(roi_dt,(x,y),(x+w,y+h),(0,255,0),2)
                    roi_date = thresh_dt[y:y+h,x:x+w]
                    roismall = cv2.resize(roi_date,(10,10))
                    roismall = roismall.reshape((1,100))
                    roismall = np.float32(roismall)
                    # for each in roismall:
                    retval, results, neigh_resp, dists = model_dt.findNearest(roismall, k = 1)
                    string_dt= str(int((results[0][0])))
                    # string_dt = str((results[0][0]))
                    cv2.putText(out_dt,string_dt,(x,y+h),0,1,(0,255,0))
                    # print(string_dt)
                    dt.append(string_dt)
        return dt                     



def speed_digit_recognizer(image,x,y,h,w):
	    #     #######   training part    ############### 

	    # BINARY MODEL
        # samples_speed = np.loadtxt('./speed_generalsamples.data',np.float32)
        # responses_speed = np.loadtxt('./speed_generalresponses.data',np.float32)

        # # # #OTSU MODEL
        # samples_speed = np.loadtxt('./models_OTSU/generalsamples.data',np.float32)
        # responses_speed = np.loadtxt('./models_OTSU/generalresponses.data',np.float32)
        
        # # #OTSU MODEL_newimage
        samples_speed = np.loadtxt('./problem_solved/models_OTSU_newimage/generalsamples.data',np.float32)
        responses_speed = np.loadtxt('./problem_solved/models_OTSU_newimage/generalresponses.data',np.float32)

        # samples_speed = np.loadtxt('./problem_solved/19_model/speed_generalsamples.data',np.float32)
        # responses_speed = np.loadtxt('./problem_solved/19_model/speed_generalresponses.data',np.float32)


        responses_speed = responses_speed.reshape((responses_speed.size,1))
        model = cv2.ml.KNearest_create()
        model.train(samples_speed, cv2.ml.ROW_SAMPLE, responses_speed)
        roi_speed = image[y:y+h, x:x+w]
        out_speed = np.zeros(roi_speed.shape,np.uint8)
        gray_speed = cv2.cvtColor(roi_speed,cv2.COLOR_BGR2GRAY)
        blur_speed = cv2.GaussianBlur(gray_speed,(5,5),0)
        #_, thresh_speed = cv2.threshold(blur_speed, 127,255, cv2.THRESH_BINARY)
        _, thresh_speed = cv2.threshold(blur_speed, 0,255, cv2.THRESH_OTSU)

        contours_speed,hierarchy_speed = cv2.findContours(thresh_speed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        speed = []
        for cnt_speed in contours_speed:
            if cv2.contourArea(cnt_speed)<800:           
                [x, y, w, h] = cv2.boundingRect(cnt_speed)
                if  h>28:
                    cv2.rectangle(roi_speed,(x,y),(x+w,y+h),(0,255,0),2)
                    roi_sp = thresh_speed[y:y+h,x:x+w]
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


def mask_speed_digit_recognizer(image,x,y,h,w):
	    #     #######   training part    ############### 
	    
	    # BINARY MODEL
        # samples_speed = np.loadtxt('./speed_generalsamples.data',np.float32)
        # responses_speed = np.loadtxt('./speed_generalresponses.data',np.float32)

        # # # #OTSU MODEL
        # samples_speed = np.loadtxt('./models_OTSU/generalsamples.data',np.float32)
        # responses_speed = np.loadtxt('./models_OTSU/generalresponses.data',np.float32)
        
        # # #OTSU MODEL_newimage
        # samples_speed = np.loadtxt('./speed_model/speed_generalsamples.data',np.float32)
        # responses_speed = np.loadtxt('./speed_model/speed_generalresponses.data',np.float32)

        samples_speed = np.loadtxt('./mask_speed_generalsamples.data',np.float32)
        responses_speed = np.loadtxt('./mask_speed_generalresponses.data',np.float32)

        responses_speed = responses_speed.reshape((responses_speed.size,1))
        model = cv2.ml.KNearest_create()
        model.train(samples_speed, cv2.ml.ROW_SAMPLE, responses_speed)
        roi_speed = image[y:y+h, x:x+w]
        out_speed = np.zeros(roi_speed.shape,np.uint8)
        gray_speed = cv2.cvtColor(roi_speed,cv2.COLOR_BGR2GRAY)
        blur_speed = cv2.GaussianBlur(gray_speed,(3, 3),0)
        #_, thresh_speed = cv2.threshold(blur_speed, 127,255, cv2.THRESH_BINARY)
        # _, thresh_speed = cv2.threshold(blur_speed, 0,255, cv2.THRESH_OTSU)

        lower_white = np.array([136])
        upper_white = np.array([255])
        mask_white = cv2.inRange(blur_speed, lower_white, upper_white)
        mask_white = cv2.bitwise_and(blur_speed, blur_speed, mask = mask_white)

        contours_speed,hierarchy_speed = cv2.findContours(mask_white,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        speed = []
        for cnt_speed in contours_speed:
            if cv2.contourArea(cnt_speed) < 800:           
                [x, y, w, h] = cv2.boundingRect(cnt_speed)
                if  h>25:
                    cv2.rectangle(roi_speed,(x,y),(x+w,y+h),(0,255,0),2)
                    roi_sp = mask_white[y:y+h,x:x+w]
                    roismall_resize = cv2.resize(roi_sp,(10,10))
                    roismall_reshape = roismall_resize.reshape((1,100))
                    roismall = np.float32(roismall_reshape)
                    # for each in roismall:
                    retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                    string_speed= str(int((results[0][0])))
                    # string_dt = str((results[0][0]))
                    cv2.putText(out_speed,string_speed,(x,y+h),0,1,(0,255,0))
                    # print(string_speed)
                    speed.append(string_speed)
        return speed



                
 
