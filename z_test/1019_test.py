import pytesseract
import numpy as np
import cv2
import pandas as pd
import sys
from datetime import datetime


# VIDEOS FROM 1003 STEP 2&1


# cap = cv2.VideoCapture("./use_video.mp4")
# cap = cv2.VideoCapture("./night_mode.mp4")
# cap = cv2.VideoCapture("./night_light.mp4")
# cap = cv2.VideoCapture("./cut_video.mp4")
# cap = cv2.VideoCapture("./video_2.mp4")
# cap = cv2.VideoCapture("./1min.mp4")


#Videos from 1015  STEP 3

# cap = cv2.VideoCapture("./1015_video_list/dim_10_10frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/dim_10_20frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/outdoor_12_brightness_max_22frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/outdoor_12_saturation_max_22frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/indoor_10_22frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/standard_15_22frame.mp4")
cap = cv2.VideoCapture("./1015_video_list/outdoor_12_22frame.mp4")

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



c = 1

while(cap.isOpened()):

    ret, frame = cap.read()
    if not ret:
        break

    if c%2==0:
        
        # DATE and TIME 
        samples_dt = np.loadtxt('./z_test/date_time_generalsamples.data', np.float32)
        responses_dt = np.loadtxt('./z_test/date_time_generalresponses.data', np.float32)
        responses_dt = responses_dt.reshape((responses_dt.size, 1))
        model_dt = cv2.ml.KNearest_create()
        model_dt.train(samples_dt, cv2.ml.ROW_SAMPLE, responses_dt)
        
        roi_dt = frame[51:131, 28:873]
        out_dt = np.zeros(roi_dt.shape,np.uint8)
        gray_dt = cv2.cvtColor(roi_dt,cv2.COLOR_BGR2GRAY)
        blur_dt = cv2.GaussianBlur(gray_dt,(5,5),0)
        _, thresh_dt = cv2.threshold(blur_dt, 127,255, cv2.THRESH_OTSU)
        
        contours_dt,hierarchy_dt = cv2.findContours(thresh_dt,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        dt = []

        sys.stdout = open('./outdoor_12_22frame_output.txt', 'a', -1, 'utf-8')

        for cnt_dt in contours_dt:

            if cv2.contourArea(cnt_dt)>300:           
                x, y, w, h = 28, 51, 873-28, 131-51
                [x, y, w, h] = cv2.boundingRect(cnt_dt)
                if  h>32:
                    cv2.rectangle(roi_dt,(x,y),(x+w,y+h),(0,255,0),2)
                    roi_dt_ = thresh_dt[y:y+h,x:x+w]
                    roismall = cv2.resize(roi_dt_,(10,10))
                    roismall = roismall.reshape((1,100))
                    roismall = np.float32(roismall)
                    # for each in roismall:
                    retval, results, neigh_resp, dists = model_dt.findNearest(roismall, k = 1)
                    string_dt= str(int((results[0][0])))
                    # string_dt = str((results[0][0]))
                    cv2.putText(out_dt,string_dt,(x,y+h),0,1,(0,255,0))
                    # print(string_dt)
                    dt.append(string_dt)
        # print(dt)
        date_time = "".join(dt)
        date_time = date_time[::-1]
        date = date_time[:8]
        time = date_time[8:]

        print(date[:4]+"-"+date[4:6]+"-"+date[6:], end = ',', sep = ',')
        print(time[:2]+":"+time[2:4]+":"+time[4:], end = ',', sep = ',')


        # SPEED
        #         #######   training part    ############### 
        samples_speed = np.loadtxt('./z_test/speed_generalsamples.data',np.float32)
        responses_speed = np.loadtxt('./z_test/speed_generalresponses.data',np.float32)
        responses_speed = responses_speed.reshape((responses_speed.size,1))

        model_speed = cv2.ml.KNearest_create()
        model_speed.train(samples_speed, cv2.ml.ROW_SAMPLE, responses_speed)

        # ############################# testing part  #########################
        # roi_speed =frame[569:617, 1099:1128]
        y_2,h_2,x_2,w_2 = 568, 57, 1028, 97
        # y_2,h_2,x_2,w_2 = 569,48,1099,29
        # y_2,h_2,x_2,w_2 = 569, 47, 1047, 76
        roi_speed =frame[y_2:y_2+h_2, x_2:x_2+w_2]

        out_speed = np.zeros(roi_speed.shape,np.uint8)
        gray_speed = cv2.cvtColor(roi_speed,cv2.COLOR_BGR2GRAY)
        blur_speed = cv2.GaussianBlur(gray_speed,(5,5),0)
        _, thresh_speed = cv2.threshold(blur_speed, 127,255, cv2.THRESH_BINARY)
        
        contours_speed,hierarchy_speed = cv2.findContours(thresh_speed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        speed = []
        for cnt_speed in contours_speed:
            if cv2.contourArea(cnt_speed)>25:           
                x, y, w, h = x_2, y_2, w_2, h_2
                [x, y, w, h] = cv2.boundingRect(cnt_speed)
                if  h>28:
                    cv2.rectangle(roi_speed,(x,y),(x+w,y+h),(0,255,0),2)
                    roi_speed = thresh_speed[y:y+h,x:x+w]
                    roismall = cv2.resize(roi_speed,(10,10))
                    roismall = roismall.reshape((1,100))
                    roismall = np.float32(roismall)
                    # for each in roismall:
                    retval, results, neigh_resp, dists = model_speed.findNearest(roismall, k = 1)
                    string_speed= str(int((results[0][0])))
                    # string_dt = str((results[0][0]))
                    cv2.putText(out_speed,string_speed,(x,y+h),0,1,(0,255,0))
                    # print(string_speed)
                    speed.append(string_speed)
        speed = "".join(speed)
        speed = speed[::-1]
        print(speed, end = ',', sep = ',')


        # # FOURTH ROI : LANE
        y_3,h_3,x_3,w_3 = 450,43,933,52
        roi_4 = frame[y_3:y_3+h_3,x_3:x_3+w_3]
        hsv_green_lane = hsv_green(roi_4)
        hsv_white_lane = hsv_white(roi_4)
        rec_3 = draw_rec(frame,"Lane_mode", x_3,y_3,h_3,w_3)


        hsv_green_lane_per_row = np.average(hsv_green_lane, axis = 0)
        avg_green_lane_ = np.average(hsv_green_lane_per_row, axis = 0)
        avg_green_lane = np.average(avg_green_lane_)

        hsv_white_lane_per_row = np.average(hsv_white_lane, axis = 0)
        avg_white_lane_ = np.average(hsv_white_lane_per_row, axis = 0)
        avg_white_lane = np.average(avg_white_lane_)
        # print(avg_white_lane)
        # print(avg_green_lane)

        # # FIFTH ROI : AUTO
        y_4,h_4,x_4,w_4 = 459,56,1217,51
        roi_5 = frame[y_4:y_4+h_4,x_4:x_4+w_4]
        hsv_green_auto = hsv_green(roi_5)
        hsv_white_auto = hsv_white(roi_5)
        rec_4 = draw_rec(frame,"Auto_mode", x_4,y_4,h_4,w_4)



        hsv_green_auto_per_row = np.average(hsv_green_auto, axis = 0)
        avg_green_auto_ = np.average(hsv_green_auto_per_row, axis = 0)
        avg_green_auto = np.average(avg_green_auto_)

        hsv_white_auto_per_row = np.average(hsv_white_auto, axis = 0)
        avg_white_auto_ = np.average(hsv_white_auto_per_row, axis = 0)
        avg_white_auto = np.average(avg_white_auto_)

        # print("white average")
        # print(avg_white_auto)
        # print("green average")
        # print(avg_green_auto)

        #Show all the windows




        cv2.imshow("Frame", frame)
        cv2.imshow('roi_dt', roi_dt)
        cv2.imshow('out_dt', out_dt)
        cv2.imshow('roi_speed', roi_speed)
        cv2.imshow('out_speed', out_speed)
        cv2.imshow("hsv_green_lane", hsv_white_lane)
        cv2.imshow("hsv_white_lane", hsv_green_lane)
        cv2.imshow("hsv_green_auto", hsv_white_auto)
        cv2.imshow("hsv_white_auto", hsv_green_auto)
        
        cv2.moveWindow("Frame", 100, 0)
        cv2.moveWindow('roi_dt', 0, 100)
        cv2.moveWindow('out_dt', 100, 100)
        cv2.moveWindow('roi_speed', 0, 200)
        cv2.moveWindow('out_speed', 100, 200)
        cv2.moveWindow('hsv_green_lane', 0,700)
        cv2.moveWindow('hsv_white_lane', 150,700)
        cv2.moveWindow('hsv_green_auto', 0,900)
        cv2.moveWindow('hsv_white_auto', 150,900)
        

        # print the results
        # print(text_date_time, text_3, end = ',', sep = ',')
        
        # lane
        # print("average : lane")
        if avg_green_lane > 0 :
            print("1", end = ',', sep = ',')
        else : 
            print("0", end = ',', sep = ',')

        # auto
        # print("average : auto")
        if avg_green_auto > 0 or avg_white_auto > 0:
            print("1")
        else :
            print("0")

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    c+=1


cap.release()
# out.release()
cv2.destroyAllWindows()