import pytesseract
import numpy as np
import cv2
import pandas as pd
import sys


# VIDEOS FROM 1003 STEP 2&1


# cap = cv2.VideoCapture("./use_video.mp4")
# cap = cv2.VideoCapture("./night_mode.mp4")
# cap = cv2.VideoCapture("./night_light.mp4")
# cap = cv2.VideoCapture("./cut_video.mp4")
# cap = cv2.VideoCapture("./video_2.mp4")
# cap = cv2.VideoCapture("./1min.mp4")


#Videos from 1015  STEP 3

cap = cv2.VideoCapture("./dim_10_10frame.mp4")

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
    
    if c%23==0:

        # # FIRST ROI : DATE
        # roi_1 = frame[38:140, 25:490]
        # blur_1 = converter(roi_1)
        # text_1 = pytesseract.image_to_string(blur_1, config='-c tessedit_char_whitelist=-:0123456789 --psm 13 --oem 3')
        # text_1 = text_1.strip()
        # # SECOND ROI : TIME
        # roi_2 =frame[38:144,513:890]
        # blur_2 = converter(roi_2)
        # text_2 = pytesseract.image_to_string(blur_2, config='-c tessedit_char_whitelist=:0123456789 --psm 13 --oem 3')
        # text_2 = text_2.strip()
        

        # # DATE&TIME COMBINED ROI 
        # y_1,h_1,x_1,w_1 = 51,80,28,845
        # roi_date_time =frame[y_1:y_1+h_1,x_1:x_1+w_1]
        # date_time = converter(roi_date_time)
        # text_date_time = pytesseract.image_to_string(date_time, config='-c tessedit_char_whitelist=-:0123456789 --psm 13 --oem 1')
        # text_date_time = text_date_time.strip()
        # rec_1 = draw_rec(frame,"Current Time", x_1,y_1,h_1,w_1)
        # # rec_3 = cv2.rectangle(frame, (966, 643), (1031, 712), (0,255,0), 2)
        # rec_3 = cv2.putText(rec_3, 'Speed', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 1) 
        



        # THIRD ROI : SPEED
                #######   training part    ############### 
        samples = np.loadtxt('generalsamples.data',np.float32)
        responses = np.loadtxt('generalresponses.data',np.float32)
        responses = responses.reshape((responses.size,1))

        model = cv2.ml.KNearest_create()
        model.train(samples, cv2.ml.ROW_SAMPLE, responses)

        ############################# testing part  #########################


        y_2,h_2,x_2,w_2 = 568,57,1028,97
        roi_3 =frame[y_2:y_2+h_2,x_2:x_2+w_2]
        #blur_3= converter(roi_3)
        #text_3 = pytesseract.image_to_string(blur_3, config='-c tessedit_char_whitelist=0123456789 --psm 11 --oem 3')
        #text_3 = text_3.strip()
        rec_2 = draw_rec(frame,"Speed", x_2,y_2,h_2,w_2)
        out = np.zeros(roi_3.shape,np.uint8)
        gray = cv2.cvtColor(roi_3,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        # thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

        _, thresh = cv2.threshold(blur, 127,255, cv2.THRESH_BINARY)

        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt)>50:
                [x,y,w,h] = cv2.boundingRect(cnt)
                if  h>28:
                    cv2.rectangle(roi_3,(x,y),(x+w,y+h),(0,255,0),2)
                    roi = thresh[y:y+h,x:x+w]
                    roismall = cv2.resize(roi,(10,10))
                    roismall = roismall.reshape((1,100))
                    roismall = np.float32(roismall)
                    for each in roismall:
                        retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                        string = str(int((results[0][0])))
                        cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
                        print(string)

        cv2.imshow('roi_3',roi_3)
        cv2.imshow('out',out)
        # # # FOURTH ROI : LANE
        # y_3,h_3,x_3,w_3 = 450,43,933,52
        # roi_4 = frame[y_3:y_3+h_3,x_3:x_3+w_3]
        # hsv_green_lane = hsv_green(roi_4)
        # hsv_white_lane = hsv_white(roi_4)
        # rec_3 = draw_rec(frame,"Lane_mode", x_3,y_3,h_3,w_3)


        # hsv_green_lane_per_row = np.average(hsv_green_lane, axis = 0)
        # avg_green_lane_ = np.average(hsv_green_lane_per_row, axis = 0)
        # avg_green_lane = np.average(avg_green_lane_)

        # hsv_white_lane_per_row = np.average(hsv_white_lane, axis = 0)
        # avg_white_lane_ = np.average(hsv_white_lane_per_row, axis = 0)
        # avg_white_lane = np.average(avg_white_lane_)
        # # print(avg_white_lane)
        # # print(avg_green_lane)

        # # # FIFTH ROI : AUTO
        # y_4,h_4,x_4,w_4 = 459,56,1217,51
        # roi_5 = frame[y_4:y_4+h_4,x_4:x_4+w_4]
        # hsv_green_auto = hsv_green(roi_5)
        # hsv_white_auto = hsv_white(roi_5)
        # rec_4 = draw_rec(frame,"Auto_mode", x_4,y_4,h_4,w_4)



        # hsv_green_auto_per_row = np.average(hsv_green_auto, axis = 0)
        # avg_green_auto_ = np.average(hsv_green_auto_per_row, axis = 0)
        # avg_green_auto = np.average(avg_green_auto_)

        # hsv_white_auto_per_row = np.average(hsv_white_auto, axis = 0)
        # avg_white_auto_ = np.average(hsv_white_auto_per_row, axis = 0)
        # avg_white_auto = np.average(avg_white_auto_)

        # # print("white average")
        # # print(avg_white_auto)
        # # print("green average")
        # # print(avg_green_auto)

        #Show all the windows
        #cv2.imshow("Frame", frame)
        # cv2.imshow("blur_1",blur_1)
        # cv2.imshow("blur_2",blur_2)
        # cv2.imshow("Date_time",date_time)
        #cv2.imshow("Speed",blur)
        # cv2.imshow("hsv_green_lane", hsv_white_lane)
        # cv2.imshow("hsv_white_lane", hsv_green_lane)
        # cv2.imshow("hsv_green_auto", hsv_white_auto)
        # cv2.imshow("hsv_white_auto", hsv_green_auto)
        
        #cv2.moveWindow("Frame", 0,0)
        # cv2.moveWindow('blur_1', 0,0)
        # cv2.moveWindow('blur_2', 0,250)
        # cv2.moveWindow('Date_time', 0,230)
        #cv2.moveWindow('Speed', 0,450)        
        # cv2.moveWindow('hsv_green_lane', 0,700)
        # cv2.moveWindow('hsv_white_lane', 150,700)
        # cv2.moveWindow('hsv_green_auto', 0,900)
        # cv2.moveWindow('hsv_white_auto', 150,900)
        

        # sys.stdout = open('output.txt', 'a', -1, 'utf-8')
        # print the results
        # print(text_date_time, text_3, end = ',', sep = ',')
        
        # # lane
        # # print("average : lane")
        # if avg_green_lane > 0 :
        #     print("1", end = ',', sep = ',')
        # else : 
        #     print("0", end = ',', sep = ',')

        # # auto
        # # print("average : auto")
        # if avg_green_auto > 0 or avg_white_auto > 0:
        #     print("1")
        # else :
        #     print("0")

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    c+=1


cap.release()
# out.release()
cv2.destroyAllWindows()