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

# cap = cv2.VideoCapture("./1015_video_list/dim_10_10frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/dim_10_20frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/outdoor_12_brightness_max_22frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/outdoor_12_saturation_max_22frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/indoor_10_22frame.mp4")
cap = cv2.VideoCapture("./1015_video_list/standard_15_22frame.mp4")

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

    if c%22==0:

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
        

        # DATE and TIME 
        samples_dt = np.loadtxt('./date_time_generalsamples.data', np.float32)
        responses_dt = np.loadtxt('./date_time_generalresponses.data', np.float32)
        responses_dt = responses_dt.reshape((responses_dt.size, 1))
        model_dt = cv2.ml.KNearest_create()
        model_dt.train(samples_dt, cv2.ml.ROW_SAMPLE, responses_dt)
        
        roi_dt = frame[51:131, 28:873]
        # roi_dt = frame[59:112, 822:870] # one digit
        out_dt = np.zeros(roi_dt.shape,np.uint8)
        gray_dt = cv2.cvtColor(roi_dt,cv2.COLOR_BGR2GRAY)
        blur_dt = cv2.GaussianBlur(gray_dt,(5,5),0)
        _, thresh_dt = cv2.threshold(blur_dt, 127,255, cv2.THRESH_OTSU)
        
        contours_dt,hierarchy_dt = cv2.findContours(thresh_dt,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


        for cnt_dt in contours_dt:
            if cv2.contourArea(cnt_dt)>300:           
                x, y, w, h = 28, 51, 845, 80
                [x, y, w, h] = cv2.boundingRect(cnt_dt)
                if  h>32:
                    cv2.rectangle(roi_dt,(x,y),(x+w,y+h),(0,255,0),2)
                    roi_dt = thresh_dt[y:y+h,x:x+w]
                    roismall = cv2.resize(roi_dt,(10,10))
                    roismall = roismall.reshape((1,100))
                    roismall = np.float32(roismall)
                    # for each in roismall:
                    retval, results, neigh_resp, dists = model_dt.findNearest(roismall, k = 1)
                    string_dt = str(int((results[0][0])))
                    # string_dt = str((results[0][0]))
                    cv2.putText(out_dt,string_dt,(x,y+h),0,1,(0,255,0))
                    print(string_dt)



        # # THIRD ROI : SPEED
        #         #######   training part    ############### 
        # samples_speed = np.loadtxt('./speed_generalsamples.data',np.float32)
        # responses_speed = np.loadtxt('./speed_generalresponses.data',np.float32)
        # responses_speed = responses_speed.reshape((responses_speed.size,1))

        # model = cv2.ml.KNearest_create()
        # model.train(samples_speed, cv2.ml.ROW_SAMPLE, responses_speed)

        # ############################# testing part  #########################


        # y_2,h_2,x_2,w_2 = 568,57,1028,97
        # roi_3 =frame[568:568+57, 1028:1028+97]
        # # roi_3_1 =frame[569:617, 1099:1128]
        # roi_3_1 =frame[569:617, 1099:1127]
        # roi_3_2 =frame[569:617, 1069:1098]
        # roi_3_3 =frame[569:617, 1039:1068]


        # # blur_3= converter(roi_3)
        # # text_3 = pytesseract.image_to_string(blur_3, config='-c tessedit_char_whitelist=0123456789 --psm 11 --oem 3')
        # #text_3 = text_3.strip()

        # # rec_2 = draw_rec(frame,"Speed", x_2,y_2,h_2,w_2)
        # out_1 = np.zeros(roi_3_1.shape,np.uint8)
        # out_2 = np.zeros(roi_3_2.shape,np.uint8)
        # out_3 = np.zeros(roi_3_3.shape,np.uint8)

        # gray_1 = cv2.cvtColor(roi_3_1,cv2.COLOR_BGR2GRAY)
        # gray_2 = cv2.cvtColor(roi_3_2,cv2.COLOR_BGR2GRAY)
        # gray_3 = cv2.cvtColor(roi_3_3,cv2.COLOR_BGR2GRAY)

        # blur_1 = cv2.GaussianBlur(gray_1,(5,5),0)
        # blur_2 = cv2.GaussianBlur(gray_2,(5,5),0)
        # blur_3 = cv2.GaussianBlur(gray_3,(5,5),0)
        # # thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

        # _, thresh_1 = cv2.threshold(blur_1, 127,255, cv2.THRESH_BINARY)
        # _, thresh_2 = cv2.threshold(blur_2, 127,255, cv2.THRESH_BINARY)
        # _, thresh_3 = cv2.threshold(blur_3, 127,255, cv2.THRESH_BINARY)

        # contours_1,hierarchy_1 = cv2.findContours(thresh_1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # contours_2,hierarchy_2 = cv2.findContours(thresh_2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # contours_3,hierarchy_3 = cv2.findContours(thresh_3,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


        # # 백의자리
        # for cnt_3 in contours_3:
        #     if cv2.contourArea(cnt_3)>50:           
        #         x, y, w, h = 1039, 569, 29, 48
        #         [x, y, w, h] = cv2.boundingRect(cnt_3)
        #         if  h>28:
        #             cv2.rectangle(roi_3_3,(x,y),(x+w,y+h),(0,255,0),2)
        #             roi_three = thresh_3[y:y+h,x:x+w]
        #             roismall = cv2.resize(roi_three,(10,10))
        #             roismall = roismall.reshape((1,100))
        #             roismall = np.float32(roismall)
        #             # for each in roismall:
        #             retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
        #             string_3 = str(int((results[0][0])))
        #             cv2.putText(out_3,string_3,(x,y+h),0,1,(0,255,0))
        #             print(string_3, end = '')


        # # # 십의자리  
        # for cnt_2 in contours_2:
        #     if cv2.contourArea(cnt_2)>50:          
        #         x, y, w, h = 1069, 569, 29, 48
        #         [x, y, w, h] = cv2.boundingRect(cnt_2)
        #         if  h>28:
        #             cv2.rectangle(roi_3_2,(x,y),(x+w,y+h),(0,255,0),2)
        #             roi_two = thresh_2[y:y+h,x:x+w]
        #             roismall = cv2.resize(roi_two,(10,10))
        #             roismall = roismall.reshape((1,100))
        #             roismall = np.float32(roismall)
        #             # for each in roismall:
        #             retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
        #             string_2 = str(int((results[0][0])))
        #             cv2.putText(out_2,string_2,(x,y+h),0,1,(0,255,0))
        #             print(string_2, end = '')


        # for cnt_1 in contours_1:
        #     if cv2.contourArea(cnt_1)>50:
        #         # 일의자리
        #         x, y, w, h = 1099, 569, 29, 48
        #         [x, y, w, h] = cv2.boundingRect(cnt_1 )
        #         if  h>28:
        #             cv2.rectangle(roi_3_1,(x,y),(x+w,y+h),(0,255,0),2)
        #             roi_one = thresh_1[y:y+h,x:x+w]
        #             roismall = cv2.resize(roi_one,(10,10))
        #             roismall = roismall.reshape((1,100))
        #             roismall = np.float32(roismall)
        #             # for each in roismall:
        #             retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
        #             string_1 = str(int((results[0][0])))
        #             cv2.putText(out_1,string_1,(x,y+h),0,1,(0,255,0))
        #             print(string_1)


                
        cv2.imshow('roi_dt', roi_dt)
        # cv2.imshow('roi_3',roi_3)
        # cv2.imshow('roi_3_3',roi_3_3)
        # cv2.imshow('roi_3_2',roi_3_2)
        # cv2.imshow('roi_3_1',roi_3_1)

        cv2.imshow('out_dt', out_dt)
        # cv2.imshow('out_3',out_3)
        # cv2.imshow('out_2',out_2)
        # cv2.imshow('out_1',out_1)


        cv2.moveWindow('roi_dt', 0, 100)
        # cv2.moveWindow('roi_3', 0, 200)
        # cv2.moveWindow('roi_3_3', 0, 300)
        # cv2.moveWindow('roi_3_2', 0, 400)
        # cv2.moveWindow('roi_3_1', 0, 500)
        cv2.moveWindow('out_dt', 0, 600 )
        # cv2.moveWindow('out_3', 0, 700 )
        # cv2.moveWindow('out_2', 0, 800)
        # cv2.moveWindow('out_1', 0, 900)




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
        cv2.imshow("Frame", frame)
        # cv2.imshow("blur_1",blur_1)
        # cv2.imshow("blur_2",blur_2)
        # cv2.imshow("Date_time",date_time)
        #cv2.imshow("Speed",blur)
        # cv2.imshow("hsv_green_lane", hsv_white_lane)
        # cv2.imshow("hsv_white_lane", hsv_green_lane)
        # cv2.imshow("hsv_green_auto", hsv_white_auto)
        # cv2.imshow("hsv_white_auto", hsv_green_auto)
        
        cv2.moveWindow("Frame", 100, 0)
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