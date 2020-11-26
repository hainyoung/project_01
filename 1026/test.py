import numpy as np
import cv2
import sys

# sys.path.append("C:\\Users\\Fazliddin\\Desktop\\FINAL\\functions.py") 

# Importing the functions 
from functions import *

#Videos from 1015  STEP 3
# cap = cv2.VideoCapture("./1015_video_list/dim_10_10frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/dim_10_16frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/dim_10_20frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/dim_10_fullframe.mp4")
# cap = cv2.VideoCapture("./1015_video_list/indoor_10_22frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/indoor_12_22frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/indoor_15_22frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/outdoor_8_22frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/outdoor_12_22frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/outdoor_12_brightness_max_22frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/outdoor_12_saturation_max_22frame.mp4")
# cap = cv2.VideoCapture("./1015_video_list/standard_15_22frame.mp4")

# 1023
# cap = cv2.VideoCapture("./1023_video_list/1023_cut_video.mp4")
cap = cv2.VideoCapture("./1023_video_list/daejeon_night.mp4")
# cap = cv2.VideoCapture("./1023_video_list/1023_output_video.mp4")



c = 1
while(cap.isOpened()):
    # sys.stdout = open('./1023_output.txt', 'a', -1, 'utf-8')     

    ret, frame = cap.read()

    if not ret:
        break

    if c%10==0:

        # ROI: DATE and TIME
        x_1,y_1,h_1,w_1 = 18,41,90,863
        rec_1 = draw_rec(frame,"Current Time", x_1,y_1,h_1,w_1) 
        dt = date_time_recognizer(frame,28,51,80,845)
                   
        date_time = "".join(dt)
        date_time = date_time[::-1]
        # print(date_time[:4] + "-" + date_time[4:6] + "-" + date_time[6:8], end = ",")
        # print(date_time[8:10] + ":" + date_time[10:12] + ":" + date_time[12:14], end = ",")
        # print("date:" + date_time[:4] + "-" + date_time[4:6] + "-" + date_time[6:8], end = ",")
        print(date_time[:4] + "-" + date_time[4:6] + "-" + date_time[6:8], end = ",")
        # print("time:" + date_time[8:10] + ":" + date_time[10:12] + ":" + date_time[12:14], end = ",")
        print(date_time[8:10] + ":" + date_time[10:12] + ":" + date_time[12:14], end = ",")

        # ROI: SPEED
        #y_2,h_2,x_2,w_2 = 524,89,1146,140 cut video
        #roi_date_time =frame[y_1:y_1+h_1,x_1:x_1+w_1] cut video
        
        # x_2,y_2,h_2,w_2 = 1029,550,77,136
        # x_2,y_2,h_2,w_2 = 950,490,77,136 # 1023 video
        x_2,y_2,h_2,w_2 = 1004,505,77,136 # daejeon video
        
        rec_2 = draw_rec(frame,"Speed", x_2,y_2,h_2,w_2) 
        
        # roi_speed = speed_digit_recognizer(frame,1039,560,57,96) # thresh O / mask X
        # roi_speed = mask_speed_digit_recognizer(frame,1039,560,57,96) # thresh X / mask O
        
        # roi_speed = speed_digit_recognizer(frame,960,500,56,88) # 1023 video
        # roi_speed = mask_speed_digit_recognizer(frame,960,500,56,88) # 1023 video

        roi_speed = speed_digit_recognizer(frame,1010,511,74,134)        

        #roi_speed = speed_digit_recognizer(frame,1146,524,89,140) cut video
        speed = "".join(roi_speed)
        speed = speed[::-1]

        # speed = int(speed)
        # print(type(speed))
        if not speed:
            # print("speed:", end = ',')
            print("", end = ',')

        elif int(speed) > 180:
            # print("speed:", end = ',')
            print("", end = ',')

        else:
            # print("speed:" + speed, end = ",")
            print(speed, end = ",")

        # ROI: LANE
        # y_3,h_3,x_3,w_3 = 450,43,933,52 # 1015 video
        y_3,h_3,x_3,w_3 = 393,29,860,44 # 1023 video
        # y_3,h_3,x_3,w_3 = 405,33,914,44 # daejeon video
        
        roi_4 = frame[y_3:y_3+h_3,x_3:x_3+w_3]
        hsv_green_lane = hsv_green(roi_4)
        hsv_white_lane = hsv_white(roi_4)
        rec_3 = draw_rec(frame,"LDWS", x_3,y_3,h_3,w_3)

        hsv_green_lane_per_row = np.average(hsv_green_lane, axis = 0)
        avg_green_lane_ = np.average(hsv_green_lane_per_row, axis = 0)
        avg_green_lane = np.average(avg_green_lane_)

        hsv_white_lane_per_row = np.average(hsv_white_lane, axis = 0)
        avg_white_lane_ = np.average(hsv_white_lane_per_row, axis = 0)
        avg_white_lane = np.average(avg_white_lane_)

        # # FIFTH ROI : AUTO
        # y_4,h_4,x_4,w_4 = 459, 40, 1217, 51 # 1015 video
        y_4,h_4,x_4,w_4 = 400,33,1153,36 # 1023 video
        # y_4,h_4,x_4,w_4 = 416,35,1204,39 # daejeon video
        
        roi_5 = frame[y_4:y_4+h_4,x_4:x_4+w_4]
        hsv_green_auto = hsv_green(roi_5)
        hsv_white_auto = hsv_white(roi_5)
        rec_4 = draw_rec(frame,"LKAS", x_4,y_4,h_4,w_4)

        hsv_green_auto_per_row = np.average(hsv_green_auto, axis = 0)
        avg_green_auto_ = np.average(hsv_green_auto_per_row, axis = 0)
        avg_green_auto = np.average(avg_green_auto_)

        hsv_white_auto_per_row = np.average(hsv_white_auto, axis = 0)
        avg_white_auto_ = np.average(hsv_white_auto_per_row, axis = 0)
        avg_white_auto = np.average(avg_white_auto_)
         
        # lane
        if avg_green_lane > 0 :
            # print("LDWS:" + "1", end = ',', sep = ',')
            print("1", end = ',', sep = ',')
        else : 
            # print("LDWS:" + "0", end = ',', sep = ',')
            print("0", end = ',', sep = ',')

        # auto       
        if avg_green_auto > 0 or avg_white_auto > 0:
            # print("LKAS:" + "1")
            print("1")
        else :
            # print("LKAS:" + "0")
            print("0")

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

        #Show all the windows
        cv2.imshow("Frame", frame)    
        cv2.moveWindow("Frame",0, 0)
        
    c+=1


cap.release()
# out.release()
cv2.destroyAllWindows()