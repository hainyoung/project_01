import numpy as np
import cv2
import sys

sys.path.append("C:\\Users\\Fazliddin\\Desktop\\step_4_video_solution\\functions_step_4.py") 

# Importing the functions 
from functions_step_4 import *

#Videos from 1015  STEP 3

# cap = cv2.VideoCapture("./dim_10_10frame.mp4")
# cap = cv2.VideoCapture("./dim_10_20frame.mp4")
# cap = cv2.VideoCapture("./outdoor_12_brightness_max_22frame.mp4")
# cap = cv2.VideoCapture("./outdoor_12_saturation_max_22frame.mp4")
# cap = cv2.VideoCapture("./indoor_10_22frame.mp4")
# cap = cv2.VideoCapture("./standard_15_22frame.mp4")
# cap = cv2.VideoCapture("./dim_10_fullframe.mp4")
# cap = cv2.VideoCapture("./outdoor_12_22frame.mp4")
# cap = cv2.VideoCapture("./outdoor_8_22frame.mp4")
# cap = cv2.VideoCapture("./indoor_15_22frame.mp4")
# cap = cv2.VideoCapture("./indoor_12_22frame.mp4")

#Step 4 1023 videos
# cap = cv2.VideoCapture("./1023_1.mp4")
# cap = cv2.VideoCapture("./1023_2.mp4")
# cap = cv2.VideoCapture("./daejon_sample_1.mp4")

#Step 5
cap = cv2.VideoCapture("./1107_inside.mp4")



c = 1
while(cap.isOpened()):
    ret, frame = cap.read()
    if c%10==0:

        # # ROI: DATE and TIME
        # x_1,y_1,h_1,w_1 = 18,41,90,863
        # rec_1 = draw_rec(frame,"Current Time", x_1,y_1,h_1,w_1) 
        # dt = date_time_recognizer(frame,28,51,80,845)           
        # date_time = "".join(dt)
        # date_time = date_time[::-1]
        # # print(date_time[:4] + "-" + date_time[4:6] + "-" + date_time[6:8], end = ",")
        # # print(date_time[8:10] + ":" + date_time[10:12] + ":" + date_time[12:14], end = ",")
        # print("date:" + date_time[:4] + "-" + date_time[4:6] + "-" + date_time[6:8], end = ", ")
        # print("time:" + date_time[8:10] + ":" + date_time[10:12] + ":" + date_time[12:14], end = ", ")

        # # # ROI: SPEED  STEP 4 VIDEO
        # # x_2,y_2,h_2,w_2 = 919,483,88,178
        # # rec_2 = draw_rec(frame,"Speed", x_2,y_2,h_2,w_2) 
        # # roi_speed = speed_digit_recognizer(frame,940,497,59,111)
        # # speed = "".join(roi_speed)
        # # speed = speed[::-1]
        # # print("speed:" + speed, end = ",")


        # # ROI: SPEED  STEP 5 11_07
        # x_2,y_2,h_2,w_2 = 950,590,80,178
        # rec_2 = draw_rec(frame,"Speed", x_2,y_2,h_2,w_2) 
        # roi_speed = speed_digit_recognizer(frame,993,607,48,86)
        # # speed = "".join(roi_speed)
        # # speed = speed[::-1]
        # roi_speed = str(roi_speed)
        # print("speed:" + roi_speed)



        # ROI: LANE
        y_3,h_3,x_3,w_3 = 385,40,852,60
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
        y_4,h_4,x_4,w_4 = 392,40,1140,60
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
            print("LDWS:" + "1", end = ', ', sep = ',')
        else : 
            print("LDWS:" + "0", end = ', ', sep = ',')

        # auto       
        if avg_green_auto > 0 or avg_white_auto > 0:
            print("LKAS:" + "1")
        else :
            print("LKAS:" + "0")

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
        # sys.stdout = open('daejon_sample_2.txt', 'a', -1, 'utf-8')     
        #Show all the windows
        cv2.imshow("Frame", frame)    
        # cv2.moveWindow("Frame",0, 0)
        
    c+=1
cap.release()
# out.release()
cv2.destroyAllWindows()