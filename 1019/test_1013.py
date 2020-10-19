import pytesseract
import numpy as np
import cv2
import pandas as pd
import sys
import time

start = time.time()
# cap = cv2.VideoCapture("./use_video.mp4")
cap = cv2.VideoCapture("./night_mode.mp4")
# cap = cv2.VideoCapture("./night_light.mp4")
# cap = cv2.VideoCapture("./cut_video.mp4")
# cap = cv2.VideoCapture("./video_2.mp4")
# cap = cv2.VideoCapture("./1min.mp4")

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


c = 1

while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    if c % 10  == 0:
        # cv2.waitKey(1)

    
        # # FIRST ROI : DATE
        # roi_1 = frame[38:140, 25:490]
        # blur_1 = converter(roi_1)
        # text_1 = pytesseract.image_to_string(blur_1, config='-c tessedit_char_whitelist=-:0123456789 --psm 13 --oem 3')
        # text_1 = text_1.strip()

        # SECOND ROI : TIME
        # roi_2 =frame[38:144,513:890]
        # roi_2 =frame[51:131,513:890]
        # blur_2 = converter(roi_2)
        # text_2 = pytesseract.image_to_string(blur_2, config='-c tessedit_char_whitelist=:0123456789 --psm 13 --oem 3')
        # text_2 = text_2.strip()
    
        # # DATE&TIME COMBINED ROI 
        date_time =frame[51:131,28:873]
        date_time = converter(date_time)
        text_date_time = pytesseract.image_to_string(date_time, config='-c tessedit_char_whitelist=-:0123456789 --psm 13 --oem 1')
        text_date_time = text_date_time.strip()


        # THIRD ROI : SPEED
        roi_3 =frame[524:613,1146:1260]
        blur_3= converter(roi_3)
        text_3 = pytesseract.image_to_string(blur_3, config='-c tessedit_char_whitelist=0123456789 --psm 3 --oem 3')
        text_3 = text_3.strip()

        # # FOURTH ROI : LANE
        roi_4 = frame[417:476,1079:1154]
        hsv_green_lane = hsv_green(roi_4)
        hsv_white_lane = hsv_white(roi_4)

        hsv_green_lane_per_row = np.average(hsv_green_lane, axis = 0)
        avg_green_lane_ = np.average(hsv_green_lane_per_row, axis = 0)
        avg_green_lane = np.average(avg_green_lane_)
        avg_green_lane = round(avg_green_lane, 4)

        hsv_white_lane_per_row = np.average(hsv_white_lane, axis = 0)
        avg_white_lane_ = np.average(hsv_white_lane_per_row, axis = 0)
        avg_white_lane = np.average(avg_white_lane_)
        avg_white_lane = round(avg_white_lane, 4)

        # print(avg_white_lane)
        # print(avg_green_lane)

        # # FIFTH ROI : AUTO
        roi_5 = frame[436:491,1341:1410]
        hsv_green_auto = hsv_green(roi_5)
        hsv_white_auto = hsv_white(roi_5)

        hsv_green_auto_per_row = np.average(hsv_green_auto, axis = 0)
        avg_green_auto_ = np.average(hsv_green_auto_per_row, axis = 0)
        avg_green_auto = np.average(avg_green_auto_)
        avg_green_auto = round(avg_green_auto, 4)

        hsv_white_auto_per_row = np.average(hsv_white_auto, axis = 0)
        avg_white_auto_ = np.average(hsv_white_auto_per_row, axis = 0)
        avg_white_auto = np.average(avg_white_auto_)
        avg_white_auto = round(avg_white_auto, 4)

        # print("white average")
        # print(avg_white_auto)
        # print("green average")
        # print(avg_green_auto)

        #Show all the windows
        cv2.imshow("Frame", frame)
        # cv2.imshow("blur_1",blur_1)
        # cv2.imshow("blur_2",blur_2)
        cv2.imshow("date_time",date_time)
        cv2.imshow("speed",blur_3)
        cv2.imshow("hsv_green_lane", hsv_white_lane)
        cv2.imshow("hsv_white_lane", hsv_green_lane)
        cv2.imshow("hsv_green_auto", hsv_white_auto)
        cv2.imshow("hsv_white_auto", hsv_green_auto)
        
        cv2.moveWindow("Frame", 0,0)
        # cv2.moveWindow('blur_1', 0,0)
        # cv2.moveWindow('blur_2', 0,250)
        cv2.moveWindow('date_time', 0,250)
        cv2.moveWindow('speed', 0,500)        
        cv2.moveWindow('hsv_green_lane', 0,700)
        cv2.moveWindow('hsv_white_lane', 150,700)
        cv2.moveWindow('hsv_green_auto', 0,900)
        cv2.moveWindow('hsv_white_auto', 150,900)
        
        # sys.stdout = open('output_cut_video_3frame_datetime.txt', 'a', -1, 'utf-8')
        # print the results
        print(text_date_time, text_3, end = ',', sep = ',')
        
        # lane
        # print("average : lane")
        if avg_green_lane > 0 :
            print("1", end = ',', sep = ',')
        else : 
            print("0", end = ',', sep = ',')
            # print("0", end = ',', sep = ',')

        # auto
        # print("average : auto")
        if avg_green_auto > 0 or avg_white_auto > 0 :
            print("1")
        else :
            print("0")
            # print("0", end = ',', sep = ',')

        # avg_lane & avg_auto
        # print(avg_white_lane, end = ',', sep = ',')
        # print(avg_green_lane, end = ',', sep = ',')
        # print(avg_white_auto, end = ',', sep = ',')
        # print(avg_green_auto)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    c += 1
    


cap.release()
# out.release()
cv2.destroyAllWindows()

# print("time :", time.time() - start)


# output_cut_video_3frame.txt : time : 4538.550533294678 
# -> 1"15"38 / time, speed, lane, auto, lane_white_avg, lane_green_avg, auto_white_avg, auto_green_avg

# output_cut_video_3frame_datetime.txt : time : 5289.195523023605
# -> 1"28"09 / date_time, speed, lane, auto, lane_white_avg, lane_green_avg, auto_white_avg, auto_green_avg
