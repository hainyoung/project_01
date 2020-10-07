import sys
import numpy as np
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

cap = cv2.VideoCapture('./use_video.mp4')

while True:
    ret, frame = cap.read()

    # 1. date_roi
    date_roi = frame[38:140, 25:490]
    date_gray = cv2.cvtColor(date_roi, cv2.COLOR_BGR2GRAY)
    # thresh_date, date_bin = cv2.threshold(date_gray, 50, 255, cv2.THRESH_BINARY)
    # OTSU
    thresh_date, date_bin = cv2.threshold(date_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # print(thresh_date) # 61
    # blur_date = cv2.medianBlur(date_bin, 1)
    # blur_date = cv2.resize(blur_date, None, fx = 2, fy = 2, interpolation = cv2.INTER_LINEAR)
    date = pytesseract.image_to_string(date_bin, config='-c tessedit_char_whitelist=-0123456789 --psm 6 --oem 3')

    # 2. time_roi    
    time_roi = frame[38:140, 513:890]
    time_gray = cv2.cvtColor(time_roi, cv2.COLOR_BGR2GRAY)
    # thresh_time, time_bin = cv2.threshold(time_gray, 70, 255, cv2.THRESH_BINARY)
    # OTSU
    thresh_time, time_bin = cv2.threshold(time_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # print(thresh_time) # 60
    # blur_time = cv2.medianBlur(time_bin, 1)
    # blur_time = cv2.resize(blur_time, None, fx = 2, fy = 2, interpolation = cv2.INTER_LINEAR)
    time = pytesseract.image_to_string(time_bin, config='-c tessedit_char_whitelist=:0123456789 --psm 6 --oem 3')

    # 3. speed_roi
    speed_roi = frame[524:613, 1146:1260]
    speed_gray = cv2.cvtColor(speed_roi, cv2.COLOR_BGR2GRAY)
    # thresh_speed, speed_bin = cv2.threshold(speed_gray, 180, 255, cv2.THRESH_BINARY)
    # OTSU
    thresh_speed, speed_bin = cv2.threshold(speed_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # print(thresh_speed) # 170
    blur_speed = cv2.medianBlur(speed_bin, 1)
    blur_speed = cv2.resize(blur_speed, None, fx = 2, fy = 2, interpolation = cv2.INTER_LINEAR)
    speed = pytesseract.image_to_string(speed_bin, config='-c tessedit_char_whitelist=0123456789 --psm 6 --oem 3')

    # 4. lane_roi
    lane_roi = frame[417:476, 1079:1154]
    # print(lane_roi)
    # print(type(lane_roi)) # <class 'numpy.ndarray'>

    # lane_hsv = cv2.cvtColor(lane_roi, cv2.COLOR_BGR2HSV)
    # lower_green_lane = np.array([50, 100, 100])
    # upper_green_lane = np.array([70, 255, 255])
    # lane_mask = cv2.inRange(lane_hsv, lower_green_lane, upper_green_lane)
    # lane_frame = cv2.bitwise_and(lane_roi, lane_roi, mask = lane_mask)

    # 5. auto_roi
    auto_roi = frame[436:491, 1341:1410]
    auto_roi = cv2.cvtColor(auto_roi, cv2.COLOR_BGR2HSV)
    lower_green_auto = np.array([50, 100, 100])
    upper_green_auto = np.array([70, 255, 255])
    auto_mask = cv2.inRange(auto_roi, lower_green_auto, upper_green_auto)
    auto_frame = cv2.bitwise_and(auto_roi, auto_roi, mask = auto_mask)

    # avg_color_per_row = np.average(auto_frame, axis = 0)
    # avg_color = np.average(avg_color_per_row, axis = 0)

    # for i in avg_color:
    #     if i > 1 :
    #         print("1")
    #     else:
    #         print("0")


    # if not ret:
    #     print('error')
    #     break

    # cv2.imshow('frame', frame)
    cv2.imshow('date', date_bin)
    cv2.imshow('time', time_bin)
    cv2.imshow('speed', speed_bin)
    cv2.imshow('lane', lane_roi)
    cv2.imshow('auto', auto_frame)

    cv2.moveWindow('date', 0, 0)
    cv2.moveWindow('time', 0, 100)
    cv2.moveWindow('speed', 0, 200)
    cv2.moveWindow('lane', 0, 300)
    cv2.moveWindow('auto', 0, 400)

    # sys.stdout = open('output.txt', 'a', -1, "utf-8")
    print(date, time, speed, end = '')  

    # k = cv2.waitKey(1) & 0xFF
    if cv2.waitKey(33) > 0:
        break

cap.release()
cv2.destroyAllWindows()

