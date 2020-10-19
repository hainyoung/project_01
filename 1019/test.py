import sys
import pytesseract
import numpy as np
import cv2
import pandas as pd

# pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

cap = cv2.VideoCapture("./use_video.mp4")
# cap = cv2.VideoCapture("./night_mode.mp4")
# cap = cv2.VideoCapture("./night_light.mp4")
# cap = cv2.VideoCapture("./cut_video.mp4")
# cap = cv2.VideoCapture("./video_2.mp4")

c = 1

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if c % 2 == 0 :
        cv2.waitKey(1)

    c += 1

    # # FIRST ROI : DATE
    # roi_1 = frame[38:140, 25:490]
    # gray_1 = cv2.cvtColor(roi_1, cv2.COLOR_BGR2GRAY)
    # # threshold the image using Otus method to preprocess for tesseract
    # thresh = cv2.threshold(gray_1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # # perform a median blur to smooth image slightly
    # blur_1 = cv2.medianBlur(thresh, 3)
    # # resize image to double the original size as tesseract does better with certain text size
    # blur_1 = cv2.resize(blur_1, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
    # #recognize the number
    # text_1 = pytesseract.image_to_string(blur_1, config='-c tessedit_char_whitelist=-0123456789 --psm 6 --oem 3')
       

    # # SECOND ROI : TIME
    roi_2 =frame[38:144, 513:890]
    gray_2 = cv2.cvtColor(roi_2, cv2.COLOR_BGR2GRAY)
    # threshold the image using Otsus method to preprocess for tesseract
    thresh = cv2.threshold(gray_2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # perform a median blur to smooth image slightly
    blur_2 = cv2.medianBlur(thresh, 3)
    # resize image to double the original size as tesseract does better with certain text size
    blur_2 = cv2.resize(blur_2, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
    #recognize the number
    text_2 = pytesseract.image_to_string(blur_2, config='-c tessedit_char_whitelist=:0123456789 --psm 13 --oem 3')
       

    # # THIRD ROI : SPEED
    # roi_3 =frame[524:613, 1146:1260]
    # gray_3 = cv2.cvtColor(roi_3, cv2.COLOR_BGR2GRAY)
    # # threshold the image using Otsus method to preprocess for tesseract
    # thresh = cv2.threshold(gray_3, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # # perform a median blur to smooth image slightly
    # blur_3 = cv2.medianBlur(thresh, 3)
    # # resize image to double the original size as tesseract does better with certain text size
    # blur_3 = cv2.resize(blur_3, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
    # #recognize the number
    # text_3 = pytesseract.image_to_string(blur_3, config='-c tessedit_char_whitelist=0123456789 --psm 6 --oem 3')  
       

    # FOURTH ROI : LANE
    roi_4 = frame[432:461, 1092:1129]
    hsv_1 = cv2.cvtColor(roi_4, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0,0,128])
    upper_white = np.array([255,255,255])

    lower_green = np.array([30, 30, 30])
    upper_green = np.array([70, 255, 255])

    mask_white_4 = cv2.inRange(hsv_1, lower_white, upper_white)
    mask_white_4 = cv2.bitwise_and(roi_4, roi_4, mask = mask_white_4)

    mask_green_4 = cv2.inRange(hsv_1, lower_green, upper_green)
    mask_green_4 = cv2.bitwise_and(roi_4, roi_4, mask = mask_green_4)

    # find average for setting threshold
    avg_roi_4_white_per_row = np.average(mask_white_4, axis = 0)
    avg_roi_4_white = np.average(avg_roi_4_white_per_row, axis = 0)
    # avg_roi_4_white = np.average(avg_roi_4_white_1)

    avg_roi_4_green_per_row = np.average(mask_green_4, axis = 0)
    avg_roi_4_green = np.average(avg_roi_4_green_per_row, axis = 0)
    # avg_roi_4_green = np.average(avg_roi_4_green_1)

    print(avg_roi_4_white)
    print(avg_roi_4_green)

    # print("avg_roi_4_green")
    # print(avg_roi_4_green, avg_roi_4_white)
    # print("avg_roi_4_white")
    # print(avg_roi_4_white)

    # if avg_l > 1 :
    #     print("1")
    # else :
        # print("0")
   
   
    # FIFTH ROI : AUTO
    roi_5 = frame[454:478, 1365:1393]

    # hsv version
    hsv_2 = cv2.cvtColor(roi_5, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0,0,128])
    upper_white = np.array([255,255,255])

    lower_green = np.array([30, 30, 30])
    upper_green = np.array([70, 255, 255])

    mask_white_5 = cv2.inRange(hsv_2, lower_white, upper_white)
    mask_white_5 = cv2.bitwise_and(roi_5, roi_5, mask = mask_white_5)
    
    mask_green_5 = cv2.inRange(hsv_2, lower_green, upper_green)
    mask_green_5 = cv2.bitwise_and(roi_5, roi_5, mask = mask_green_5)

    # find average for setting threshold
    avg_roi_5_white_per_row = np.average(mask_white_5, axis = 0)
    avg_roi_5_white_1 = np.average(avg_roi_5_white_per_row, axis = 0)
    avg_roi_5_white = np.average(avg_roi_5_white_1)

    avg_roi_5_green_per_row = np.average(mask_green_5, axis = 0)
    avg_roi_5_green_1 = np.average(avg_roi_5_green_per_row, axis = 0)
    avg_roi_5_green = np.average(avg_roi_5_green_1)

    # print("avg_roi_5_green")

    # sys.stdout = open('average_list.txt', 'a', -1, 'utf-8')
    # print(avg_roi_4_white, avg_roi_4_green, avg_roi_5_white, avg_roi_5_green, sep = ',')
    
    # print("avg_roi_5_white")
    # print(avg_roi_5_white)


    # if avg_lane > 1 :
    #     print("1")
    # else :
        # print("0")

    # print(mask_roi_5)
    # avg_roi_5_per_row = np.average(mask_green_5, axis = 0) # standard : row
    # print(avg_roi_5_per_row)
    # print(avg_roi_5_per_row.shape) # (69, 3)
    # avg_roi_5 = np.average(avg_roi_5_per_row, axis = 0)
    # print(avg_roi_5.shape) # (3,)
    # print(avg_roi_5) 

    # avg_auto = np.average(avg_roi_5)
    # print(avg_auto)

    # if avg_auto > 1 :
    #     print("1")
    # else :
    #     print("0")


    #show all the windows
    cv2.imshow("Frame", frame)
    # cv2.imshow("blur_1",blur_1)
    cv2.imshow("blur_2",blur_2)
    # cv2.imshow("blur_3",blur_3)
    cv2.imshow("mask_white_4", mask_white_4)
    cv2.imshow("mask_green_4", mask_green_4)
    cv2.imshow("mask_white_5", mask_white_5)
    cv2.imshow("mask_green_5", mask_green_5)
    # cv2.imshow("mask_roi_5", mask_roi_5)
    cv2.moveWindow("Frame",0,0)
    # cv2.moveWindow('blur_1', 0,0)
    cv2.moveWindow('blur_2', 0,240)
    # cv2.moveWindow('blur_3', 0,500)        
    cv2.moveWindow('mask_white_4', 0,730)
    cv2.moveWindow('mask_green_4', 150,730)
    cv2.moveWindow('mask_white_5', 0,830)
    cv2.moveWindow('mask_green_5', 150,830)
    # cv2.moveWindow('mask_roi_5', 0,930)

    # print the results
    # print(f'\r{text_1}{text_2}{text_3}', end = '')  


cap.release()
