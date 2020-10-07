import sys
import numpy as np
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def showvideo():
    try:
        cap = cv2.VideoCapture('./001_project/data/cut_video.avi')
        print('playing the video')

    except:
        print('palying the video failed')
        return

    while True:
        ret, frame = cap.read()
        # print(frame.shape)

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # _, frame_bin = cv2.threshold(frame, 77, 255, cv2.THRESH_BINARY)
        # th, frame_bin = cv2.threshold(grayframe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # print(th)

        date_frame = frame[54:91, 32:263]
        date = pytesseract.image_to_string(date_frame, config='-c tessedit_char_whitelist=-0123456789 --psm 6')
        
        time_frame = frame[55:92, 268:467]
        time = pytesseract.image_to_string(time_frame, config='-c tessedit_char_whitelist=:0123456789 --psm 6')


        speed_frame = frame[542:586, 1190:1260]
        speed = pytesseract.image_to_string(speed_frame, config='-c tessedit_char_whitelist=0123456789 --psm 6')

        lane_frame = frame[433:462, 1091:1130]
        
        auto_frame = frame[451:480, 1365:1395]

        if not ret:
            print('error')
            break

        # cv2.imshow('frame', frame)
        cv2.imshow('date_frame', date_frame)
        cv2.imshow('time_frame', time_frame )
        cv2.imshow('speed_frame', speed_frame )
        # cv2.imshow('lane_frame', lane_frame )
        # cv2.imshow('auto_frame', auto_frame )

        # sys.stdout = open('output.txt', 'a', -1, "utf-8")
        # print(date, end=

        # k = cv2.waitKey(1) & 0xFF
        if cv2.waitKey(33) > 0:
            break

    cap.release()
    cv2.destroyAllWindows()

showvideo()
