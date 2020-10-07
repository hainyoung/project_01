import numpy as np
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# cap = cv2.VideoCapture('./001_project/data/A01_20201006104159.avi')
# cut = cap[49:94, 17:472]

# while True:
#     ret1, frame1 = cap.read()
#     ret2, frame2 = cut.read()

#     cv2.imshow('frame1', frame1)
#     cv2.imshow('frame2', frame2)

#     if cv2.waitKey(33) > 0 :
#         break


# # img = cv2.imread("./test_cap.bmp")
# # cv2.namedWindow('original', cv2.WINDOW_NORMAL)
# # cv2.imshow('original', img)

# # subimg = img[49:94, 17:472]
# # cv2.namedWindow('cutting', cv2.WINDOW_NORMAL)
# # cv2.imshow('cutting', subimg)

# # cv2.waitKey()
# cap.release()
# cut.release()
# cv2.destroyAllWindows()

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

        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        time_frame = grayframe[24:124, 11:487]
        speed_frame = grayframe[533:598, 1164:1260]
        lane_frame = grayframe[419:470, 1087:1138]
        auto_frame = grayframe[444:488, 1355:1408]


        if not ret:
            print('error')
            break

        cv2.imshow('frame', grayframe)
        cv2.imshow('time_frame', time_frame )
        cv2.imshow('speed_frame', speed_frame )
        cv2.imshow('lane_frame', lane_frame )
        cv2.imshow('auto_frame', auto_frame )

        print("time :", pytesseract.image_to_string(time_frame, lang=None))
        print("speed :", pytesseract.image_to_string(speed_frame, lang=None))


        # k = cv2.waitKey(1) & 0xFF
        if cv2.waitKey(1) > 0:
            break

    cap.release()
    cv2.destroyAllWindows()

showvideo()
