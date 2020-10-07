import sys
import cv2
import numpy as np

green = cv2.imread("./001_project/data/old/offoff.jpg")
green = green[417:476, 1079:1154]

white = cv2.imread("./001_project/data/old/onoff.jpg")
white = white[436:491, 1341:1410]

# print(type(src))

print("green.shape :", green.shape)
print(green)

print("-----------------------------------------------")
print("white.shape :", white.shape)
print(white)

green_mean = np.mean(green)
print(green_mean) # 98.92399246704332

white_mean = np.mean(white)
print(white_mean) # 46.903030303030306
