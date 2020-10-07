import sys
import cv2
import numpy as np

'''
# lane

green = cv2.imread("./001_project/data/old/on_off.bmp")
green = green[417:476, 1079:1154]
# cv2.imshow('green', green)


white = cv2.imread("./001_project/data/old/off_off.bmp")
white = white[436:491, 1341:1410]
# cv2.imshow('white', white)

cv2.waitKey()
cv2.destroyAllWindows()


print("green.shape :", green.shape)
print(green)

print("-----------------------------------------------")
print("white.shape :", white.shape)
print(white)

green_mean = np.mean(green)
print(green_mean) # 44.27841807909604

print(np.min(green)) # 0
print(np.max(green)) # 255

white_mean = np.mean(white)
print(white_mean) # 33.612648221343875
print(np.min(white)) # 28
print(np.max(white)) # 51
'''

img_color = cv2.imread("./001_project/data/old/on_off.bmp", cv2.IMREAD_COLOR)

height, width, channel = img_color.shape
img_gray = np.zeros((height, width), np.uint8)

for y in range(0, height):
    for x in range(0, width):
        # case1 : color image
        b = img_color.item(y, x, 0)
        g = img_color.item(y, x, 1)
        r = img_color.item(y, x, 2)

        gray = (int(b) + int(g) + int(r)) / 3.0

        # case2 : grayscale image
        img_gray.itemset(y, x, gray)

cv2.imshow('bgr', img_color)
cv2.imshow('gray', img_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()