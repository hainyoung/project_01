import numpy as np
import cv2

img = cv2.imread("./001_project/images/onon.jpg")
cv2.namedWindow('original', cv2.WINDOW_NORMAL)
cv2.imshow('original', img)

subimg = img[545:590, 814:867] # y_min:y_max, x_min, x_max # opencv : height x width
cv2.namedWindow('cutting', cv2.WINDOW_NORMAL)
cv2.imshow('cutting', subimg)

print("original shape :", img.shape)
print("cutting shape :", subimg.shape)

cv2.waitKey()
cv2.destroyAllWindows()