import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

lower_green = (30, 30, 30)
upper_green = (70, 255, 255)

# img = mpimg.imread('./test_roi_01.jpg', cv2.IMREAD_COLOR)
img = mpimg.imread('./001_project/data/test.bmp', cv2.IMREAD_COLOR)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img_mask = cv2.inRange(img_hsv, lower_green, upper_green)

img_result = cv2.bitwise_and(img, img, mask=img_mask)

imgplot = plt.imshow(img_result)

plt.axis('off')
plt.show()