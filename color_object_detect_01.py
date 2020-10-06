# HSV 이미지 사용하여 특정 색상 object 추출
# 구현
# 1. RGB 이미지를 입력받아 HSV 이미지로 변환
# 2. 색상의 범위에 따라 특정 색상의 객체를 추출하는 마스크 생성
# 3. 생성한 마스크에 따라 이미지를 계산하여 특정한 색상의 객체만 추출되는 결과 이미지를 만듦
# 4. opencv 사용해도 되고 안 해도 할 수 있다?


# RGB to HSV
# 특정 색상 객체를 추출하기 위해서 HSV 이미지가 왜 필요한가?

# RGB model
# RGB 모델은 빛의 삼원색을 이용하여 색을 표현하는 기본적인 색상 모델
# Red, Green, Blue 3가지 성분의 조합으로 표현
# R, G, B의 값은 0 ~ 255 사이 값들로 표현

# HSV 모델
# - HSV 모델은 인간의 색인지에 기반을 둔 색상 모델

# - Hue(색조), Saturation(채도), Value(명도), 3가지 성분의 조합으로 표현

# - Hue(색조) : 색의 종류. 0º~360º의 범위를 갖는다.

# - Saturation(채도) : 색의 선명도, 진함의 정도(가장 진한 상태를 100%로 한다)

# - Value(명도) : 색의 밝기, 밝은 정도(가장 밝은 상태를 100%로 한다)

 

# RGB 에서 HSV로 변환시키는 이유
# - RGB 이미지에서 색 정보를 검출하기 위해서는 R, G, B 세가지 속성을 모두 참고해야한다.

# - 하지만 HSV 이미지에서는 H(Hue)가 일정한 범위를 갖는 순수한 색 정보를 가지고 있기 때문에 RGB 이미지보다 쉽게 색을 분류할 수 있다.


import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# set color-range

# 01
# lower_green = (10, 80, 80)
# upper_green = (70, 255, 255)

# 02
lower_green = (36, 0, 0)
upper_green = (70, 255, 255)

# print(lower_green) # (30, 80, 80)
# a = print(lower_green)
# print(a)

# read image 
img = mpimg.imread('./001_project/images/onon.jpg', cv2.IMREAD_COLOR)

# convert BGR to HSV 
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# make a mask (*color range)
img_mask = cv2.inRange(img_hsv, lower_green, upper_green)

# make a result image
img_result = cv2.bitwise_and(img, img, mask=img_mask)

imgplot = plt.imshow(img_result)

# plt.show()

# save the result image
# cv2.imwrite('test01.jpg', img_result)
# cv2.imwrite('test02.jpg', img_result)
