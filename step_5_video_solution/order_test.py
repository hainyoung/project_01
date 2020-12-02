import numpy as np
import cv2
import sys

sys.path.append("C:\\Users\\Fazliddin\\Desktop\\step_4_video_solution\\order_func.py") 

# Importing the functions 
from order_func import *

#Videos from 1015  STEP 3

# cap = cv2.VideoCapture("./dim_10_10frame.mp4")
# cap = cv2.VideoCapture("./dim_10_20frame.mp4")
# cap = cv2.VideoCapture("./outdoor_12_brightness_max_22frame.mp4")
# cap = cv2.VideoCapture("./outdoor_12_saturation_max_22frame.mp4")
# cap = cv2.VideoCapture("./indoor_10_22frame.mp4")
# cap = cv2.VideoCapture("./standard_15_22frame.mp4")
# cap = cv2.VideoCapture("./dim_10_fullframe.mp4")
# cap = cv2.VideoCapture("./outdoor_12_22frame.mp4")
# cap = cv2.VideoCapture("./outdoor_8_22frame.mp4")
# cap = cv2.VideoCapture("./indoor_15_22frame.mp4")
# cap = cv2.VideoCapture("./indoor_12_22frame.mp4")

#Step 4 1023 videos
# cap = cv2.VideoCapture("./1023_1.mp4")
# cap = cv2.VideoCapture("./1023_2.mp4")
# cap = cv2.VideoCapture("./daejon_sample_1.mp4")


#Step 5
# cap = cv2.VideoCapture("./1107_inside.mp4")

# final
cap = cv2.VideoCapture("./1127/r_1_1_inside.mp4")

print("rtime, speed, ldws, lkas")

c = 1
while(cap.isOpened()):
	ret, frame = cap.read()
	# sys.stdout = open('./1107_order_output.txt', 'a', -1, 'utf-8')     

	if c%10==0:

		# ROI: DATE and TIME
		x_1,y_1,h_1,w_1 = 18,41,90,863
		rec_1 = draw_rec(frame,"Current Time", x_1,y_1,h_1,w_1) 
		dt = date_time_recognizer(frame,28,51,80,845)           
		date_time = "".join(dt)
		date_time = date_time[::-1]
		print(date_time[:4] + "-" + date_time[4:6] + "-" + date_time[6:8], end = " ")
		print(date_time[8:10] + ":" + date_time[10:12] + ":" + date_time[12:14], end = ",", sep = ',')
		# print("date:" + date_time[:4] + "-" + date_time[4:6] + "-" + date_time[6:8], end = ", ")
		# print("time:" + date_time[8:10] + ":" + date_time[10:12] + ":" + date_time[12:14], end = ", ")

		# # ROI: SPEED  STEP 4 VIDEO
		# x_2,y_2,h_2,w_2 = 919,483,88,178
		# rec_2 = draw_rec(frame,"Speed", x_2,y_2,h_2,w_2) 
		# roi_speed = speed_digit_recognizer(frame,940,497,59,111)
		# speed = "".join(roi_speed)
		# speed = speed[::-1]
		# print("speed:" + speed, end = ",")


		# ROI: SPEED  STEP 5 11_07
		# x_2,y_2,h_2,w_2 = 950,590,80,178
		# rec_2 = draw_rec(frame,"Speed", x_2,y_2,h_2,w_2) 
		# roi_speed = speed_digit_recognizer(frame,993,607,48,86)
		# # speed = "".join(roi_speed)
		# # speed = speed[::-1]
		# roi_speed = str(roi_speed)
		# print("speed:" + roi_speed)

		# x_2,y_2,h_2,w_2 = 950,590,80,178
		y_or,h_or,x_or,w_or = 590,80,950,178
		rec_2 = draw_rec(frame,"Speed", x_or,y_or,h_or,w_or)

		samples_speed = np.loadtxt('./models_OTSU_newimage/generalsamples.data',np.float32)
		responses_speed = np.loadtxt('./models_OTSU_newimage/generalresponses.data',np.float32)


		responses_speed = responses_speed.reshape((responses_speed.size,1))
		model = cv2.ml.KNearest_create()
		model.train(samples_speed, cv2.ml.ROW_SAMPLE, responses_speed)
		roi_speed = frame[y_or:y_or+h_or, x_or:x_or+w_or]
		out_speed = np.zeros(roi_speed.shape,np.uint8)
		gray_speed = cv2.cvtColor(roi_speed,cv2.COLOR_BGR2GRAY)
		blur_speed = cv2.GaussianBlur(gray_speed,(5,5),0)
		#_, thresh_speed = cv2.threshold(blur_speed, 127,255, cv2.THRESH_BINARY)
		_, thresh_speed = cv2.threshold(blur_speed, 0,255, cv2.THRESH_OTSU)

		contours_speed,hierarchy_speed = cv2.findContours(thresh_speed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		# print(contours_speed)
		speed_0 = []
		speed_1 = []
		speed_2 = []
		for cnt_speed in contours_speed:
			if cv2.contourArea(cnt_speed)<800:
				[x, y, w, h] = cv2.boundingRect(cnt_speed)
				# print(x)
				if h > 28:
				# print(x)
					cv2.rectangle(roi_speed,(x,y),(x+w,y+h),(0,255,0),1)
					roi_sp = thresh_speed[y:y+h,x:x+w]
					roismall = cv2.resize(roi_sp,(10,10))
					roismall = roismall.reshape((1,100))
					roismall = np.float32(roismall)
					# print("{},{}".format(x,w), end = ",")
					# # for each in roismall:
					retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
					# print(retval)
					string_speed= str(int((results[0][0])))
					# string_speed= int((results[0][0]))
					# cv2.putText(out_speed,string_speed,(x,y+h),0,1,(0,255,0))
					
					if 40<x<70:
						speed_0.append(string_speed)
					elif 70<x<110:
						speed_1.append(string_speed)
					elif x>110:
						speed_2.append(string_speed)

		if not speed_0 and not speed_1:
			print("".join(speed_2), end = ',', sep = ',')
		elif not speed_0:
			speed_list = speed_1 + speed_2
			print("".join(speed_list), end = ',', sep = ',')
		else :
			speed_list = speed_0 + speed_1 + speed_2
			print("".join(speed_list), end = ',', sep = ',')


		# ROI: LANE
		y_3,h_3,x_3,w_3 = 507,29,906,43
		roi_4 = frame[y_3:y_3+h_3,x_3:x_3+w_3]
		hsv_green_lane = hsv_green(roi_4)
		hsv_white_lane = hsv_white(roi_4)
		rec_3 = draw_rec(frame,"LDWS", x_3,y_3,h_3,w_3)

		hsv_green_lane_per_row = np.average(hsv_green_lane, axis = 0)
		avg_green_lane_ = np.average(hsv_green_lane_per_row, axis = 0)
		avg_green_lane = np.average(avg_green_lane_)

		hsv_white_lane_per_row = np.average(hsv_white_lane, axis = 0)
		avg_white_lane_ = np.average(hsv_white_lane_per_row, axis = 0)
		avg_white_lane = np.average(avg_white_lane_)

		# # FIFTH ROI : AUTO
		y_4,h_4,x_4,w_4 = 513,33,1199,36
		roi_5 = frame[y_4:y_4+h_4,x_4:x_4+w_4]
		hsv_green_auto = hsv_green(roi_5)
		hsv_white_auto = hsv_white(roi_5)
		rec_4 = draw_rec(frame,"LKAS", x_4,y_4,h_4,w_4)

		hsv_green_auto_per_row = np.average(hsv_green_auto, axis = 0)
		avg_green_auto_ = np.average(hsv_green_auto_per_row, axis = 0)
		avg_green_auto = np.average(avg_green_auto_)

		hsv_white_auto_per_row = np.average(hsv_white_auto, axis = 0)
		avg_white_auto_ = np.average(hsv_white_auto_per_row, axis = 0)
		avg_white_auto = np.average(avg_white_auto_)

		# lane
		if avg_green_lane > 0 or avg_white_lane > 0:
		    print("1", end = ',', sep = ',')
		else : 
		    print("0", end = ',', sep = ',')

		# auto       
		if avg_green_auto > 0 or avg_white_auto > 0:
		    print("1")
		else :
		    print("0")


		cv2.imshow("Frame", frame)    
		# cv2.moveWindow("Frame",0, 0)

		if(cv2.waitKey(1) & 0xFF == ord('q')):
			break
			#Show all the windows

	c+=1

cap.release()
# out.release()
cv2.destroyAllWindows()