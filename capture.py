import cv2
 
vidcap = cv2.VideoCapture('./light_speed_recording.avi')
 
count = 0
 
while(vidcap.isOpened()):
    ret, image = vidcap.read()
    if not ret :
        break
 
    cv2.imwrite("./crop_light_speed_imgs/light_speed%d.jpg" % count, image)
 
    print('Saved frame%d.jpg' % count)
    count += 1
 
vidcap.release()

