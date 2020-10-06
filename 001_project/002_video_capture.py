import cv2

vidcap = cv2.VideoCapture('./video.mp4')

count = 0

while(vidcap.isOpened()):
    ret, image = vidcap.read()

    if ret is False:
        print("Failed")

    else: 
        cv2.imwrite('./images/framed%d.jpg' % count, image)
        print('Saved frame%d.jpg' % count)

    count += 1

vidcap.release()
