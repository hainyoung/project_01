import cv2


def main():
    # these are the limits of the cropped area
    # x_0 = 822
    # x_1 = 870
    # y_0 = 59
    # y_1 = 112
    x_0 = 960
    x_1 = 1050
    y_0 = 500
    y_1 = 556


    #y_2,h_2,x_2,w_2 = 568,57,1028,97

    # y_2,h_2,x_2,w_2 = 500,56,960,90 # 1023_output_video

    # cap = cv2.VideoCapture('./condition_2.mp4')
    cap = cv2.VideoCapture('./1023_video_list/1023_output_video.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')

    # passing the dimensions of cropped area to VideoWriter
    # out_video = cv2.VideoWriter('1023_output_video_recording.avi', fourcc, 15.0, (y_1-y_0, x_1-x_0))
    out_video = cv2.VideoWriter('./1023_output_video_recording.avi', fourcc, 22.0, (556-500, 1050-960))

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            frame_crop = frame[y_0:y_1, x_0:x_1]
            out_video.write(frame_crop)
            cv2.imshow("crop", frame_crop)
            key = cv2.waitKey(25)
            if key == ord('q'):
                break
        else:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
