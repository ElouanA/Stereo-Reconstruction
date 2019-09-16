import cv2
import time

CAMERA_WIDTH=620
CAMERA_HEIGHT=360
left = cv2.VideoCapture(0)
left.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
left.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
left.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
right = cv2.VideoCapture(1)
right.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
right.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
right.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
t=0
while (True):
    if not(right.grab() and left.grab()):
        print("no frame")
    else: 
        _,leftFrame=left.retrieve()
        _,rightFrame=right.retrieve()

        cv2.imshow('left',leftFrame)
        cv2.imshow('right',rightFrame)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.imwrite('D (' +str(t)+').jpg',rightFrame)
            cv2.imwrite('G ('+str(t)+').jpg',leftFrame)
            t+=1
            time.sleep(0.75)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 