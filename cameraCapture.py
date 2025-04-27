#!/usr/bin/env python
import cv2 as cv

def cameraCapture():
    cap=cv.VideoCapture(0)
    if not cap.isOpened():
        print('Error:Could not access the camera')
        exit()
    try:
        while True:
            ret,frame=cap.read()
            if not ret:
                print("Error: Failed to grap Frame.")
                break
            cv.imshow('Camera Feed',frame)
            if cv.waitKey(1) & 0xFF == ord('x'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    
    cap.release()
    cv.destroyAllWindows()
