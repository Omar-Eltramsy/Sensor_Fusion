import serial
from ultralytics import YOLO   
import cv2 as cv
import numpy as np
import Camera_Calibration
import convert3D_2D

def load(capFile):
    data=np.load(capFile)
    print(data)
    return data['camMatrix'],data['distcoeff'],data['rvec'],data['tvec']


def detect_objects(frame, model):
    results = model(frame)  # Perform inference
    detected_centers = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            detected_centers.append((center_x, center_y))
    return detected_centers

def main():
    # Configuration 
    calibration_file = 'calibration.npz'
    serial_port = 'COM5'
    baud_rate = 9600
    camera_index = 0 # Default camera

    # Load Calibration Data 
    camMatrix, distcoeff, rvec, tvec = load(calibration_file)

    try:
        # Initialize Serial Connection 
        ser = serial.Serial(serial_port, baud_rate,timeout=1) 
    except serial.SerialException as e:
        print(f'failed to open serial port {serial_port}: {e}')
        return[]
    # Detect objects and overlay their centers
    model = YOLO("yolov8n.pt")  
    
    # Initialize Camera 
    cap = cv.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print('Error opening camera.')
        return
    
    while True:
        # Read distance and servo angle
        if ser.in_waiting>0:
            line=ser.readline().decode('utf-8').strip()
            distance,angle,x,y=map(float,line.split(','))
            worldpoint=np.array([[x,y,0]],dtype=np.float32)
            
            # project world to 2D
            projected=convert3D_2D.convert3D_2D(camMatrix,rvec,tvec,worldpoint)
            u,v=projected[0]
            
            # Display on camera feed
            ret,frame=cap.read()
            if not ret:
                print('Failed to capture frame.')
                break
            frame_height, frame_width, _ = frame.shape

            if 0 <= u < frame_width and 0 <= v < frame_height:  # Ensure u and v are defined
                cv.circle(frame,(int(u),int(v)),5,(255,0,255),-3)
            else:
                if abs(u) > frame_width or abs(v) > frame_height :
                    print(f"Projected point ({u}, {v}) is too far from the camera.")
                    continue

            detected_centers = detect_objects(frame, model)
            for center_x, center_y in detected_centers:
                cv.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)  

            cv.imshow('Projected Point',frame)
            if cv.waitKey(1)& 0xFF==ord('x'):
                break
    cap.release()
    cv.destroyAllWindows()
    ser.close()

if __name__=='__main__':
    main()

