#!/usr/bin/env python
import numpy as np
import serial.tools.list_ports
import serial
import time
import cv2 as cv
from ultralytics import YOLO   
import os

def Calibration(imgpath):

    # Define the chess board rows and columns
    row=9
    col=6

    # Set the termination criteria for the corner sub-pixel algorithm
    criteria=(cv.TermCriteria_EPS+cv.TermCriteria_MAX_ITER,30, 0.001)

    # Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
    worldptscur=np.zeros((row*col,3),np.float32)
    worldptscur[:,:2]=np.mgrid[0:row,0:col].T.reshape(-1,2)
    
    # Create arrays to store the objects points and the image points
    worldtPoint=[]
    imgPoint=[]

    img=cv.imread(imgpath)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cornerFound,corners=cv.findChessboardCorners(gray,(row,col),None)
    # Make sure the chess board pattern was found in the image
    if cornerFound:
        # Refine the corner position
        cornerRefind=cv.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
        
        worldtPoint.append(worldptscur)
        imgPoint.append(cornerRefind)
        
        cv.drawChessboardCorners(img,(row,col),cornerRefind,cornerFound)
        cv.imshow('Chess Board',img)
        cv.waitKey(1)
    else:
        print('Chessboard corners not found in the image.')
        return None,None,None,None
        
    cv.destroyAllWindows()
    
    # calibration
    repError,camMatrix,distcoeff,rvec,tvec=cv.calibrateCamera(worldtPoint,imgPoint,gray.shape[::-1],None,None)
    print('intrinsic Matrix:\n',camMatrix)
    print(f"Distortion Coefficients:\n{distcoeff}")

    # to save the paramter
    save_path=os.path.join(os.getcwd(),'calibration.npz')
    if not os.path.exists(save_path):
        np.savez(save_path,camMatrix=camMatrix,distcoeff=distcoeff,rvec=rvec,tvec=tvec)
        print('file is saved')
    else:
        print('File already exit')

def load(capFile):
    data=np.load(capFile)
    print(data)
    return data['camMatrix'],data['distcoeff'],data['rvec'],data['tvec']

'''
we use this equation 
u=K[R|t]X
to Convert world points into image coordinates.
'''

def convert3D_2D(camMatrix,rvec,tvec,worldPoint):
    R,_=cv.Rodrigues(rvec)
    '''
    T=[ R    tvec
        0   1    ]
    '''
    T=np.eye(4)
    T[:3,:3]=R
    T[:3,3]=tvec.flatten()
    print("Transformation Matrix:\n", T)
    # Project 3D points to 2D
    projectlist=[]
    for point in worldPoint:
        point_homogeneous = np.append(point, 1)
        
        # Transform to camera frame
        point_camera = T @ point_homogeneous
        
        # Project to image plane
        point_projected = camMatrix @ point_camera[:3]
        point_projected /= point_projected[2]  # Normalize by z-coordinate
        projectlist.append(point_projected[:2])
    print('Projected Points:\n',np.array(projectlist))
    return np.array(projectlist)


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

    # Initialize Serial Connection 
    ser = serial.Serial(serial_port, baud_rate) 

    # Detect objects and overlay their centers
    model = YOLO("yolov8n.pt")  
    
    # Initialize Camera 
    cap = cv.VideoCapture(camera_index)
    
    while True:
        # Read distance and servo angle
        if ser.in_waiting>0:
            line=ser.readline().decode('utf-8').strip()
            distance,angle,x,y=map(float,line.split(','))
            worldpoint=np.array([[x,y,0]],dtype=np.float32)
            
            # project world to 2D
            projected=convert3D_2D(camMatrix,rvec,tvec,worldpoint)
            u,v=projected[0]
            
            # Display on camera feed
            ret,frame=cap.read()
            if not ret:
                print('Failed to capture frame.')
            frame_height, frame_width, _ = frame.shape
            frame = cv.undistort(frame, camMatrix, distcoeff)

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



serial_port = 'COM5'
baud_rate = 9600
serial_object = serial.Serial(serial_port, baud_rate)
data = []

for _ in range(38):
    if serial_object.in_waiting:  # Check if data is available
        packet = serial_object.readline()
        val = packet.decode('utf-8').strip('\n').strip('\r') 
        data.append(val) 
        print(val)
    else:
        print("No data available at the moment")
    time.sleep(1)  

# Close the serial connection after use
serial_object.close()

print("Data has been successfully Readen")

if __name__=='__main__':
    Image_path=r'C:\Users\etram\Downloads\Uni\3.2\mecha\calibration_images\calibration_sample.jpg'
    Calibration(Image_path)
    main()

