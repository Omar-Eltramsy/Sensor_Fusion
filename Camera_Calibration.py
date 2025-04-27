import numpy as np
import os
import cv2 as cv

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
