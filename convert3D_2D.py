import cv2 as cv
import numpy as np

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

