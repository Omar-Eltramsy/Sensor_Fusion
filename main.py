#!/usr/bin/env python
import numpy as np
import Camera_Calibration 
import convert3D_2D 
import Camera_Feeder
if __name__=='__main__':
    Image_path=r'C:\Users\etram\Downloads\Uni\3.2\mecha\calibration_images\calibration_sample.jpg'
    Camera_Calibration.Calibration(Image_path)
    Camera_Feeder.main()

