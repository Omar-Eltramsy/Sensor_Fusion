# Sensor_Fusion
This project performs camera calibration using a chessboard pattern and real-time object detection using YOLOv8.<br>
It reads distance and angle measurements from a serial device (like Arduino or a sensor), projects 3D world points onto the 2D camera image, and displays them alongside detected objects.<br>
## Features
* Camera Calibration using OpenCV
* 3D to 2D Projection of real-world coordinates
* Real-time Object Detection with YOLOv8
* Serial Communication to receive external sensor data
* Data logging from the serial port
## Requirements
* Python 3.8+
* OpenCV
* NumPy
* PySerial
* Ultralytics (for YOLOv8)

You can install all dependencies via:
```
pip install opencv-python numpy pyserial ultralytics 
```


