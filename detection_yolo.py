from ultralytics import YOLO
import cv2 as cv
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
result=model(source='0',show=True,conf=0.4,save=True)
