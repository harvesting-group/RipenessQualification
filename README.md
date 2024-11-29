RCE-Net and RCE-Net w RetinexNet

RCE-Net and RCE-Net w RetinexNet are based on YOLOv8(https://github.com/ultralytics/ultralytics) and RetinexNet(https://github.com/aasharma90/RetinexNet_PyTorch). RCE-Net is an end-to-end detection, segmentation, and ripeness network based on the yolov8.
RetinexNet is introduced into RCE-Net to alleviate the effects of various light intensities. 

Install
git clone https://github.com/ultralytics/ultralytics.git

Training
Demo
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
sys.path.append('./ultralytics-8.0.221') # 自己的文件夹

from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/v8/yolov8n-ripe.yaml")  # build a new model from scratch

model.train(data="ultralytics/datasets/LightStrawberry.yaml", epochs=1000, device='0')  # train the model 7.5 MSE
metrics = model.val()  # evaluate model performance on the validation set






