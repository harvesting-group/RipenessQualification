import sys
sys.path.append('/media/strawberry/HP P500/paper-1/v8_ripe_baseline/ultralytics-8.0.221')
import os
from ultralytics import YOLO
import time  
from retinexnet.retinex_pred import RetinexNet
import torch
from copy import deepcopy
import matplotlib.pyplot as plt  
import cv2
from thop import profile  
import torch.nn as nn

class RetinexNet_v8_Ripe(nn.Module):
    def __init__(self):  
        super(RetinexNet_v8_Ripe, self).__init__()  
        self.retinexnet =  RetinexNet()
        self.v8_ripe = YOLO("runs/ripe/train5/weights/best.pt")
        self.fc1 = nn.Linear(16 * 6 * 6, 10)   

    def forward(self, x): 
        new_img = self.retinexnet.predict() 
        result = self.v8_ripe.predict(source=new_img)
        return result 

 
if __name__ == '__main__':
    # imgs_path = '/media/strawberry/HP P500/paper-1/v8_ripe_baseline/ultralytics/datasets/LightData-v9/images/val' 
    imgs_path = '/media/strawberry/HP P500/paper-1/v8_ripe_baseline/ultralytics/datasets/LightData-v9/images/val'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    
    model = YOLO("runs/ripe/train5/weights/best.pt").to(device)
    images = os.listdir(imgs_path)
    retinexnet = RetinexNet().cuda()

    # # 计算FPS
    for i in range(10):
        image_name = imgs_path + os.sep + images[i]
        new_path = '/media/strawberry/HP P500/paper-1/v8_ripe_baseline/ours_fps/predict/'
        
        im = retinexnet.predict(test_low_data_names=image_name, res_dir=new_path, ckpt_dir='ours_fps/ckpts/')

        results = model.predict(source = im, save_txt=False)


    start_time = time.time()  
    for item in images:
        # print(item)
        image_name = imgs_path + os.sep + item
        new_path = '/media/strawberry/HP P500/paper-1/v8_ripe_baseline/ours_fps/predict'
        
        im = retinexnet.predict(test_low_data_names=image_name, res_dir=new_path, ckpt_dir='ours_fps/ckpts/')
        
        results = model.predict(source = im, save_txt = False)
        results[0].path=item

    end_time = time.time()  
    elapsed_time = end_time - start_time 
    print('elapsed_time:', elapsed_time) 
    
    # 计算 FPS 
    fps = len(images) / elapsed_time  
    print(f"处理 {len(images)} 帧所耗时间: {elapsed_time:.2f} 秒")  
    print(f"FPS: {fps:.2f}")

    #计算参数量
    total_params = sum(p.numel() for p in model.parameters())+sum(p.numel() for p in retinexnet.parameters())  
    print(f"Total Parameters: {total_params/ 1_000_000 }") 