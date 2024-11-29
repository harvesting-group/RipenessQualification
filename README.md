# RCE-Net and RCE-Net w RetinexNet  

RCE-Net and RCE-Net w RetinexNet are based on YOLOv8(https://github.com/ultralytics/ultralytics) and RetinexNet(https://github.com/aasharma90/RetinexNet_PyTorch). RCE-Net is an end-to-end detection, segmentation, and ripeness network based on the yolov8. RetinexNet is introduced into RCE-Net to alleviate the effects of various light intensities. 

### Dataset
[LightStrawberry](https://1drv.ms/f/s!AiiOmEgnYIvmj_8D1ucFNEEh10En0A?e=Cqr7Tx "LightStrawberry") 

## Download
```python
git clone https://github.com/harvesting-group/RCE-Net-w-RetinexNet.git
```
## Envrionment 
```python
conda create -n rcenet python=3.9 -y
conda activate rcenet
pip install ultralytics
```
## Demo
### Training
```python  
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
sys.path.append('./ultralytics-8.0.221') 

from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/v8/yolov8n-ripe.yaml")  # build a new model from scratch
model.train(data="ultralytics/datasets/LightStrawberry.yaml", epochs=1000, device='0')  # train the model 7.5 MSE
metrics = model.val()  # evaluate model performance on the validation set
```
### Validation
```python
import sys
sys.path.append('/media/strawberry/HP P500/paper-1/v8_ripe_baseline/ultralytics-8.0.221') 
from ultralytics import YOLO

model = YOLO('RCE-Net.pt')
source = './LightStrawberry/images/val'
results = model(source, mode='predict', save=True,show_labels=True,save_txt = True)  # list of Results objects 
```
### Inference

```python
python rcenet_w_retinexnet_inference.py
```


