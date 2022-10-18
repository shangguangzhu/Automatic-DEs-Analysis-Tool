@[TOC](Automatic DEs Analysis Tool (ADAT))

# The introduction of ADAT
ADAT is an automatic double emulsion droplets (DEs) analysis tool which is based on [YOLOv5](https://github.com/ultralytics/yolov5)[^1] and [Deep Sort](https://github.com/ZQPei/deep_sort_pytorch)[^2] . It can automatically track and give the data needed for systematic analysis of the DEs. These data include the duration of the DEs steady state,  the variation of the DEs size, number and position with time. 
[^1]:
[^2]:
# The guide to use ADAT
## 1. Install
Clone repo and install requirements.txt in a Python>=3.7.0 environment, including PyTorch>=1.7.
```bash
git clone https://github.com/shangguangzhu/Automatic-DEs-Analysis-Tool  # clone
pip install -r requirements.txt  # install
```
## 2. Build dataset
As an example, a COCO dataset containing four classes: bubble, single emulsion droplet, double emulsion droplet(one core) and double emulsion droplet(multi cores)  can be get from [here](). You can also make your own COCO dataset with [LableImg](). 

## 3. Training on the YOLOv5 model

```bash
python3 dataset.py --source ... --show-vid  # show live inference results as well
```

## 4. Tracking of the DEs
 
## 5. Tutorials
* [Yolov5 training on Custom Data (link to external repository)](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp;
* [Deep Sort deep descriptor training (link to external repository)](https://github.com/ZQPei/deep_sort_pytorch#training-the-re-id-model)&nbsp;
* [Yolov5 deep_sort pytorch evaluation](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/wiki/Evaluation)&nbsp;