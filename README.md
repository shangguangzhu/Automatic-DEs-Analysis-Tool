@[TOC](Automatic DEs Analysis Tool (ADAT))

# The introduction of ADAT
ADAT is an automatic double emulsion droplets (DEs) analysis tool which is based on [YOLOv5](https://github.com/ultralytics/yolov5)and [Deep Sort](https://github.com/ZQPei/deep_sort_pytorch). It can automatically track and give the data needed for systematic analysis of the DEs. These data include the duration of the DEs steady state,  the variation of the DEs size, number and position with time. 
# The guide to use ADAT
## 1. Install
Clone repo and install **requirements.txt** in a Python>=3.7.0 environment, including PyTorch>=1.7.
```bash
git clone https://github.com/shangguangzhu/Automatic-DEs-Analysis-Tool  # clone
pip install -r requirements.txt  # install
```
## 2. Build dataset
As an example, a COCO dataset containing four classes: bubble, single emulsion droplet, double emulsion droplet(one core) and double emulsion droplet(multi cores) which used in the paper can be downloaded [here](). 
You can also make your own COCO dataset with [LablelImg](https://github.com/heartexlabs/labelImg). 

## 3. Training on the YOLOv5 model
**DATASET.py** is used to divide the COCO dataset into training set and validation set in a desired proportion. 
After that, use **yolov5\train.py** to train the YOLO model and there are five pretrained models can be chose.
![](https://img-blog.csdnimg.cn/img_convert/4802dde1e4f5f7b40ea381b559798e0f.png)
## 4. Analysis of the DEs
 **DATASET.py** is used to calculate  the duration of the DEs steady state,  the variation of the DEs size, number and position with time. 
## 5. Tutorials
* [The introduction of LabelImg (link to external repository)](https://github.com/heartexlabs/labelImg/blob/master/README.rst)&nbsp;
* [Yolov5 training on Custom Data (link to external repository)](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp;
*  [Tips for Best Training Results (link to external repository)](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)&nbsp;
* [Deep Sort deep descriptor training (link to external repository)](https://github.com/ZQPei/deep_sort_pytorch#training-the-re-id-model)&nbsp;

## 6.Cite
If you find this project useful in your research, please consider cite:
```latex
@misc{Automatic DEs Analysis Tool,
    title={Automatic DEs Analysis Tool},
    author={Guangzhu Shang et.al},
    howpublished = {\url{https://github.com/shangguangzhu/Automatic-DEs-Analysis-Tool}},
    year={2022}
}
```
