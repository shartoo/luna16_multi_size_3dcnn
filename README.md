# luna16_multi_size_3dcnn
An implement of paper "Multi-level Contextual 3D CNNs for False Positive Reduction in Pulmonary Nodule Detection"

The detail about the paper can be found [luna16 3DCNN](http://shartoo.github.io/LUNA2016-3DCNN/)

## 0 required

+ numpy

+ PIL(or Pillow)

+ numpy

+ SimpleITK

+ pandas

+ matplotlib

+ Tensorflow >1.3

## 1 data

You can download LUNA16 dataset from [BaiduCloudDisk](链接: https://pan.baidu.com/s/1vUJtAshK51Zi5rFhqG2t7g 提取码: y929),there are laso some torrent file to download with other tools.



## 2 process step

First run `data_prepare.py` to extract cubic(both real nodule and fake ones) from raw CT files. This may take hours and the output of this step is

+ cubic_npy
+ cubic_normalization_npy
+ cubic_normalization_test

the total size of those file is around  100GB and take one night in my PC(16GB RAM,i5),please leave enough disk.

Then run `main.py` to train model,inference step will be ran as follow,this step is rather slow cause of huge number of data.


