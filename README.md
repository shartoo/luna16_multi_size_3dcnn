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

## 1 process step

First run `data_prepare.py` to extract cubic(both real nodule and fake ones) from raw CT files. This may take hours and the output of this step is

+ cubic_npy
+ cubic_normalization_npy
+ cubic_normalization_test

the total size of those file is around  100GB,please leave enough disk.

Then run `main.py` to train model,inference step will be ran as follow,this step is rather slow cause of huge number of data.


