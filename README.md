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

Or you can download from official [website](https://luna16.grand-challenge.org/download/)

### 1.1 data overview

The original data from luna16 are consist of below:

+ **subset0.zip to subset9.zip**: 10 zip files which contain all CT images
+ **annotations.csv**: csv file that contains the annotations used as reference standard for the 'nodule detection' track
+ **sampleSubmission.csv**: an example of a submission file in the correct format
+ **candidates_V2.csv**: csv file that contains the candidate locations for the ‘false positive reduction’ track

As you can know ,the positive sample data (annotations.csv)  and the false sample data(candidates_V2.csv) are already annotated .What we need to do is just extracting them from
medical format(sth like CT) to images.There is no need to worry about positive/negative data.

**annotations.csv**

|seriesuid|coordX|coordY|coordZ|diameter_mm|
|---|---|---|---|---|
|1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860|-128.6994211|-175.3192718|-298.3875064|5.651470635|
|1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860|103.7836509|-211.9251487|-227.12125|4.224708481|
|1.3.6.1.4.1.14519.5.2.1.6279.6001.100398138793540579077826395208|69.63901724|-140.9445859|876.3744957|5.786347814|
|1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405|-24.0138242|192.1024053|-391.0812764|8.143261683|

unit of `coordX`,`coordY`,`coordZ`,`diameter_mm` are **mm**
 
 
 **candidates.csv**
 
|seriesuid|coordX|coordY|coordZ|class|
|---|---|---|---|---|
|1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860|68.42|-74.48|-288.7|0|
|1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860|-95.20936148|-91.80940617|-377.4263503|0|
|1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860|-24.76675476|-120.3792939|-273.3615387|0|
|1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860|-63.08|-65.74|-344.24|0|

Value of class column means postive(1) or negative(0).

### 1.2 how to prepare data



## 2 process step

First run `data_prepare.py` to extract cubic(both real nodule and fake ones) from raw CT files. This may take hours and the output of this step is

+ cubic_npy
+ cubic_normalization_npy
+ cubic_normalization_test

the total size of those file is around  100GB and take one night in my PC(16GB RAM,i5),please leave enough disk. There will be some ValueError like:

```
<class 'Exception'> : could not broadcast input array from shape (40,40,25) into shape (40,40,26)
  File "H:/workspace/luna16_multi_size_3dcnn/data_prepare.py", line 142, in extract_fake_cubic_from_mhd
    int(v_center[2] - 13):int(v_center[2] + 13)]
ValueError: could not broadcast input array from shape (40,40,25) into shape (40,40,26)
Traceback (most recent call last):
```
It's ok to go cause not all false positive candidates are need,reading the csv files and you'll know false positive data are much more than positive data.

Then run `main.py` to train model,inference step will be ran as follow,this step is rather slow cause of huge number of data.


