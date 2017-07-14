# -*- coding:utf-8 -*-
'''
this is the enterance of this project
'''

import tensorflow as tf
import os
from model import model
import numpy as np

if __name__ =='__main__':
    batch_size =32
    learning_rate = 0.01
    keep_prob =0.7
    path = '/data/LUNA2016/cubic_normalization_npy'
    test_path = '/data/LUNA2016/cubic_normalization_test'

    print(" beigin...")
    model = model(learning_rate,keep_prob,batch_size,40)
    model.inference(path,test_path,0,True)







