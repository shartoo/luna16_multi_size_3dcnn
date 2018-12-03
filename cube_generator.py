
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import traceback
import random
from PIL import Image
from data_prepare import truncate_hu,normalazation


def compute_iou(x,y,z,r,real_x,real_y,real_z,real_r):
    '''
            compute iou between slided cube with real nodule cube
    :param x:           x coordinate of slide cube(generated)
    :param y:           y coordinate of slide cube(generated)
    :param z:           z coordinate of slide cube(generated)
    :param r:           diameter coordinate of slide cube(generated)
    :param real_x:      x coordinate of real nodule (annotated)
    :param real_y:      y coordinate of real nodule (annotated)
    :param real_z:      z coordinate of real nodule(annotated)
    :param real_r:      diameter of real nodule(annotated)
    :return:
    '''
    score = 0.01

    return score
def generate_positive_cubes(mini_df,img_file,plot_output_path,normalization_output_path,cube_x,cube_y,cube_z,threshold=0.75):
    '''
        generate positive cubes  by selection IOU with true cube greater than threshold value

    :param mini_df:         data frame contain annotation information
    :param img_file:        image path
    :param cube_x:          shape of current cube
    :param cube_y:
    :param cube_z:
    :param threshold:       a threshould value
    :return:
    '''
    file_name = os.path.basename(img_file)
    if mini_df.shape[0]>0: # some files may not have a nodule--skipping those
        # load the data once
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        # go through all nodes
        print("begin to process real nodules...")
        img_array = img_array.transpose(2,1,0)      # take care on the sequence of axis of v_center ,transfer to x,y,z
        x_size,y_size,z_size =img_array.shape
        # sliding a whole CT image space
        for x in range(0,x_size-cube_x,cube_x):
            for y in range(0,y_size-cube_y,cube_y):
                for z in range(0,z_size-cube_z,cube_z):
                    # real nodule coordinates
                    for node_idx, cur_row in mini_df.iterrows():
                        node_x = cur_row["coordX"]
                        node_y = cur_row["coordY"]
                        node_z = cur_row["coordZ"]
                        diameter = cur_row["diameter"]
                        center = np.array([node_x, node_y, node_z])  # nodule center
                        v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
                        node_x, node_y, node_z = v_center[0],v_center[1],v_center[2]
                        diameter = diameter/spacing
                        score = compute_iou(x,y,z,node_x,node_y,node_z,diameter)
                        # find a positive cube to extract
                        if score>threshold:
                            cube = img_array[x:x + cube_x, y:y + cube_y, z:z + cube_z]
                            nodule_pos_str = str(node_x) + "_" + str(node_y) + "_" + str(node_z)
                            np.save(os.path.join(plot_output_path, "images_%s_%d_pos%s_size%dx%d.npy" % (
                            str(file_name), node_idx, nodule_pos_str,cube_x,cube_y)), cube)
                            truncate_hu(cube)
                            cube = normalazation(cube)
                            np.save(os.path.join(normalization_output_path, "%d_real_size%dx%d.npy" % (node_idx,cube_x,cube_y)), cube)
