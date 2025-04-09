import os
import ntpath
import SimpleITK
import numpy as np
import pandas as pd
import cv2

from data.dataclass.CTData import CTData
from util.seg_util import normalize_hu_values,get_segmented_lungs
from util import image_util
from constant import tianchi

TARGET_VOXEL_MM = 1.0
MHD_INFO_HEAD = "patient_id,shape_0,shape_1,shape_2,origin_x,origin_y,origin_z,direction_z(1_-1)," \
                "spacing_x,spacing_y,spacing_z,rescale_x,rescale_y,rescale_z"

def get_all_mhd_file(BASE_DATA_DIR,base_head,max):
    """
       get all mhd file list ,tianchi mhd file consist of train_subset00,train_subset01,... test_subset00,test_subset01,..

    :param base_head:       'train' or 'test',or 'val', to construct train_subset00,test_subset01,val_subset02...
    :param max:             the max suffix of path ,such as train_subset09, then max=09
    :return:                all mhd file list
    """
    mhd_files = []
    for index in range(0,max):
        if index<10:
            index = "0"+str(index)
        else:
            index =str(index)
        sub_path = os.path.join(BASE_DATA_DIR,base_head+"_subset"+index)
        for name in os.listdir(sub_path):
            if name.endswith(".mhd"):
                mhd_files.append(os.path.join(sub_path,name))
    return mhd_files


def get_luna16_mhd_file(mhd_root):
    """
       get all mhd file list ,tianchi mhd file consist of train_subset00,train_subset01,... test_subset00,test_subset01,..

    :param mhd_root:       'train' or 'test',or 'val', to construct train_subset00,test_subset01,val_subset02...
    :return:                all mhd file list
    """
    mhd_files = []
    for root, _, files in os.walk(mhd_root):
        for file in files:
            if file.lower().endswith('.mhd'):
                real_file = os.path.join(mhd_root, root, file)
                mhd_files.append(real_file)
    return mhd_files
def read_csv_to_pandas(mhd_info,col_sepator ='\t'):
    """
       read csv information into pandas dataframe

    :param mhd_info:      csv file of mhd file
    :param col_sepator:  sepator string of columns
    :return:
    """
    with open(mhd_info, 'r') as csv:
        head = csv.readline().split(",")  # get header of csv
        indexs = []
        lines = csv.readlines()
        list = []
        for line in lines:
            list.append(line.split(col_sepator))
            indexs.append(line.split(col_sepator)[0])  #the first element should be id of patient
        df = pd.DataFrame(data=list, columns=head,index=indexs)
        return df

def extract_image_from_mhd(mhd_file_path,png_save_path_root =None):
    """
        extract image from mhd file and return mhd information

    :param mhd_file_path:       mhd file to extract
    :param png_save_path_root:  file path where to save the extracted image (both image and mask image will be saved)
                                ,if this param is None means only mhd information returns,no image extracted
    :return:
    """
    mhd_info = []
    patient_id = ntpath.basename(mhd_file_path).replace(".mhd", "")
    print("Patient: ", patient_id)
    mhd_info.append(patient_id)
    if not os.path.exists(png_save_path_root):
        os.mkdir(png_save_path_root)
    dst_dir = png_save_path_root+'/' + patient_id + "/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(mhd_file_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)
    (shape0,shape1,shape2) = img_array.shape
    mhd_info.append(str(shape2))
    mhd_info.append(str(shape1))
    mhd_info.append(str(shape0))

    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)
    mhd_info.append(str(origin[0]))
    mhd_info.append(str(origin[1]))
    mhd_info.append(str(origin[2]))

    direction = np.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)
    direct_arow = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    if direction.tolist() == direct_arow:
        print("positive direction..")
        mhd_info.append(str(1))
    else:
        mhd_info.append(str(-1))

    spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    mhd_info.append(str(spacing[0]))
    mhd_info.append(str(spacing[1]))
    mhd_info.append(str(spacing[2]))

    rescale = spacing /TARGET_VOXEL_MM
    print("Rescale: ", rescale)
    mhd_info.append(str(rescale[0]))
    mhd_info.append(str(rescale[1]))
    mhd_info.append(str(rescale[2]))

    if png_save_path_root is None:              # get mhd information only
        return mhd_info

    if not os.path.exists(dst_dir):
        if img_array.shape[1]== 512:
            img_array = image_util.rescale_patient_images(img_array, spacing, TARGET_VOXEL_MM)
        img_list = []
        for i in range(img_array.shape[0]):
            img = img_array[i]
            seg_img, mask = get_segmented_lungs(img.copy())
            img_list.append(seg_img)
            img = normalize_hu_values(img)
            cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_i.png", img * 255)
            cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_m.png", mask * 255)
    return mhd_info

