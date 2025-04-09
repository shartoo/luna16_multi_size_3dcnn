import os
import glob
import pydicom
import numpy as np
import cv2
from tqdm import tqdm

from util.seg_util import get_segmented_lungs,normalize_hu_values
from util.image_util import rescale_patient_images


def is_dicom_file(filename):
    '''
       if current file is a dicom file
    :param filename:      file need to be judged
    :return:
    '''
    file_stream = open(filename, 'rb')
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b'DICM':
        return True
    return False

def get_dicom_thickness(dicom_slices):
    """
        计算切片厚度
    :param dicom_slices:    dicom 读取的 dicom数据
    :return:
    """
    if len(dicom_slices) > 1:
        try:
            slice_thickness = abs(dicom_slices[0].ImagePositionPatient[2] - dicom_slices[1].ImagePositionPatient[2])
        except:
            try:
                slice_thickness = abs(dicom_slices[0].SliceLocation - dicom_slices[1].SliceLocation)
            except:
                # 如果无法计算，尝试从SliceThickness标签中获取
                try:
                    slice_thickness = float(dicom_slices[0].SliceThickness)
                except:
                    print("警告: 无法确定切片厚度，使用默认值1.0mm")
                    slice_thickness = 1.0
    else:
        try:
            slice_thickness = float(dicom_slices[0].SliceThickness)
        except:
            print("警告: 只有一个切片，无法计算切片厚度，使用默认值1.0mm")
            slice_thickness = 1.0
    return slice_thickness

def load_dicom_slices(dicom_path):
    """
        load dicom file path and stack into list

    :param dicom_path:     a dicom path
    :return:            dicom list
    """
    dicom_files = []
    for root, _, files in os.walk(dicom_path):
        for file in files:
            if file.lower().endswith(('.dcm', '.dicom')):
                real_file = os.path.join(dicom_path, root, file)
                current_if_dicom = is_dicom_file(real_file)
                if current_if_dicom:
                    dicom_files.append(real_file)
    if not dicom_files:
        raise ValueError(f"在路径 {dicom_path} 中未找到DICOM文件")
    # 加载所有切片
    slices = []
    for file in dicom_files:
        try:
            ds = pydicom.dcmread(file)
            slices.append(ds)
        except Exception as e:
            print(f"无法读取DICOM文件 {file}: {e}")
    # # 按照Z轴位置排序切片
    slices.sort(key=lambda x: int(x.InstanceNumber))
    slice_thickness = get_dicom_thickness(slices)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(slices):
    '''
        transfer dicom array to pixel array,and remove border(HU==-2000)

    :param slices:  dicom list
    :return:        pixel array of one patient's dicom
    '''
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def getinfo_dicom(dicom_path):
    print('dicom_path: ', dicom_path)
    slices = load_dicom_slices(dicom_path)
    print(type(slices[0]), slices[0].ImagePositionPatient)
    print(len(slices), "\t", slices[0].SliceThickness, "\t", slices[0].PixelSpacing)
    print("Orientation: ", slices[0].ImageOrientationPatient)
    #assert slices[0].ImageOrientationPatient == [1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]
    pixels = get_pixels_hu(slices)
    image = pixels
    print(image.shape)

    invert_order = slices[1].ImagePositionPatient[2] > slices[0].ImagePositionPatient[2]
    print("Invert order: ", invert_order, " - ", slices[1].ImagePositionPatient[2], ",",
          slices[0].ImagePositionPatient[2])

    pixel_spacing = slices[0].PixelSpacing
    pixel_spacing.append(slices[0].SliceThickness)
    # save dicom source image size
    dicom_size = [image.shape[0], image.shape[1], image.shape[2]]

    return pixel_spacing, dicom_size, invert_order

def extract_dicom_images_patient(dicom_path, target_dir):
    slices = load_dicom_slices(dicom_path)
    assert slices[0].ImageOrientationPatient == [1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]
    pixels = get_pixels_hu(slices)
    image = pixels
    invert_order = slices[1].ImagePositionPatient[2] > slices[0].ImagePositionPatient[2]
    pixel_spacing = slices[0].PixelSpacing
    pixel_spacing.append(slices[0].SliceThickness)
    # save dicom source image size
    dicom_size = [image.shape[0], image.shape[1], image.shape[2]]
    image = rescale_patient_images(image, pixel_spacing)
    png_size = [image.shape[0], image.shape[1], image.shape[2]]
    if not invert_order:
        image = np.flipud(image)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    else:
        print("png dir already exists, return directly")
        return pixel_spacing, dicom_size, png_size, invert_order
    png_files = glob.glob(target_dir + "*.png")
    for file in png_files:
        os.remove(file)
    for i in tqdm(range(image.shape[0])):
        img_path = patient_dir + "/img_" + str(i).rjust(4, '0') + "_i.png"
        org_img = image[i]
        img, mask = get_segmented_lungs(org_img.copy())
        org_img = normalize_hu_values(org_img)
        cv2.imwrite(img_path, org_img * 255)
        cv2.imwrite(img_path.replace("_i.png", "_m.png"), mask * 255)
    return pixel_spacing, dicom_size, png_size,invert_order

