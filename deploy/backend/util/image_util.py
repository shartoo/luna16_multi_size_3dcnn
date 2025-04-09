from typing import Tuple

import cv2
import os
import numpy
import glob
import random
import numpy as np
from scipy import ndimage

def get_normalized_img_unit8(img):
    img = img.astype(numpy.float)
    min = img.min()
    max = img.max()
    img -= min
    img /= max - min
    img *= 255
    res = img.astype(numpy.uint8)
    return res


def load_patient_images(png_path, wildcard="*.*", exclude_wildcards=[]):
    print("png path is\t",png_path)
    src_dir = png_path
    src_img_paths = glob.glob(src_dir +'/'+ wildcard)
    for exclude_wildcard in exclude_wildcards:
        exclude_img_paths = glob.glob(src_dir + exclude_wildcard)
        src_img_paths = [im for im in src_img_paths if im not in exclude_img_paths]
    src_img_paths.sort()

    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in src_img_paths]
    images = [im.reshape((1, ) + im.shape) for im in images]
    res = numpy.vstack(images)
    return res


def draw_overlay(png_path: str, p_x: float, p_y: float, p_z: float, index: str,  BOX_size:int = 20) -> None:
    """
    在图像上绘制覆盖层
    Args:
        png_path: PNG图像路径
        p_x: X坐标（百分比）
        p_y: Y坐标（百分比）
        p_z: Z坐标（百分比）
        index: 索引标识
        :param BOX_size:
    """
    patient_img = load_patient_images(png_path + "/png/", "*_i.png", [])
    z = int(p_z * patient_img.shape[0])
    y = int(p_y * patient_img.shape[1])
    x = int(p_x * patient_img.shape[2])
     # 包围盒大小
    x1 = x - BOX_size
    y1 = y - BOX_size
    x2 = x + BOX_size
    y2 = y + BOX_size
    target_img = patient_img[z, :, :]
    cv2.rectangle(target_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv2.imwrite(png_path + "/" + index + ".png", target_img)

def prepare_image_for_net3D(img,MEAN_PIXEL_VALUE = 41):
    '''
        normalization of image (average and zero center)

    :param img:               image to be normalization
    :param MEAN_PIXEL_VALUE:
    :return:
    '''
    img = img.astype(numpy.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img


def move_png2dir(target_dir):
    import shutil
    first_dir = []
    for path in os.listdir(target_dir):
        if os.path.isdir(os.path.join(target_dir,path)):
            first_dir.append(os.path.join(target_dir,path))
    for d in first_dir:
        tmp_path = []
        for file in os.listdir(d):
            tmp_file_path = os.path.join(d,file)
            png_path = os.path.join(d,'png')
            if not os.path.exists(png_path):
                os.mkdir(png_path)
            if tmp_file_path.endswith(".png"):
                shutil.move(tmp_file_path,os.path.join(png_path,file))
                print("move file from %s  to   %s " %(tmp_file_path,os.path.join(png_path,file)))

def rescale_patient_images(images_zyx, org_spacing_xyz, target_voxel_mm =1.0, is_mask_image=False, verbose=False):
    '''
                rescale a 3D image to specified size

    :param images_zyx:              source image
    :param org_spacing_xyz:
    :param target_voxel_mm:
    :param is_mask_image:
    :param verbose:
    :return:
    '''
    if verbose:
        print("Spacing: ", org_spacing_xyz)
        print("Shape: ", images_zyx.shape)

    # print "Resizing dim z"
    resize_x = 1.0
    resize_y = float(org_spacing_xyz[2]) / float(target_voxel_mm)
    interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
    res = cv2.resize(images_zyx, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff
    res = res.swapaxes(0, 2)
    res = res.swapaxes(0, 1)
    # print "Shape: ", res.shape
    resize_x = float(org_spacing_xyz[0]) / float(target_voxel_mm)
    resize_y = float(org_spacing_xyz[1]) / float(target_voxel_mm)

    # cv2 can handle max 512 channels..
    if res.shape[2] > 512:
        res = res.swapaxes(0, 2)
        res1 = res[:256]
        res2 = res[256:]
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res = numpy.vstack([res1, res2])
        res = res.swapaxes(0, 2)
    else:
        res = cv2.resize(res, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
    res = res.swapaxes(0, 2)
    res = res.swapaxes(2, 1)
    if verbose:
        print("Shape after: ", res.shape)
    return res


def rescale_patient_images2(images_zyx, target_shape, verbose=False):
    if verbose:
        print("Target: ", target_shape)
        print("Shape: ", images_zyx.shape)

    # print "Resizing dim z"
    resize_x = 1.0
    interpolation = cv2.INTER_NEAREST if False else cv2.INTER_LINEAR
    res = cv2.resize(images_zyx, dsize=(target_shape[1], target_shape[0]), interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff
    # print "Shape is now : ", res.shape

    res = res.swapaxes(0, 2)
    res = res.swapaxes(0, 1)

    # cv2 can handle max 512 channels..
    if res.shape[2] > 512:
        res = res.swapaxes(0, 2)
        res1 = res[:256]
        res2 = res[256:]
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=(target_shape[2], target_shape[1]), interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=(target_shape[2], target_shape[1]), interpolation=interpolation)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res = numpy.vstack([res1, res2])
        res = res.swapaxes(0, 2)
    else:
        res = cv2.resize(res, dsize=(target_shape[2], target_shape[1]), interpolation=interpolation)

    res = res.swapaxes(0, 2)
    res = res.swapaxes(2, 1)
    if verbose:
        print("Shape after: ", res.shape)
    return res


def resize_image(image: np.ndarray, new_shape: Tuple[int, ...]) -> np.ndarray:
    """
    调整图像大小

    Args:
        image: 输入图像
        new_shape: 新形状

    Returns:
        np.ndarray: 调整大小后的图像
    """
    # 处理单通道或多通道图像
    if len(image.shape) == 3 and len(new_shape) == 2:
        # 处理3D图像调整为2D
        resized_image = np.zeros((image.shape[0], new_shape[0], new_shape[1]))
        for i in range(image.shape[0]):
            resized_image[i] = cv2.resize(image[i], (new_shape[1], new_shape[0]))
        return resized_image
    elif len(image.shape) == 2 and len(new_shape) == 2:
        # 处理2D图像
        return cv2.resize(image, (new_shape[1], new_shape[0]))
    else:
        # 处理任意维度图像
        resize_factor = tuple(n / o for n, o in zip(new_shape, image.shape))
        return ndimage.zoom(image, resize_factor, mode='nearest')

def cv_flip(img,cols,rows,degree):
    '''
        flip image by degree

    :param img:         image array to be fliped
    :param cols:        width of image
    :param rows:        height of image
    :param degree:      degree to flip
    :return:
    '''
    M = cv2.getRotationMatrix2D((cols / 2, rows /2), degree, 1.0)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def random_rotate_img(img, chance, min_angle, max_angle):
    '''
        random rotation an image

    :param img:         image to be rotated
    :param chance:      random probability
    :param min_angle:   min angle to rotate
    :param max_angle:   max angle to rotate
    :return:            image after random rotated
    '''
    import cv2
    if random.random() > chance:
        return img
    if not isinstance(img, list):
        img = [img]

    angle = random.randint(min_angle, max_angle)
    center = (img[0].shape[0] / 2, img[0].shape[1] / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    res = []
    for img_inst in img:
        img_inst = cv2.warpAffine(img_inst, rot_matrix, dsize=img_inst.shape[:2], borderMode=cv2.BORDER_CONSTANT)
        res.append(img_inst)
    if len(res) == 0:
        res = res[0]
    return res


def random_flip_img(img, horizontal_chance=0, vertical_chance=0):
    '''
        random flip image,both on horizontal and vertical

    :param img:                 image to be flipped
    :param horizontal_chance:   flip probability to flipped on horizontal direction
    :param vertical_chance:     flip probability to flipped on vertical  direction
    :return:                    image after flipped
    '''
    import cv2
    flip_horizontal = False
    if random.random() < horizontal_chance:
        flip_horizontal = True

    flip_vertical = False
    if random.random() < vertical_chance:
        flip_vertical = True

    if not flip_horizontal and not flip_vertical:
        return img

    flip_val = 1
    if flip_vertical:
        flip_val = -1 if flip_horizontal else 0

    if not isinstance(img, list):
        res = cv2.flip(img, flip_val)  # 0 = X axis, 1 = Y axis,  -1 = both
    else:
        res = []
        for img_item in img:
            img_flip = cv2.flip(img_item, flip_val)
            res.append(img_flip)
    return res


def random_scale_img(img, xy_range, lock_xy=False):
    if random.random() > xy_range.chance:
        return img

    if not isinstance(img, list):
        img = [img]

    import cv2
    scale_x = random.uniform(xy_range.x_min, xy_range.x_max)
    scale_y = random.uniform(xy_range.y_min, xy_range.y_max)
    if lock_xy:
        scale_y = scale_x

    org_height, org_width = img[0].shape[:2]
    xy_range.last_x = scale_x
    xy_range.last_y = scale_y

    res = []
    for img_inst in img:
        scaled_width = int(org_width * scale_x)
        scaled_height = int(org_height * scale_y)
        scaled_img = cv2.resize(img_inst, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)
        if scaled_width < org_width:
            extend_left = (org_width - scaled_width) / 2
            extend_right = org_width - extend_left - scaled_width
            scaled_img = cv2.copyMakeBorder(scaled_img, 0, 0, extend_left, extend_right, borderType=cv2.BORDER_CONSTANT)
            scaled_width = org_width

        if scaled_height < org_height:
            extend_top = (org_height - scaled_height) / 2
            extend_bottom = org_height - extend_top - scaled_height
            scaled_img = cv2.copyMakeBorder(scaled_img, extend_top, extend_bottom, 0, 0, borderType=cv2.BORDER_CONSTANT)
            scaled_height = org_height

        start_x = (scaled_width - org_width) / 2
        start_y = (scaled_height - org_height) / 2
        tmp = scaled_img[start_y: start_y + org_height, start_x: start_x + org_width]
        res.append(tmp)

    return res


class XYRange:
    def __init__(self, x_min, x_max, y_min, y_max, chance=1.0):
        self.chance = chance
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.last_x = 0
        self.last_y = 0

    def get_last_xy_txt(self):
        res = "x_" + str(int(self.last_x * 100)).replace("-", "m") + "-" + "y_" + str(int(self.last_y * 100)).replace(
            "-", "m")
        return res


def random_translate_img(img, xy_range, border_mode="constant"):
    if random.random() > xy_range.chance:
        return img
    import cv2
    if not isinstance(img, list):
        img = [img]

    org_height, org_width = img[0].shape[:2]
    translate_x = random.randint(xy_range.x_min, xy_range.x_max)
    translate_y = random.randint(xy_range.y_min, xy_range.y_max)
    trans_matrix = numpy.float32([[1, 0, translate_x], [0, 1, translate_y]])

    border_const = cv2.BORDER_CONSTANT
    if border_mode == "reflect":
        border_const = cv2.BORDER_REFLECT

    res = []
    for img_inst in img:
        img_inst = cv2.warpAffine(img_inst, trans_matrix, (org_width, org_height), borderMode=border_const)
        res.append(img_inst)
    if len(res) == 1:
        res = res[0]
    xy_range.last_x = translate_x
    xy_range.last_y = translate_y
    return res


def data_augmentation(image: np.ndarray, augment_type: str = 'random') -> np.ndarray:
    """
    对图像进行数据增强

    Args:
        image: 输入图像
        augment_type: 增强类型，可选'random', 'flip', 'rotate', 'shift'

    Returns:
        np.ndarray: 增强后的图像
    """
    if augment_type == 'random':
        # 随机选择一种增强方式
        augment_choices = ['flip', 'rotate', 'shift', 'none']
        choice = np.random.choice(augment_choices)

        if choice == 'flip':
            return data_augmentation(image, 'flip')
        elif choice == 'rotate':
            return data_augmentation(image, 'rotate')
        elif choice == 'shift':
            return data_augmentation(image, 'shift')
        else:
            return image

    elif augment_type == 'flip':
        # 随机翻转
        axis = np.random.randint(0, image.ndim)
        return np.flip(image, axis=axis)

    elif augment_type == 'rotate':
        # 随机旋转
        if image.ndim == 2:
            angle = np.random.randint(0, 360)
            return ndimage.rotate(image, angle, reshape=False, mode='nearest')
        else:
            # 3D旋转
            axes = tuple(np.random.choice(range(image.ndim), size=2, replace=False))
            angle = np.random.randint(0, 360)
            return ndimage.rotate(image, angle, axes=axes, reshape=False, mode='nearest')

    elif augment_type == 'shift':
        # 随机平移
        shift = np.random.randint(-5, 6, size=image.ndim)
        return ndimage.shift(image, shift, mode='nearest')

    return image