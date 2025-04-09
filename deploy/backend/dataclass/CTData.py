#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from enum import Enum
import matplotlib.pyplot as plt
from util.dicom_util import load_dicom_slices, get_pixels_hu, get_dicom_thickness
from util.seg_util import get_segmented_lungs, normalize_hu_values


class CTFormat(Enum):
    DICOM = 1
    MHD = 2
    UNKNOWN = 3

class CTData:
    """
    统一的CT数据类，用于处理不同格式的CT图像数据
    支持DICOM和MHD格式的加载、处理和分析
    """
    def __init__(self):
        # 基本属性
        self.pixel_data = None  # 像素数据，3D体素数组 (z,y,x)
        self.lung_seg_img = None    # 单独抽取肺部CT图像数据
        self.lung_seg_mask = None   # 肺部CT的掩码
        self.origin = None  # 坐标原点 (x,y,z)，单位为mm
        self.spacing = None  # 体素间距 (x,y,z)，单位为mm
        self.orientation = None  # 方向矩阵
        self.z_axis_flip = False    # z 轴是否是翻转的
        self.size = None  # 图像尺寸 (z,y,x)
        self.data_format = None  # 数据格式(DICOM/MHD)
        self.metadata = {}  # 其他元数据信息
        self.hu_converted = False  # 是否已转换为HU值
        self.preprocessed = False   # 数据是否已经处理过

    @classmethod
    def from_dicom(cls, dicom_path):
        """
        从DICOM文件夹加载CT数据

        Args:
            dicom_path: DICOM文件夹路径

        Returns:
            CTData对象
        """
        ct_data = cls()
        ct_data.data_format = CTFormat.DICOM
        slices = load_dicom_slices(dicom_path)
        ct_data.pixel_data = get_pixels_hu(slices)
        ct_data.z_axis_flip = slices[1].ImagePositionPatient[2] > slices[0].ImagePositionPatient[2]
        ct_data.hu_converted = True
        slice_thickness = get_dicom_thickness(slices)
        # 设置像素间距
        try:
            ct_data.spacing = [
                float(slices[0].PixelSpacing[0]),
                float(slices[0].PixelSpacing[1]),
                float(slice_thickness)
            ]
        except:
            print("警告: 无法获取像素间距，使用默认值[1.0, 1.0, 1.0]")
            ct_data.spacing = [1.0, 1.0, 1.0]
        # 设置原点
        try:
            ct_data.origin = [
                float(slices[0].ImagePositionPatient[0]),
                float(slices[0].ImagePositionPatient[1]),
                float(slices[0].ImagePositionPatient[2])
            ]
        except:
            print("警告: 无法获取坐标原点，使用默认值[0.0, 0.0, 0.0]")
            ct_data.origin = [0.0, 0.0, 0.0]
        # 设置尺寸
        ct_data.size = ct_data.pixel_data.shape
        return ct_data

    @classmethod
    def from_mhd(cls, mhd_path):
        """
        从MHD/RAW文件加载CT数据

        Args:
            mhd_path: MHD文件路径

        Returns:
            CTData对象
        """
        ct_data = cls()
        ct_data.data_format = CTFormat.MHD
        try:
            # 使用SimpleITK加载MHD文件
            itk_img = sitk.ReadImage(mhd_path)
            # 获取像素数据 (注意SimpleITK返回的数组顺序为z,y,x)
            ct_data.pixel_data = sitk.GetArrayFromImage(itk_img)
            # LUNA16的MHD数据已经是HU值
            ct_data.hu_converted = True
            # 获取原点和体素间距
            ct_data.origin = list(itk_img.GetOrigin())  # (x,y,z)
            ct_data.spacing = list(itk_img.GetSpacing())  # (x,y,z)
            # 获取尺寸
            ct_data.size = ct_data.pixel_data.shape
            # 提取方向信息
            ct_data.orientation = itk_img.GetDirection()
            ct_data.z_axis_flip = False
        except Exception as e:
            raise ValueError(f"加载MHD文件时出错: {e}")

        return ct_data

    def convert_to_hu(self):
        """
        将像素值转换为HU值（如果尚未转换）
        """
        if self.hu_converted:
            print("数据已经是HU值格式")
            return

        if self.data_format == CTFormat.DICOM:
            # 已在from_dicom中处理
            self.hu_converted = True
        elif self.data_format == CTFormat.MHD:
            # LUNA16的MHD数据已经是HU值
            self.hu_converted = True
        else:
            raise ValueError("未知数据格式，无法转换为HU值")


    def resample_pixel(self, new_spacing=[1, 1, 1]):
        """
        将CT体素重采样为指定间距

        Args:
            new_spacing: 目标体素间距 [x, y, z]

        Returns:
            重采样后的CTData对象
        """
        # 确保数据已转换为HU值
        if not self.hu_converted:
            self.convert_to_hu()
        # 为了符合scipy.ndimage的要求，将spacing和pixel_data的顺序调整为[z,y,x]
        spacing_zyx = [self.spacing[2], self.spacing[1], self.spacing[0]]
        new_spacing_zyx = [new_spacing[2], new_spacing[1], new_spacing[0]]
        # 计算新尺寸
        resize_factor = np.array(spacing_zyx) / np.array(new_spacing_zyx)
        new_shape = np.round(np.array(self.pixel_data.shape) * resize_factor)
        # 计算实际重采样因子
        real_resize = new_shape / np.array(self.pixel_data.shape)
        # 执行重采样 - 使用三线性插值
        resampled_data = ndimage.zoom(self.pixel_data, real_resize, order=1)
        # 创建新的CTData对象
        resampled_ct = CTData()
        resampled_ct.pixel_data = resampled_data
        resampled_ct.spacing = new_spacing
        resampled_ct.origin = self.origin
        resampled_ct.orientation = self.orientation
        resampled_ct.size = resampled_data.shape
        resampled_ct.data_format = self.data_format
        resampled_ct.hu_converted = self.hu_converted
        resampled_ct.preprocessed = self.preprocessed
        return resampled_ct

    def filter_lung_img_mask(self):
        """
            只保留肺部区域像素，并且归一化到 0-1 之间
        :return:
        """
        pixel_data = self.pixel_data.copy()
        seg_img = []
        seg_mask = []
        for index in range(pixel_data.shape[0]):
            one_seg_img ,one_seg_mask = get_segmented_lungs(pixel_data[index])
            one_seg_img = normalize_hu_values(one_seg_img)
            seg_img.append(one_seg_img)
            seg_mask.append(one_seg_mask)
        self.lung_seg_img = np.array(seg_img)
        self.lung_seg_mask = np.array(seg_mask)
    def world_to_voxel(self, world_coord):
        """
        将世界坐标(mm)转换为体素坐标

        Args:
            world_coord: 世界坐标 [x,y,z] (mm)

        Returns:
            体素坐标 [x,y,z]
        """
        voxel_coord = np.zeros(3, dtype=int)
        for i in range(3):
            voxel_coord[i] = int(round((world_coord[i] - self.origin[i]) / self.spacing[i]))

        return voxel_coord

    def voxel_to_world(self, voxel_coord):
        """
        将体素坐标转换为世界坐标(mm)

        Args:
            voxel_coord: 体素坐标 [x,y,z]

        Returns:
            世界坐标 [x,y,z] (mm)
        """
        world_coord = np.zeros(3, dtype=float)
        for i in range(3):
            world_coord[i] = voxel_coord[i] * self.spacing[i] + self.origin[i]

        return world_coord

    def extract_cube(self, center_world_mm, size_mm,if_fixed_radius = False):
        """
        提取指定中心点和大小的立方体区域

        Args:
            center_world_mm:    立方体中心的世界坐标 [x,y,z] (mm)
            size_mm:            立方体在世界坐标系的大小(mm)，可以是数值或[x,y,z]形式
            if_fixed_radius:    是否为固定半径。默认是False(即不不是固定的，就说明每个结节半径都不一样，按照标注文件半径抽取)

        Returns:
            立方体像素数据
        """
        # 确保数据已加载
        if self.pixel_data is None:
            raise ValueError("未加载数据")
        if self.lung_seg_img is None:
            print("肺部区域数据没有分割，现在开始分割..")
            self.filter_lung_img_mask()
        # 将世界坐标转换为体素坐标（注意：SimpleITK数组顺序为z,y,x）
        center_voxel = self.world_to_voxel(center_world_mm)
        # 交换坐标顺序为z,y,x以匹配pixel_data
        center_voxel_zyx = [center_voxel[2], center_voxel[1], center_voxel[0]]
        # 如果使用固定半径，那么只需要中心坐标即可，此时size_mm 就是像素半径了，直接从 lung_seg_img 按照像素半径抽取即可
        if if_fixed_radius:
            half_size = [int(size_mm/2), int(size_mm/2), int(size_mm/2)]
        else:
            # 计算立方体边长(体素数) [luna2016 的标注数据中每个结节半径不同，按照标注抽取的结节大小不一，最好使用固定半径]
            size_voxel = [int(size_mm / self.spacing[2]),
                          int(size_mm / self.spacing[1]),
                          int(size_mm / self.spacing[0])]
            # 计算立方体边界
            half_size = [s // 2 for s in size_voxel]
        # 提取立方体数据
        z_min = max(0, center_voxel_zyx[0] - half_size[0])
        y_min = max(0, center_voxel_zyx[1] - half_size[1])
        x_min = max(0, center_voxel_zyx[2] - half_size[2])

        z_max = min(self.lung_seg_img.shape[0], center_voxel_zyx[0] + half_size[0])
        y_max = min(self.lung_seg_img.shape[1], center_voxel_zyx[1] + half_size[1])
        x_max = min(self.lung_seg_img.shape[2], center_voxel_zyx[2] + half_size[2])
        # 提取子体积
        cube = self.lung_seg_img[z_min:z_max, y_min:y_max, x_min:x_max]
        return cube

    def visualize_slice(self, slice_idx=None, axis=0, show_lung_only=False):
        """
            可视化单个切片
        Args:
            slice_idx:          切片索引，如果为None则取中心切片
            axis:               沿哪个轴切片 (0=z, 1=y, 2=x)
            show_lung_only:     是否只显示肺部，其他区域都作为背景黑色
        """
        # 确保数据已加载
        if self.pixel_data is None:
            raise ValueError("未加载数据")
        # 确定切片索引
        if slice_idx is None:
            slice_idx = self.pixel_data.shape[axis] // 2
        # 提取切片数据
        if show_lung_only:
            if axis == 0:  # z轴
                slice_data = self.lung_seg_img[slice_idx, :, :]
            elif axis == 1:  # y轴
                slice_data = self.lung_seg_img[:, slice_idx, :]
            else :  # x轴
                slice_data = self.lung_seg_img[:, :, slice_idx]
        else:
            if axis == 0:  # z轴
                slice_data = self.pixel_data[slice_idx, :, :]
            elif axis == 1:  # y轴
                slice_data = self.pixel_data[:, slice_idx, :]
            else:  # x轴
                slice_data = self.pixel_data[:, :, slice_idx]
        # 创建图像
        plt.figure(figsize=(10, 8))
        # 仅显示图像
        plt.imshow(slice_data, cmap='gray')
        # 设置标题
        axis_name = ['z', 'y', 'x'][axis]
        title = f"切片 {slice_idx} (沿{axis_name}轴)"
        plt.title(title)
        plt.colorbar(label='像素值')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_nodule(self, coord_x,coord_y, coord_z, diameter):
        """
             结节可视化
        :param coord_x:
        :param coord_y:
        :param coord_z:
        :param diameter:
        :return:
        """
        # 提取结节立方体
        cube_size = max(32, int(diameter * 1.5))  # 确保立方体足够大
        cube = self.extract_cube([coord_x, coord_y, coord_z], cube_size)
        # 转换为体素坐标
        voxel_coord = self.world_to_voxel([coord_x, coord_y, coord_z])
        # 显示三个正交面
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # 提取中心切片
        center_z = cube.shape[0] // 2
        center_y = cube.shape[1] // 2
        center_x = cube.shape[2] // 2
        # 绘制三个正交面
        axes[0].imshow(cube[center_z, :, :], cmap='gray')
        axes[0].set_title(f'轴向视图 (z={center_z})')
        axes[0].axis('off')
        axes[1].imshow(cube[:, center_y, :], cmap='gray')
        axes[1].set_title(f'冠状位视图 (y={center_y})')
        axes[1].axis('off')
        axes[2].imshow(cube[:, :, center_x], cmap='gray')
        axes[2].set_title(f'矢状位视图 (x={center_x})')
        axes[2].axis('off')
        fig.suptitle(f"结节- 位置: ({coord_x:.1f}, {coord_y:.1f}, {coord_z:.1f})mm, " +
                     f"直径: {diameter:.1f}mm,", fontsize=14)
        plt.tight_layout()
        plt.show()

    def save_as_nifti(self, output_path):
        """
        将CT数据保存为NIfTI格式

        Args:
            output_path: 输出文件路径
        """
        # 确保数据已加载
        if self.pixel_data is None:
            raise ValueError("未加载数据")

        # 创建SimpleITK图像
        # 注意：SimpleITK的数组顺序为z,y,x
        img = sitk.GetImageFromArray(self.pixel_data)
        img.SetOrigin(self.origin)
        img.SetSpacing(self.spacing)

        if self.orientation is not None:
            img.SetDirection(self.orientation)
        # 保存为NIfTI格式
        sitk.WriteImage(img, output_path)
        print(f"已保存为NIfTI格式: {output_path}")

