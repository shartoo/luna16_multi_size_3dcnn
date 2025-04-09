#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os,torch
import numpy as np
import cv2
from typing import  Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import ndimage

def normal_cube_to_tensor(cube_data):
    """
        将cube 数据归一化并转换为 pytorch tensor 。 用在训练和推理过程
    :param cube_data: shape为 [32,32,32] 的 ndarray
    :return:
    """
    cube_data = cube_data.astype(np.float32)
    # 归一化到 [0, 1] 范围
    min_val = np.min(cube_data)
    max_val = np.max(cube_data)
    data_range = max_val - min_val
    # 避免除以零
    if data_range < 1e-10:
        normalized_cube = np.zeros_like(cube_data)
    else:
        normalized_cube = (cube_data - min_val) / data_range
    # 检查是否有无效值并修复
    if np.isnan(normalized_cube).any() or np.isinf(normalized_cube).any():
        normalized_cube = np.nan_to_num(normalized_cube, nan=0.0, posinf=1.0, neginf=0.0)
    # 转换为PyTorch张量并添加批次和通道维度
    cube_tensor = torch.from_numpy(normalized_cube).float().unsqueeze(0)  # (1, 1, 32, 32, 32)
    return cube_tensor


@dataclass
class NoduleCube:
    """
    肺结节立方体类，表示肺结节区域的3D立方体数据
    与CT数据无关，仅处理已提取的立方体数据
    """
    # 基本属性
    cube_size: int = 64  # 立方体大小（默认64x64x64）
    pixel_data: Optional[np.ndarray] = None  # 像素数据 shape: [cube_size, cube_size, cube_size]
    
    # 结节特征
    center_x: int = 0  # 结节中心x坐标
    center_y: int = 0  # 结节中心y坐标
    center_z: int = 0  # 结节中心z坐标
    radius: float = 0.0  # 结节半径
    malignancy: int = 0  # 恶性度 (0 为良性 / 1 为恶性)
    
    # 文件路径
    npy_path: str = ""  # npy文件路径
    png_path: str = ""  # png文件路径

    def __post_init__(self):
        """初始化后调用"""
        # 如果提供了npy_path但没有pixel_data，尝试加载
        if self.npy_path and self.pixel_data is None:
            self.load_from_npy()
        # 如果提供了png_path但没有pixel_data，尝试加载
        elif self.png_path and self.pixel_data is None:
            self.load_from_png()

    def load_from_npy(self) -> None:
        """从NPY文件加载立方体数据"""
        if not os.path.exists(self.npy_path):
            raise FileNotFoundError(f"文件不存在: {self.npy_path}")
            
        try:
            self.pixel_data = np.load(self.npy_path)
            # 验证尺寸
            if len(self.pixel_data.shape) != 3:
                raise ValueError(f"像素数据必须是3D数组，当前形状: {self.pixel_data.shape}")
            
            # 如果尺寸不匹配，调整大小
            if (self.pixel_data.shape[0] != self.cube_size or 
                self.pixel_data.shape[1] != self.cube_size or 
                self.pixel_data.shape[2] != self.cube_size):
                self.resize(self.cube_size)
                
        except Exception as e:
            raise ValueError(f"加载NPY文件时出错: {e}")

    def save_to_npy(self, output_path: str) -> str:
        """
        将立方体数据保存为NPY文件
        
        Args:
            output_path: 输出路径
            
        Returns:
            保存的文件路径
        """
        if self.pixel_data is None:
            raise ValueError("没有像素数据可保存")
            
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, self.pixel_data)
        self.npy_path = output_path
        return output_path
    def save_to_png(self, output_path: str) -> str:
        """
        将立方体数据保存为PNG图像（8x8网格布局）
        
        Args:
            output_path: 输出PNG文件路径
            
        Returns:
            保存的文件路径
        """
        if self.pixel_data is None:
            raise ValueError("没有像素数据可保存")
            
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 计算每个切片在最终图像中的位置（8行8列布局）
        rows, cols = 8, 8
        if self.cube_size != 64:
            # 如果不是64x64x64，计算合适的行列数，保持接近正方形
            total_slices = self.cube_size
            rows = int(np.sqrt(total_slices))
            while total_slices % rows != 0:
                rows -= 1
            cols = total_slices // rows
        
        # 创建拼接图像
        img_height = self.cube_size
        img_width = self.cube_size
        combined_img = np.zeros((rows * img_height, cols * img_width), dtype=np.uint8)
        
        # 填充拼接图像
        for i in range(self.cube_size):
            row = i // cols
            col = i % cols
            
            slice_data = self.pixel_data[i]
            
            # 确保数据在0-255范围内
            if slice_data.max() <= 1.0:
                slice_data = (slice_data * 255).astype(np.uint8)
            else:
                slice_data = slice_data.astype(np.uint8)
            
            # 将切片放入拼接图像
            y_start = row * img_height
            x_start = col * img_width
            combined_img[y_start:y_start + img_height, x_start:x_start + img_width] = slice_data
        
        # 保存拼接图像
        cv2.imwrite(output_path, combined_img)
        self.png_path = output_path
        return output_path

    def load_from_png(self) -> None:
        """从PNG图像加载立方体数据（8x8网格布局）"""
        if not os.path.exists(self.png_path):
            raise FileNotFoundError(f"文件不存在: {self.png_path}")
            
        try:
            # 读取PNG图像
            img = cv2.imread(self.png_path, cv2.IMREAD_GRAYSCALE)
            
            # 确定行列数
            rows, cols = 8, 8
            if self.cube_size != 64:
                # 如果不是64x64x64，计算合适的行列数
                total_slices = self.cube_size
                rows = int(np.sqrt(total_slices))
                while total_slices % rows != 0:
                    rows -= 1
                cols = total_slices // rows
            
            # 确认图像尺寸正确
            expected_height = rows * self.cube_size
            expected_width = cols * self.cube_size
            if img.shape[0] != expected_height or img.shape[1] != expected_width:
                raise ValueError(f"图像尺寸不匹配: 期望{expected_height}x{expected_width}, 实际{img.shape[0]}x{img.shape[1]}")
            
            # 创建3D数组
            cube_data = np.zeros((self.cube_size, self.cube_size, self.cube_size), dtype=np.float32)
            
            # 从PNG图像提取每个切片
            for i in range(self.cube_size):
                row = i // cols
                col = i % cols
                
                y_start = row * self.cube_size
                x_start = col * self.cube_size
                
                slice_data = img[y_start:y_start + self.cube_size, x_start:x_start + self.cube_size]
                cube_data[i] = slice_data.astype(np.float32) / 255.0  # 归一化到[0,1]范围
            
            self.pixel_data = cube_data
            
        except Exception as e:
            raise ValueError(f"加载PNG文件时出错: {e}")

    def set_cube_data(self, pixel_data: np.ndarray) -> None:
        """
        设置立方体像素数据
        
        Args:
            pixel_data: 3D像素数据
        """
        if len(pixel_data.shape) != 3:
            raise ValueError(f"像素数据必须是3D数组，当前形状: {pixel_data.shape}")
        
        self.pixel_data = pixel_data
        
        # 如果尺寸不匹配，调整大小
        if (self.pixel_data.shape[0] != self.cube_size or 
            self.pixel_data.shape[1] != self.cube_size or 
            self.pixel_data.shape[2] != self.cube_size):
            self.resize(self.cube_size)

    def resize(self, new_size: int) -> None:
        """
        调整立方体尺寸
        
        Args:
            new_size: 新的立方体尺寸
        """
        if self.pixel_data is None:
            raise ValueError("没有像素数据可调整大小")
        
        # 计算缩放因子
        zoom_factors = [new_size / self.pixel_data.shape[0],
                         new_size / self.pixel_data.shape[1],
                         new_size / self.pixel_data.shape[2]]
        
        # 使用scipy的ndimage进行重采样
        self.pixel_data = ndimage.zoom(self.pixel_data, zoom_factors, mode='nearest')
        self.cube_size = new_size
        
    def augment(self, rotation: bool = True, flip_axis: int = -1, noise: bool = True) -> 'NoduleCube':
        """
        数据增强
        
        Args:
            rotation: 是否进行旋转增强
            flip_axis: 是否进行翻转增强，默认为-1（不翻转）
            noise: 是否添加噪声
            
        Returns:
            增强后的新立方体实例
        """
        if self.pixel_data is None:
            raise ValueError("没有像素数据可增强")
        
        # 创建副本
        augmented_cube = self.pixel_data.copy()
        
        # 旋转增强
        if rotation:
            # 随机选择旋转角度
            angles = np.random.uniform(-20, 20, 3)  # 在xyz三个方向上随机旋转
            augmented_cube = ndimage.rotate(augmented_cube, angles[0], axes=(1, 2), reshape=False, mode='nearest')
            augmented_cube = ndimage.rotate(augmented_cube, angles[1], axes=(0, 2), reshape=False, mode='nearest')
            augmented_cube = ndimage.rotate(augmented_cube, angles[2], axes=(0, 1), reshape=False, mode='nearest')
        
        # 翻转增强
        if flip_axis >=0:
            augmented_cube = np.flip(augmented_cube, axis=flip_axis)
        
        # 添加噪声
        if noise:
            # 添加随机高斯噪声
            noise_level = np.random.uniform(0.0, 0.05)
            noise_array = np.random.normal(0, noise_level, augmented_cube.shape)
            augmented_cube = augmented_cube + noise_array
            # 确保值在[0,1]范围内
            augmented_cube = np.clip(augmented_cube, 0, 1)
        
        # 创建新实例
        new_cube = NoduleCube(
            cube_size=self.cube_size,
            center_x=self.center_x,
            center_y=self.center_y,
            center_z=self.center_z,
            radius=self.radius,
            malignancy=self.malignancy
        )
        
        new_cube.set_cube_data(augmented_cube)
        return new_cube

    def visualize_3d(self, output_path: Optional[str] = None, show: bool = True) -> None:
        """
        可视化立方体数据
        
        Args:
            output_path: 可选的输出路径，如果提供则保存图像
            show: 是否显示图像
        """
        if self.pixel_data is None:
            raise ValueError("没有像素数据可视化")
            
        # 创建图像
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # 获取中心切片
        center_z = self.pixel_data.shape[0] // 2
        center_y = self.pixel_data.shape[1] // 2
        center_x = self.pixel_data.shape[2] // 2
        
        # 显示三个正交平面
        slice_xy = self.pixel_data[center_z, :, :]
        slice_xz = self.pixel_data[:, center_y, :]
        slice_yz = self.pixel_data[:, :, center_x]
        
        # 显示三个正交视图
        axes[0, 0].imshow(slice_xy, cmap='gray')
        axes[0, 0].set_title(f'轴向视图 (Z={center_z})')
        
        axes[0, 1].imshow(slice_xz, cmap='gray')
        axes[0, 1].set_title(f'矢状位视图 (Y={center_y})')
        
        axes[0, 2].imshow(slice_yz, cmap='gray')
        axes[0, 2].set_title(f'冠状位视图 (X={center_x})')
        
        # 3D渲染视图（使用MIP: Maximum Intensity Projection）
        mip_xy = np.max(self.pixel_data, axis=0)
        mip_xz = np.max(self.pixel_data, axis=1)
        mip_yz = np.max(self.pixel_data, axis=2)
        
        axes[1, 0].imshow(mip_xy, cmap='gray')
        axes[1, 0].set_title('最大强度投影 (轴向)')
        
        axes[1, 1].imshow(mip_xz, cmap='gray')
        axes[1, 1].set_title('最大强度投影 (矢状位)')
        
        axes[1, 2].imshow(mip_yz, cmap='gray')
        axes[1, 2].set_title('最大强度投影 (冠状位)')
        
        # 添加结节信息
        nodule_info = f"结节中心: ({self.center_x}, {self.center_y}, {self.center_z})\n"
        nodule_info += f"半径: {self.radius:.1f}\n"
        nodule_info += f"恶性度: {'恶性' if self.malignancy == 1 else '良性'}"
        
        fig.suptitle(nodule_info, fontsize=12)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
            
    @classmethod
    def from_npy(cls, file_path: str, cube_size: int = 64) -> 'NoduleCube':
        """
        从NPY文件创建立方体实例
        
        Args:
            file_path: NPY文件路径
            cube_size: 立方体大小
            
        Returns:
            NoduleCube实例
        """
        cube = cls(cube_size=cube_size, npy_path=file_path)
        cube.load_from_npy()
        return cube
    
    @classmethod
    def from_png(cls, file_path: str, cube_size: int = 64) -> 'NoduleCube':
        """
        从PNG文件创建立方体实例
        
        Args:
            file_path: PNG文件路径
            cube_size: 立方体大小
            
        Returns:
            NoduleCube实例
        """
        cube = cls(cube_size=cube_size, png_path=file_path)
        cube.load_from_png()
        return cube
        
    @classmethod
    def from_array(cls, 
                  pixel_data: np.ndarray, 
                  center_x: int = 0, 
                  center_y: int = 0, 
                  center_z: int = 0,
                  radius: float = 0.0,
                  malignancy: int = 0) -> 'NoduleCube':
        """
        从numpy数组创建立方体实例
        
        Args:
            pixel_data: 3D像素数据
            center_x: 中心点X坐标
            center_y: 中心点Y坐标
            center_z: 中心点Z坐标
            radius: 结节半径
            malignancy: 恶性度(0=良性, 1=恶性)
            
        Returns:
            NoduleCube实例
        """
        if len(pixel_data.shape) != 3:
            raise ValueError(f"像素数据必须是3D数组，当前形状: {pixel_data.shape}")
            
        cube_size = pixel_data.shape[0]
        if pixel_data.shape[1] != cube_size or pixel_data.shape[2] != cube_size:
            raise ValueError(f"像素数据必须是立方体形状，当前形状: {pixel_data.shape}")
            
        cube = cls(
            cube_size=cube_size,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            radius=radius,
            malignancy=malignancy
        )
        
        cube.set_cube_data(pixel_data)
        return cube


