import os
import sys
import logging
import numpy as np
import SimpleITK as sitk
import tempfile
import zipfile
import shutil
import pydicom
from scipy import ndimage

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入CTData类，如果不可用则创建一个简化版
try:
    # 添加项目根目录到系统路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # 导入原始CTData类
    from data.dataclass.CTData import CTData
except ImportError:
    # 如果无法导入，定义一个简化版的CTData类
    class CTData:
        """CT数据类，用于加载和处理CT图像"""
        
        def __init__(self):
            self.img = None  # 原始图像数据
            self.origin = None  # 图像原点
            self.spacing = None  # 图像间距
            self.is_normalized = False  # 是否已标准化
            self.has_lung_seg = False  # 是否已进行肺部分割
            self.lung_seg_img = None  # 肺部分割图像
            self.lung_mask = None  # 肺部掩码
            
        @classmethod
        def from_dicom(cls, dicom_path):
            """从DICOM文件夹加载CT数据"""
            ct_data = cls()
            
            try:
                logger.info(f"从DICOM加载: {dicom_path}")
                
                # 处理DICOM目录或zip文件
                temp_dir = None
                
                if os.path.isfile(dicom_path) and dicom_path.endswith('.zip'):
                    # 创建临时目录解压缩
                    temp_dir = tempfile.mkdtemp()
                    with zipfile.ZipFile(dicom_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    dicom_path = temp_dir
                
                # 读取DICOM序列
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
                
                if not dicom_names:
                    raise ValueError(f"在{dicom_path}中未找到DICOM文件")
                
                reader.SetFileNames(dicom_names)
                ct_sitk_img = reader.Execute()
                
                # 清理临时目录
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                
                # 转换为numpy数组
                ct_data.img = sitk.GetArrayFromImage(ct_sitk_img)
                ct_data.origin = ct_sitk_img.GetOrigin()
                ct_data.spacing = ct_sitk_img.GetSpacing()
                
                # 转换为HU单位
                ct_data.convert_to_hu()
                
                return ct_data
            
            except Exception as e:
                logger.error(f"从DICOM加载失败: {e}")
                # 确保临时目录被清理
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                raise
        
        @classmethod
        def from_mhd(cls, mhd_path):
            """从MHD文件加载CT数据"""
            ct_data = cls()
            
            try:
                logger.info(f"从MHD加载: {mhd_path}")
                
                # 读取MHD文件
                ct_sitk_img = sitk.ReadImage(mhd_path)
                
                # 转换为numpy数组
                ct_data.img = sitk.GetArrayFromImage(ct_sitk_img)
                ct_data.origin = ct_sitk_img.GetOrigin()
                ct_data.spacing = ct_sitk_img.GetSpacing()
                
                # 转换为HU单位 (假设MHD已经是HU单位)
                # 如果不是，可以取消下面的注释
                # ct_data.convert_to_hu()
                
                return ct_data
            
            except Exception as e:
                logger.error(f"从MHD加载失败: {e}")
                raise
        
        def resample_pixels(self, new_spacing=[1.0, 1.0, 1.0]):
            """将像素重采样到指定间距"""
            if self.img is None:
                logger.error("没有图像数据可重采样")
                return
            
            logger.info(f"重采样像素，原始间距: {self.spacing}，目标间距: {new_spacing}")
            
            # 计算调整后的大小
            resize_factor = np.array(self.spacing) / np.array(new_spacing)
            new_real_shape = self.img.shape * resize_factor
            new_shape = np.round(new_real_shape).astype(np.int32)
            
            # 计算用于调整大小的实际调整因子
            real_resize_factor = new_shape / self.img.shape
            real_new_spacing = np.array(self.spacing) / real_resize_factor
            
            # 使用sitk进行重采样
            sitk_img = sitk.GetImageFromArray(self.img)
            sitk_img.SetSpacing(self.spacing)
            
            resample = sitk.ResampleImageFilter()
            resample.SetInterpolator(sitk.sitkLinear)
            resample.SetOutputSpacing(real_new_spacing)
            resample.SetSize(new_shape.tolist())
            resample.SetOutputDirection(sitk_img.GetDirection())
            resample.SetOutputOrigin(sitk_img.GetOrigin())
            
            resampled_img = resample.Execute(sitk_img)
            self.img = sitk.GetArrayFromImage(resampled_img)
            self.spacing = real_new_spacing
            
            return self
        
        def convert_to_hu(self):
            """将图像转换为HU单位"""
            if self.img is None:
                logger.error("没有图像数据可转换")
                return
            
            if self.is_normalized:
                logger.info("图像已经转换为HU单位")
                return
            
            logger.info("转换图像为HU单位")
            
            # 对于DICOM，通常需要转换为HU单位
            # 但这个实现假设数据已经是HU或类似单位
            # 如果需要，这里可以添加特定的转换逻辑
            
            self.is_normalized = True
            return self
        
        def filter_lung_img_mask(self, threshold=-320):
            """提取肺部区域图像和掩码"""
            if self.img is None:
                logger.error("没有图像数据可分割")
                return
            
            logger.info("分割肺部区域")
            
            # 确保图像已转换为HU单位
            if not self.is_normalized:
                self.convert_to_hu()
            
            # 创建阈值掩码
            threshold_image = np.copy(self.img)
            threshold_image[threshold_image < threshold] = 1
            threshold_image[threshold_image >= threshold] = 0
            
            # 获取与身体连接的区域
            from scipy import ndimage as ndi
            
            # 填充身体外部的空气
            mask = self.fill_body_mask(threshold_image)
            
            # 反转掩码以获取身体内的空气区域
            lung_mask = np.logical_xor(threshold_image, mask)
            
            # 移除小连接区域
            struct = np.ones((2, 2, 2), dtype=np.bool_)
            lung_mask = ndi.binary_opening(lung_mask, structure=struct, iterations=2)
            
            labeled_mask, num_features = ndi.label(lung_mask)
            
            # 只保留最大的两个连接区域（肺）
            if num_features > 2:
                areas = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
                labels = np.argsort(areas)[-2:] + 1
                
                # 创建新的肺掩码
                lung_mask = np.zeros_like(labeled_mask, dtype=bool)
                for label in labels:
                    lung_mask = lung_mask | (labeled_mask == label)
            
            # 保存结果
            self.lung_mask = lung_mask
            self.lung_seg_img = self.img * lung_mask
            self.has_lung_seg = True
            
            return self
        
        def fill_body_mask(self, threshold_image):
            """填充身体外部空气区域"""
            # 创建边界种子
            mask = np.zeros_like(threshold_image, dtype=bool)
            mask[0, :, :] = True
            mask[-1, :, :] = True
            mask[:, 0, :] = True
            mask[:, -1, :] = True
            mask[:, :, 0] = True
            mask[:, :, -1] = True
            
            # 找到与边界连接的所有区域（即外部空气）
            from scipy import ndimage as ndi
            mask = ndi.binary_dilation(mask, structure=np.ones((3, 3, 3)), iterations=1)
            mask = np.logical_and(mask, threshold_image > 0)
            mask = ndi.binary_fill_holes(mask)
            
            return mask


def extract_lung_from_image(sitk_image):
    """
    从SimpleITK图像中提取肺部区域
    :param sitk_image: SimpleITK CT图像
    :return: 肺部分割掩码 (numpy数组)
    """
    # 转换为numpy数组
    ct_array = sitk.GetArrayFromImage(sitk_image)
    
    # 获取图像信息
    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    direction = sitk_image.GetDirection()
    
    # 进行肺部分割
    lung_mask = segment_lung(ct_array)
    
    return lung_mask

def segment_lung(ct_array):
    """
    简单的肺部分割算法
    :param ct_array: CT图像数组 [z, y, x]
    :return: 肺部掩码
    """
    # 1. 阈值处理 - 肺部通常是空气(-1000HU)到组织(-500HU)之间
    binary_image = np.logical_and(ct_array > -1000, ct_array < -500)
    
    # 2. 对每个切片进行处理
    result_mask = np.zeros_like(binary_image, dtype=bool)
    
    for i in range(binary_image.shape[0]):
        # 获取当前切片
        slice_img = binary_image[i].copy()
        
        # 填充边界以确保背景是连通的
        slice_img[0,:] = 1
        slice_img[-1,:] = 1
        slice_img[:,0] = 1
        slice_img[:,-1] = 1
        
        # 标记切片中的连通区域
        labeled, num_labels = ndimage.label(slice_img)
        
        # 按大小排序区域
        regions = ndimage.find_objects(labeled)
        region_sizes = [(i+1, (region[0].stop - region[0].start) * (region[1].stop - region[1].start)) 
                        for i, region in enumerate(regions)]
        region_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # 第一个最大的区域通常是背景或身体，我们需要肺部区域
        lung_mask = np.zeros_like(slice_img, dtype=bool)
        
        # 找出前5个最大区域（通常背景为最大，可能左右肺为2、3大区域）
        for j in range(min(5, len(region_sizes))):
            # 排除最大区域（通常是背景或身体）
            if j == 0:
                continue
                
            region_idx = region_sizes[j][0]
            # 添加这个区域到肺部掩码
            lung_mask[labeled == region_idx] = True
        
        # 形态学操作以填充肺部内的孔洞（例如血管）
        lung_mask = ndimage.binary_closing(lung_mask, structure=np.ones((5,5)))
        lung_mask = ndimage.binary_fill_holes(lung_mask)
        
        # 保存到结果
        result_mask[i] = lung_mask
    
    # 确保切片之间的连续性
    result_mask = ndimage.binary_closing(result_mask, structure=np.ones((3,3,3)))
    
    return result_mask.astype(np.uint8)

def extract_lung_from_file(file_path):
    """
    从文件中提取肺部区域
    支持MHD和DICOM格式
    :param file_path: 文件路径
    :return: 肺部数据字典
    """
    # 根据文件类型加载图像
    if file_path.lower().endswith('.mhd'):
        # 加载MHD文件
        ct_image = sitk.ReadImage(file_path)
    elif file_path.lower().endswith(('.dcm', '.dicom')):
        # 加载单个DICOM文件
        ct_image = sitk.ReadImage(file_path)
    elif os.path.isdir(file_path):
        # 加载DICOM系列
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(file_path)
        reader.SetFileNames(dicom_names)
        ct_image = reader.Execute()
    else:
        raise ValueError(f"不支持的文件类型: {file_path}")
    
    # 提取肺部
    lung_mask = extract_lung_from_image(ct_image)
    
    # 应用掩码到原始CT
    ct_array = sitk.GetArrayFromImage(ct_image)
    lung_data = ct_array.copy()
    lung_data[~lung_mask.astype(bool)] = -2000  # 设置非肺部区域为-2000HU
    
    # 构建返回数据
    return {
        'lung_data': lung_data,
        'lung_mask': lung_mask,
        'original_ct': ct_array,
        'spacing': ct_image.GetSpacing(),
        'origin': ct_image.GetOrigin(),
        'direction': ct_image.GetDirection()
    }

def prepare_data_for_3d_rendering(lung_data):
    """
    将肺部数据转换为适合3D渲染的格式
    :param lung_data: 肺部数据字典
    :return: 渲染数据
    """
    # 获取肺部掩码的轮廓
    mask = lung_data['lung_mask']
    
    # 降采样以减少数据量
    downsampled_mask = mask[::4, ::4, ::4]
    
    # 获取表面点（简单方法：获取非零点坐标）
    points = np.argwhere(downsampled_mask > 0).tolist()
    
    # 计算边界框
    if len(points) > 0:
        points_array = np.array(points)
        min_bounds = points_array.min(axis=0).tolist()
        max_bounds = points_array.max(axis=0).tolist()
    else:
        min_bounds = [0, 0, 0]
        max_bounds = list(downsampled_mask.shape)
    
    # 构建渲染数据
    render_data = {
        'points': points,
        'dimensions': downsampled_mask.shape,
        'min_bounds': min_bounds,
        'max_bounds': max_bounds
    }
    
    return render_data 