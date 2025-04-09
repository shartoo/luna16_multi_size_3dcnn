import os
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from datetime import datetime
import logging
import time
from scipy import ndimage
from data.dataclass.CTData import CTData
from data.dataclass.NoduleCube import normal_cube_to_tensor
from deploy.backend.preprocessing.luna16_invalid_nodule_filter import nodule_valid
from models.pytorch_c3d_tiny import C3dTiny

# 推理参数
CUBE_SIZE = 32  # 扫描立方体大小 32x32x32
SCAN_STEP = 10  # 扫描步长，每次移动10个像素
PROB_THRESHOLD = 0.8  # 阈值: 大于此概率才视为结节

# 设置日志
def setup_logger(log_dir="./inference_logs"):
    """设置日志配置"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 创建logger
    logger = logging.getLogger('nodule_detection')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_ct_data(file_path):
    """
        加载CT数据（支持MHD和DICOM）并进行预处理
    Args:
        file_path: CT文件或文件夹路径
    Returns:
        CTData对象
    """
    # 判断是文件还是目录
    if os.path.isfile(file_path):
        # 假设是MHD文件
        if file_path.endswith('.mhd'):
            ct_data = CTData.from_mhd(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_path}")
    elif os.path.isdir(file_path):
        # 假设是DICOM文件夹
        ct_data = CTData.from_dicom(file_path)
    else:
        raise ValueError(f"指定路径不存在: {file_path}")
    # 重采样到1mm间距
    ct_data = ct_data.resample_pixel(new_spacing=[1, 1, 1])
    # 肺部区域分割
    ct_data.filter_lung_img_mask()
    return ct_data

def load_model(evaL_model_path, device='cuda'):
    """
    加载PyTorch模型
    
    Args:
        evaL_model_path: 模型权重文件路径
        device: 计算设备 ('cuda' 或 'cpu')

    Returns:
        加载好权重的模型
    """
    model = C3dTiny().to(device)
    # 加载权重
    model.load_state_dict(torch.load(evaL_model_path, map_location=device))
    model.eval()
    return model, device

def get_lung_bounds(lung_mask):
    """获取肺部掩码的边界框，考虑左右肺分离的情况"""
    if lung_mask.sum() == 0:
        return None
    # 使用连通区域分析找出肺部区域
    labeled_mask, num_features = ndimage.label(lung_mask > 0)
    # 如果连通区域过多，只考虑最大的几个区域（通常是左右肺）
    if num_features > 2:
        # 计算每个标签区域的体素数量
        region_sizes = np.array([(labeled_mask == i).sum() for i in range(1, num_features + 1)])
        # 只保留最大的2个区域（左右肺）
        valid_labels = np.argsort(region_sizes)[-2:] + 1
        # 创建新的掩码，只包含最大的几个区域
        refined_mask = np.zeros_like(labeled_mask)
        for label in valid_labels:
            refined_mask[labeled_mask == label] = 1
    else:
        refined_mask = lung_mask > 0
    
    # 根据z轴切片，计算每个切片的肺部区域
    z_ranges = []
    margin = 5  # 切片边距
    
    # 遍历每个z轴切片
    for z in range(refined_mask.shape[0]):
        slice_mask = refined_mask[z]
        if slice_mask.sum() > 100:  # 如果切片包含足够的肺部体素
            y_indices, x_indices = np.where(slice_mask)
            if len(y_indices) > 0:
                y_min = max(0, y_indices.min() - margin)
                y_max = min(refined_mask.shape[1], y_indices.max() + margin)
                x_min = max(0, x_indices.min() - margin)
                x_max = min(refined_mask.shape[2], x_indices.max() + margin)
                z_ranges.append((z, y_min, y_max, x_min, x_max))
    
    if not z_ranges:
        return None
    
    # 确定整体z轴范围
    z_min = z_ranges[0][0]
    z_max = z_ranges[-1][0] + 1
    
    # 收集所有y和x范围
    scan_regions = []
    for z_slice, y_min, y_max, x_min, x_max in z_ranges:
        scan_regions.append({
            'z': z_slice,
            'y_min': y_min,
            'y_max': y_max,
            'x_min': x_min,
            'x_max': x_max
        })
    
    return {
        'z_min': z_min,
        'z_max': z_max,
        'regions': scan_regions
    }

def scan_ct_data(ct_data, model, device, logger, step=SCAN_STEP):
    """
    扫描整个CT图像，预测结节位置 - 优化版
    
    Args:
        ct_data: CTData对象
        model: PyTorch模型
        device: 计算设备
        logger: 日志对象
        step: 扫描步长
        
    Returns:
        包含结节信息的DataFrame
    """
    logger.info("开始扫描CT数据...")
    
    # 获取肺部分割后的图像数据
    lung_img = ct_data.lung_seg_img
    lung_mask = ct_data.lung_seg_mask
    
    # 获取肺部边界信息
    bounds = get_lung_bounds(lung_mask)
    if bounds is None:
        logger.warning("未能找到有效的肺部区域")
        return pd.DataFrame(columns=['voxel_coord_x', 'voxel_coord_y', 'voxel_coord_z', 
                                    'world_coord_x', 'world_coord_y', 'world_coord_z', 'prob'])
    
    logger.info(f"已确定肺部区域: Z轴范围 {bounds['z_min']} 到 {bounds['z_max']}, 共 {len(bounds['regions'])} 个切片")
    
    # 创建存储结果的列表
    results = []
    
    # 计算需要扫描的总体素数估计
    total_voxels = 0
    for region in bounds['regions']:
        y_range = region['y_max'] - region['y_min']
        x_range = region['x_max'] - region['x_min']
        total_voxels += (y_range // step + 1) * (x_range // step + 1)
    
    logger.info(f"预计扫描体素数: {total_voxels}")
    
    # 开始计时
    start_time = time.time()
    batch_size = 32  # 增大批处理大小提高GPU利用率
    batch_inputs = []
    batch_positions = []
    # 跟踪进度
    processed_voxels = 0
    skipped_voxels = 0
    # 设置肺部组织比例阈值
    lung_tissue_threshold = 0.1  # 立方体中肺部组织的最小比例
    # 逐切片扫描肺部区域
    for z_idx, region in enumerate(bounds['regions']):
        z = region['z']
        # 检查是否可以放置一个完整的立方体
        if z + CUBE_SIZE > lung_img.shape[0]:
            continue
        # 在当前切片上扫描
        for y in range(region['y_min'], region['y_max'] - CUBE_SIZE + 1, step):
            for x in range(region['x_min'], region['x_max'] - CUBE_SIZE + 1, step):
                # 提取当前位置的肺部掩码立方体
                mask_cube = lung_mask[z:z+CUBE_SIZE, y:y+CUBE_SIZE, x:x+CUBE_SIZE]
                # 计算肺部组织比例
                lung_ratio = np.mean(mask_cube)
                # 如果肺部组织比例过低，跳过
                if lung_ratio < lung_tissue_threshold:
                    skipped_voxels += 1
                    continue
                # 提取当前位置的立方体
                cube = lung_img[z:z+CUBE_SIZE, y:y+CUBE_SIZE, x:x+CUBE_SIZE]
                # 预处理立方体数据
                cube_tensor = normal_cube_to_tensor(cube)
                cube_tensor = cube_tensor.unsqueeze(0)
                # 添加到批处理
                batch_inputs.append(cube_tensor)
                batch_positions.append((z, y, x))
                # 当批处理达到指定大小时进行预测
                if len(batch_inputs) == batch_size:
                    # 处理当前批次
                    process_batch(batch_inputs, batch_positions, model, device, ct_data, results)
                    batch_inputs = []
                    batch_positions = []
                processed_voxels += 1
                # 定期报告进度
                if (processed_voxels + skipped_voxels) % 1000 == 0:
                    elapsed_time = time.time() - start_time
                    progress = processed_voxels / total_voxels * 100 if total_voxels > 0 else 0
                    logger.info(f"处理进度: {processed_voxels}/{total_voxels} ({progress:.2f}%), "
                                f"已跳过: {skipped_voxels}, 耗时: {elapsed_time:.2f}秒")
    
    # 处理最后一个批次
    if batch_inputs:
        process_batch(batch_inputs, batch_positions, model, device, ct_data, results)
    
    # 创建DataFrame
    if results:
        results_df = pd.DataFrame(results)
        logger.info(f"扫描完成! 发现 {len(results_df)} 个可能的结节")
    else:
        results_df = pd.DataFrame(columns=['voxel_coord_x', 'voxel_coord_y', 'voxel_coord_z', 
                                          'world_coord_x', 'world_coord_y', 'world_coord_z', 'prob'])
        logger.info("扫描完成! 未发现任何结节")
    
    return results_df

def process_batch(batch_inputs, batch_positions, model, device, ct_data, results):
    """处理一个批次的数据"""
    # 合并批处理
    batch_tensor = torch.cat(batch_inputs, dim=0).to(device)
    # 预测
    with torch.no_grad():
        batch_outputs = model(batch_tensor)
        batch_probs = F.softmax(batch_outputs, dim=1)[:, 1]  # 类别1的概率
    # 处理每个预测结果
    for i, prob in enumerate(batch_probs):
        prob_value = prob.item()
        if prob_value > PROB_THRESHOLD:
            z_pos, y_pos, x_pos = batch_positions[i]
            # 计算中心点坐标
            center_z = z_pos + CUBE_SIZE // 2
            center_y = y_pos + CUBE_SIZE // 2
            center_x = x_pos + CUBE_SIZE // 2
            # 将体素坐标转换为世界坐标 (mm)
            world_coord = ct_data.voxel_to_world([center_x, center_y, center_z])
            # 添加结果
            results.append({
                'voxel_coord_x': center_x,
                'voxel_coord_y': center_y,
                'voxel_coord_z': center_z,
                'world_coord_x': world_coord[0],
                'world_coord_y': world_coord[1],
                'world_coord_z': world_coord[2],
                'prob': prob_value
            })

def reduce_overlapping_nodules(results_df, distance_threshold=15):
    """
    合并重叠的结节预测，使用更严格的距离阈值
    
    Args:
        results_df: 包含结节预测的DataFrame
        distance_threshold: 合并的距离阈值(体素)
        
    Returns:
        合并后的结节DataFrame
    """
    if len(results_df) <= 1:
        return results_df
    # 按概率从高到低排序
    sorted_df = results_df.sort_values('prob', ascending=False).reset_index(drop=True)
    # 创建一个布尔掩码来标记要保留的行
    keep_mask = np.ones(len(sorted_df), dtype=bool)
    # 对每一行
    for i in range(len(sorted_df)):
        if not keep_mask[i]:
            continue  # 如果此行已被标记为删除，则跳过
        # 获取当前结节的坐标
        current = sorted_df.iloc[i]
        # 比较与其他所有结节的距离
        for j in range(i + 1, len(sorted_df)):
            if not keep_mask[j]:
                continue  # 如果要比较的行已被标记为删除，则跳过
            # 获取要比较的结节坐标
            compare = sorted_df.iloc[j]
            # 计算3D欧氏距离
            distance = np.sqrt(
                (current['voxel_coord_x'] - compare['voxel_coord_x']) ** 2 +
                (current['voxel_coord_y'] - compare['voxel_coord_y']) ** 2 +
                (current['voxel_coord_z'] - compare['voxel_coord_z']) ** 2
            )
            # 如果距离小于阈值，标记为删除
            if distance < distance_threshold:
                keep_mask[j] = False
    # 应用掩码，仅保留未被标记为删除的行
    reduced_df = sorted_df[keep_mask].reset_index(drop=True)
    return reduced_df

def filter_false_positives(nodules_df, ct_data, max_nodules=10):
    """
    基于解剖学和统计特征过滤假阳性结节
    
    Args:
        nodules_df: 包含结节预测的DataFrame
        ct_data: CTData对象
        max_nodules: 每个患者允许的最大结节数量

    Returns:
        过滤后的结节DataFrame
    """
    if nodules_df.empty:
        return nodules_df
    # 获取肺部掩码
    lung_mask = ct_data.lung_seg_mask
    # 1. 限制结节总数
    if len(nodules_df) > max_nodules:
        # 只保留概率最高的前N个结节
        nodules_df = nodules_df.sort_values('prob', ascending=False).head(max_nodules)
    # 2. 基于位置过滤
    filtered_rows = []
    for i, row in nodules_df.iterrows():
        x, y, z = int(row['voxel_coord_x']), int(row['voxel_coord_y']), int(row['voxel_coord_z'])
        this_nodule_valid = nodule_valid(ct_data, x, y, z)
        if this_nodule_valid:
            # 通过所有检查，保留此结节
            filtered_rows.append(row)
    # 创建新的DataFrame
    filtered_df = pd.DataFrame(filtered_rows)
    # 3. 基于概率再次过滤
    # 如果概率低于阈值，移除
    # high_prob_threshold = 0.95  # 高概率阈值
    # filtered_df = filtered_df[filtered_df['prob'] >= high_prob_threshold]
    return filtered_df

def format_results(results_df, ct_data, patient_id):
    """
    格式化结果为最终输出的DataFrame
    
    Args:
        results_df: 合并后的结节DataFrame
        ct_data: CTData对象
        patient_id: 患者ID
        
    Returns:
        包含结节信息的最终DataFrame
    """
    # 如果没有结节，返回空的DataFrame
    if results_df.empty:
        return pd.DataFrame(columns=['patient_id', 'nodule_id', 'voxel_x', 'voxel_y', 'voxel_z', 
                                     'world_x', 'world_y', 'world_z', 'diameter_mm', 'prob'])
    # 创建最终结果列表
    final_results = []
    # 处理每个结节
    for i, row in results_df.iterrows():
        # 设置默认直径为CUBE_SIZE / 2
        diameter_mm = CUBE_SIZE / 2
        # 添加结果
        final_results.append({
            'patient_id': patient_id,
            'nodule_id': i + 1,
            'voxel_x': int(row['voxel_coord_x']),
            'voxel_y': int(row['voxel_coord_y']),
            'voxel_z': int(row['voxel_coord_z']),
            'world_x': row['world_coord_x'],
            'world_y': row['world_coord_y'],
            'world_z': row['world_coord_z'],
            'diameter_mm': diameter_mm,
            'prob': row['prob']
        })
    
    # 创建DataFrame
    final_df = pd.DataFrame(final_results)
    
    return final_df

def detect_nodules(file_path, model_path, detect_patient_id=None, device='cuda'):
    """
    主函数：对CT数据进行结节检测
    
    Args:
        file_path: CT文件或文件夹路径
        model_path: 模型权重文件路径
        detect_patient_id: 患者ID，如果为None则使用文件名
        device: 计算设备 ('cuda' 或 'cpu')
        
    Returns:
        包含结节信息的DataFrame
    """
    # 设置日志
    logger = setup_logger()
    # 如果患者ID为None，则使用文件名
    if detect_patient_id is None:
        if os.path.isfile(file_path):
            detect_patient_id = os.path.splitext(os.path.basename(file_path))[0]
        else:
            detect_patient_id = os.path.basename(file_path)
    logger.info(f"开始处理患者 {detect_patient_id} 的CT数据")
    try:
        # 加载CT数据
        logger.info(f"加载CT数据: {file_path}")
        ct_data = load_ct_data(file_path)
        # 加载模型
        logger.info(f"加载模型: {model_path}")
        model, device = load_model(model_path, device)
        # 扫描CT数据
        results_df = scan_ct_data(ct_data, model, device, logger)
        # 合并重叠结节
        logger.info("合并重叠结节...")
        reduced_df = reduce_overlapping_nodules(results_df)
        logger.info(f"合并后的结节数量: {len(reduced_df)}")
        # 过滤假阳性
        logger.info("过滤假阳性结节...")
        filtered_df = filter_false_positives(reduced_df, ct_data)
        logger.info(f"过滤后的结节数量: {len(filtered_df)}")
        # 格式化结果
        final_df = format_results(filtered_df, ct_data, patient_id)
        logger.info(f"检测完成，找到 {len(final_df)} 个结节")
        return final_df
    except Exception as e:
        logger.error(f"检测过程中出错: {str(e)}", exc_info=True)
        raise
    
if __name__ == "__main__":
    test_mhd = "H:/luna16/subset8/1.3.6.1.4.1.14519.5.2.1.6279.6001.149041668385192796520281592139.mhd"
    model_path = "../training/pytorch_checkpoints/best_model.pth"
    threshold = 0.7
    patient_id = "1.3.6.1.4.1.14519.5.2.1.6279.6001.149041668385192796520281592139"
    detect_result_csv = "./c3d_classify_result-%s.csv" %patient_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 运行检测
    result_df = detect_nodules(test_mhd, model_path, None, device)
    # 保存结果
    result_df.to_csv(detect_result_csv, index=False, encoding="utf-8")