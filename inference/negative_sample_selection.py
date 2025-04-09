import os
import sys
import numpy as np
import random
import glob
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import ndimage
# 导入CTData类
from data.dataclass.CTData import CTData
# 立方体大小
CUBE_SIZE = 32
def variance_of_laplacian(image):
    """计算图像的Laplacian方差，用于评估图像清晰度/模糊度"""
    # 计算图像的Laplacian
    laplacian = ndimage.laplace(image)
    # 返回Laplacian的方差
    return np.var(laplacian)

def texture_score(cube):
    """计算立方体的纹理分数，用于评估是否包含足够的纹理特征"""
    # 方法1: 使用Laplacian方差
    lap_scores = []
    for i in range(cube.shape[0]):
        lap_scores.append(variance_of_laplacian(cube[i]))
    lap_score = np.mean(lap_scores)
    
    # 方法2: 使用梯度幅值
    grad_x = ndimage.sobel(cube, axis=2)
    grad_y = ndimage.sobel(cube, axis=1)
    grad_z = ndimage.sobel(cube, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    grad_score = np.mean(grad_mag)
    
    # 方法3: 使用标准差（简单但有效）
    std_score = np.std(cube)
    
    # 组合分数
    combined_score = (lap_score * 0.3) + (grad_score * 0.3) + (std_score * 0.4)
    return combined_score

def is_valid_negative_sample(cube, lung_mask=None, nodule_coords=None, min_distance=20, min_texture_score=0.1):
    """
    检查立方体是否是有效的负样本
    
    Args:
        cube: 32x32x32立方体数据
        lung_mask: 肺部掩码，如果提供则检查是否在肺部内
        nodule_coords: 已知结节坐标列表，如果提供则检查是否距离已知结节足够远
        min_distance: 与已知结节的最小欧氏距离
        min_texture_score: 最小纹理分数阈值
        
    Returns:
        布尔值，表示是否是有效的负样本
    """
    # 计算纹理分数
    score = texture_score(cube)
    if score < min_texture_score:
        return False
    
    # 检查是否在肺部内
    if lung_mask is not None:
        if np.mean(lung_mask) < 0.6:  # 要求至少60%的体素在肺部内
            return False
    
    # 检查是否距离已知结节足够远
    if nodule_coords is not None and len(nodule_coords) > 0:
        cube_center = np.array([CUBE_SIZE//2, CUBE_SIZE//2, CUBE_SIZE//2])
        for nodule_coord in nodule_coords:
            distance = np.linalg.norm(cube_center - np.array(nodule_coord))
            if distance < min_distance:
                return False
    
    return True

def select_negative_samples_from_ct(ct_data, nodule_coords=None, num_samples=100, strategy='random'):
    """
    从CT数据中选择负样本
    
    Args:
        ct_data: CTData对象
        nodule_coords: 已知结节坐标列表
        num_samples: 要选择的负样本数量
        strategy: 选择策略，'random'(随机)或'kmeans'(聚类)
        
    Returns:
        负样本列表，每个样本为32x32x32的立方体
    """
    if ct_data.lung_seg_img is None:
        print("肺部区域数据未分割，正在分割...")
        ct_data.filter_lung_img_mask()
    
    lung_img = ct_data.lung_seg_img
    lung_mask = ct_data.lung_seg_mask
    
    # 获取肺部边界
    z_indices, y_indices, x_indices = np.where(lung_mask > 0)
    if len(z_indices) == 0:
        print("未找到肺部区域")
        return []
    
    z_min, z_max = z_indices.min(), z_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    
    # 调整范围，确保可以放置完整的立方体
    z_min = max(0, z_min)
    y_min = max(0, y_min)
    x_min = max(0, x_min)
    z_max = min(lung_img.shape[0] - CUBE_SIZE, z_max)
    y_max = min(lung_img.shape[1] - CUBE_SIZE, y_max)
    x_max = min(lung_img.shape[2] - CUBE_SIZE, x_max)
    
    # 收集候选点
    if strategy == 'random':
        # 随机选择策略
        candidate_samples = []
        attempts = 0
        max_attempts = num_samples * 10  # 最大尝试次数
        
        while len(candidate_samples) < num_samples and attempts < max_attempts:
            # 随机选择起始点
            z = random.randint(z_min, z_max)
            y = random.randint(y_min, y_max)
            x = random.randint(x_min, x_max)
            
            # 提取立方体和肺部掩码
            cube = lung_img[z:z+CUBE_SIZE, y:y+CUBE_SIZE, x:x+CUBE_SIZE]
            cube_mask = lung_mask[z:z+CUBE_SIZE, y:y+CUBE_SIZE, x:x+CUBE_SIZE]
            
            # 检查是否是有效的负样本
            if is_valid_negative_sample(cube, cube_mask, nodule_coords):
                candidate_samples.append({
                    'cube': cube,
                    'position': (z, y, x),
                    'score': texture_score(cube)
                })
            
            attempts += 1
        
        # 根据纹理分数排序，选择最佳样本
        candidate_samples.sort(key=lambda x: x['score'], reverse=True)
        selected_samples = candidate_samples[:num_samples]
        
        return [sample['cube'] for sample in selected_samples]
    
    elif strategy == 'kmeans':
        # 聚类选择策略（适用于大规模训练）
        # 首先生成大量候选点
        all_candidates = []
        for _ in range(num_samples * 5):
            z = random.randint(z_min, z_max)
            y = random.randint(y_min, y_max)
            x = random.randint(x_min, x_max)
            
            cube = lung_img[z:z+CUBE_SIZE, y:y+CUBE_SIZE, x:x+CUBE_SIZE]
            cube_mask = lung_mask[z:z+CUBE_SIZE, y:y+CUBE_SIZE, x:x+CUBE_SIZE]
            
            if is_valid_negative_sample(cube, cube_mask, nodule_coords):
                # 提取特征（使用简单的统计特征）
                mean_val = np.mean(cube)
                std_val = np.std(cube)
                texture = texture_score(cube)
                
                all_candidates.append({
                    'cube': cube,
                    'position': (z, y, x),
                    'features': [mean_val, std_val, texture],
                    'score': texture
                })
        
        if len(all_candidates) < num_samples:
            print(f"警告: 只找到 {len(all_candidates)} 个候选样本，少于请求的 {num_samples} 个")
            return [c['cube'] for c in all_candidates]
        
        # 提取特征矩阵
        features = np.array([c['features'] for c in all_candidates])
        
        # 标准化特征
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        
        # 使用KMeans聚类
        n_clusters = min(num_samples, len(all_candidates))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # 从每个簇中选择最佳样本
        selected_samples = []
        for i in range(n_clusters):
            cluster_samples = [all_candidates[j] for j in range(len(all_candidates)) if clusters[j] == i]
            if cluster_samples:
                best_sample = max(cluster_samples, key=lambda x: x['score'])
                selected_samples.append(best_sample)
        
        return [sample['cube'] for sample in selected_samples]
    
    else:
        raise ValueError(f"不支持的选择策略: {strategy}")

def generate_negative_samples(ct_paths, output_dir, nodules_csv=None, samples_per_ct=50):
    """
    从多个CT数据生成负样本
    
    Args:
        ct_paths: CT文件或目录路径列表
        output_dir: 输出目录
        nodules_csv: 包含已知结节信息的CSV文件路径
        samples_per_ct: 每个CT数据生成的负样本数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载已知结节信息
    nodule_coords_by_patient = {}
    if nodules_csv and os.path.exists(nodules_csv):
        import pandas as pd
        nodules_df = pd.read_csv(nodules_csv)
        for _, row in nodules_df.iterrows():
            patient_id = row['patient_id']
            if patient_id not in nodule_coords_by_patient:
                nodule_coords_by_patient[patient_id] = []
            nodule_coords_by_patient[patient_id].append(
                (row['voxel_x'], row['voxel_y'], row['voxel_z'])
            )
    
    # 处理每个CT数据
    for ct_path in ct_paths:
        try:
            # 获取患者ID
            if os.path.isfile(ct_path):
                patient_id = os.path.splitext(os.path.basename(ct_path))[0]
            else:
                patient_id = os.path.basename(ct_path)
            
            print(f"处理患者 {patient_id}...")
            
            # 加载CT数据
            if os.path.isfile(ct_path) and ct_path.endswith('.mhd'):
                ct_data = CTData.from_mhd(ct_path)
            elif os.path.isdir(ct_path):
                ct_data = CTData.from_dicom(ct_path)
            else:
                print(f"跳过不支持的文件类型: {ct_path}")
                continue
            
            # 获取已知结节坐标
            nodule_coords = nodule_coords_by_patient.get(patient_id, None)
            
            # 选择负样本
            negative_samples = select_negative_samples_from_ct(
                ct_data, 
                nodule_coords=nodule_coords,
                num_samples=samples_per_ct,
                strategy='random'  # 或 'kmeans'
            )
            
            # 保存负样本
            for i, sample in enumerate(negative_samples):
                output_path = os.path.join(output_dir, f"{patient_id}_neg_{i:03d}.npy")
                np.save(output_path, sample)
            
            print(f"已为患者 {patient_id} 生成 {len(negative_samples)} 个负样本")
            
        except Exception as e:
            print(f"处理 {ct_path} 时出错: {str(e)}")
    
    print("负样本生成完成!")

def visualize_samples(samples, output_path=None, cols=5):
    """可视化样本立方体，用于质量检查"""
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    
    for i, sample in enumerate(samples):
        if i < len(axes):
            # 显示立方体中间切片
            middle_slice = sample[sample.shape[0]//2]
            axes[i].imshow(middle_slice, cmap='gray')
            axes[i].set_title(f"Sample {i}")
            axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(len(samples), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='负样本选择工具')
    parser.add_argument('--input', type=str, required=True, help='输入CT文件或目录，或包含多个CT路径的文本文件')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--nodules', type=str, default=None, help='包含已知结节信息的CSV文件')
    parser.add_argument('--num-samples', type=int, default=50, help='每个CT数据生成的负样本数量')
    parser.add_argument('--strategy', type=str, default='random', choices=['random', 'kmeans'], help='样本选择策略')
    
    args = parser.parse_args()
    
    # 处理输入参数
    if os.path.isfile(args.input) and args.input.endswith('.txt'):
        # 从文本文件读取CT路径列表
        with open(args.input, 'r') as f:
            ct_paths = [line.strip() for line in f if line.strip()]
    else:
        # 单个CT路径
        ct_paths = [args.input]
    
    # 生成负样本
    generate_negative_samples(
        ct_paths,
        args.output,
        nodules_csv=args.nodules,
        samples_per_ct=args.num_samples
    ) 