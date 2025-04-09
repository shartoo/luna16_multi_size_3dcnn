## 去掉 Luna2016 候选结节数据中 有问题的标注数据 以及 用在预测过程中的 错误结节
import numpy as np

def nodule_valid(ct_data, voxel_coord_x, voxel_coord_y,voxel_coord_z):
    """
        判定当前结节是否 可以用来做训练cube 或者 扫描得到的cube 是否
    :param ct_data:         已经转换为0-255 ，并且已经抽取到肺部区域数据的 CTData类
    :param voxel_coord_x:   当前要判定的 cube的坐标中心位置
    :param voxel_coord_y:
    :param voxel_coord_z:
    :return:                当前结节是否可用 True(可用) / False (不可用)
    """
    lung_mask = ct_data.lung_seg_mask
    # 检查坐标是否在肺部边界内
    if (voxel_coord_z < 0 or voxel_coord_z >= lung_mask.shape[0] or
            voxel_coord_y < 0 or voxel_coord_y >= lung_mask.shape[1] or
            voxel_coord_x < 0 or voxel_coord_x >= lung_mask.shape[2]):
        return False
    # 获取周围半径为5个体素的区域
    z_min = max(0, voxel_coord_z - 5)
    z_max = min(lung_mask.shape[0], voxel_coord_z + 6)
    y_min = max(0, voxel_coord_y - 5)
    y_max = min(lung_mask.shape[1], voxel_coord_y + 6)
    x_min = max(0, voxel_coord_x - 5)
    x_max = min(lung_mask.shape[2], voxel_coord_x + 6)
    # 提取周围区域的肺部掩码
    neighborhood_mask = lung_mask[z_min:z_max, y_min:y_max, x_min:x_max]
    # 计算肺部组织占比
    lung_ratio = np.mean(neighborhood_mask)
    # 如果周围区域肺部组织占比太低，可能是假阳性
    if lung_ratio < 0.5:
        return False

    # 检查是否在肺部边缘
    # 计算当前点在肺部掩码中的位置
    if (0 < voxel_coord_z < lung_mask.shape[0] - 1 and
            0 < voxel_coord_y < lung_mask.shape[1] - 1 and
            0 < voxel_coord_x < lung_mask.shape[2] - 1):
        # 计算6-邻域（上下左右前后）中肺部体素的数量
        neighbors = [
            lung_mask[voxel_coord_z - 1, voxel_coord_y, voxel_coord_x],
            lung_mask[voxel_coord_z + 1, voxel_coord_y, voxel_coord_x],
            lung_mask[voxel_coord_z, voxel_coord_y - 1, voxel_coord_x],
            lung_mask[voxel_coord_z, voxel_coord_y + 1, voxel_coord_x],
            lung_mask[voxel_coord_z, voxel_coord_y, voxel_coord_x - 1],
            lung_mask[voxel_coord_z, voxel_coord_y, voxel_coord_x + 1]
        ]
        # 如果邻域中有过多非肺部体素，说明这可能是在肺部边缘
        if sum(neighbors) < 4:
            return False
    return True