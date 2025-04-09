#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import time
from data.dataclass.CTData import CTData
from data.dataclass.NoduleCube import NoduleCube
from data.preprocessing.luna16_invalid_nodule_filter import nodule_valid

def get_mhd_file_path(patient_id, luna16_root="H:/luna16"):
    """
    根据patient_id查找对应的MHD文件路径

    Args:
        patient_id: LUNA16数据集中的患者ID
        luna16_root: LUNA16数据集根目录

    Returns:
        MHD文件的完整路径
    """
    # LUNA16数据集的子集文件夹
    subsets = [f"subset{i}" for i in range(10)]
    # 遍历所有子集查找对应的MHD文件
    for subset in subsets:
        subset_path = os.path.join(luna16_root, subset)
        if os.path.exists(subset_path):
            mhd_file = os.path.join(subset_path, f"{patient_id}.mhd")
            if os.path.exists(mhd_file):
                return mhd_file

    # 未找到对应的MHD文件
    print(f"警告: 未找到患者 {patient_id} 的MHD文件")
    return None

def get_real_candidate(mhd_root_dir, annotation_csv, candidate_csv,save_real_candidate_csv):
    """
        官方给的 候选结节标注，其实存在问题，可能与真实结节位置相近，导致数据误判。我们需要根据一定规则剔除
    :param mhd_root_dir:
    :param annotation_csv:
    :param candidate_csv:
    :param save_real_candidate_csv:
    :return:
    """
    positive_df = pd.read_csv(annotation_csv)
    part_positive_df = positive_df[["seriesuid","coordX", "coordY","coordZ"]]
    part_positive_df["class"] = 1
    negative_df = pd.read_csv(candidate_csv)
    part_negative_df = negative_df[negative_df["class"] == 0].copy()
    concat_df = pd.concat([part_positive_df, part_negative_df],axis=0)
    unique_seriesids = concat_df["seriesuid"].unique().tolist()
    # 最终保留哪些候选结节
    keep_negative_df_list = []
    remove_nodule_num = 0
    # 找到同一个病人的 真实结节标注 和 候选结节标注
    for seid in unique_seriesids:
        # 先查看对应数据在不在
        mhd_path = get_mhd_file_path(seid, mhd_root_dir)
        if mhd_path is not None and os.path.exists(mhd_path):
            one_serid_df = concat_df[concat_df["seriesuid"] == seid].copy()
            one_seid_postive_df = one_serid_df[one_serid_df["class"] == 1].copy()
            one_seid_negative_df = one_serid_df[one_serid_df["class"] == 0].copy()
            if one_seid_postive_df.shape[0] > 0 and one_seid_negative_df.shape[0] > 0:
                for _, negative_row in one_seid_negative_df.iterrows():
                    one_negative_coord_x = negative_row["coordX"]
                    one_negative_coord_y = negative_row["coordY"]
                    one_negative_coord_z = negative_row["coordZ"]
                    keep_current_nodule = True
                    for _, positive_row in one_seid_postive_df.iterrows():
                        one_positive_coord_x = positive_row["coordX"]
                        one_positive_coord_y = positive_row["coordY"]
                        one_positive_coord_z = positive_row["coordZ"]
                        x_dist = abs(one_negative_coord_x - one_positive_coord_x)
                        y_dist = abs(one_negative_coord_y - one_positive_coord_y)
                        z_dist = abs(one_negative_coord_z - one_positive_coord_z)
                        # 最大的结节的直径是32，我们必须保证所有候选结节 不与真实结节有重叠
                        if x_dist < 16 or y_dist < 16 or z_dist < 16:
                            keep_current_nodule = False
                            remove_nodule_num = remove_nodule_num + 1
                            break
                    # 所有候选结节与真实结节都不重叠
                    if keep_current_nodule:
                        one_keep_nodule_df = pd.DataFrame([{"seriesuid": seid,
                                                            "coordX": one_negative_coord_x,
                                                            "coordY": one_negative_coord_y,
                                                            "coordZ": one_negative_coord_z,
                                                            "class": 0}])
                        keep_negative_df_list.append(one_keep_nodule_df)
    print("最终有多少结节记录\t", len(keep_negative_df_list))
    print("剔除了多少个有问题的候选结节\t", remove_nodule_num)
    if len(keep_negative_df_list) > 0:
        keep_negative_df = pd.concat(keep_negative_df_list,axis = 0)
        keep_negative_df.to_csv(save_real_candidate_csv, encoding="utf-8", index = False)
        print("最终结果保存到\t", save_real_candidate_csv)

def ctdata_annotation2nodule(ct_data, nodule_info, mal_label, cube_size=64):
    """
    处理单个结节，提取立方体并保存为PNG和可视化图像

    Args:
        ct_data: CTData实例
        nodule_info: 结节信息(Series)
        mal_label:   当前结节标签
        cube_size: 立方体大小(mm)
    Returns:
        保存的文件路径元组 (png_path, viz_path)
    """
    # 获取结节信息
    patient_id = nodule_info['seriesuid']
    coord_x = nodule_info['coordX']
    coord_y = nodule_info['coordY']
    coord_z = nodule_info['coordZ']
    # 从CT数据中提取结节立方体
    nodule_cube_data = ct_data.extract_cube([coord_x, coord_y, coord_z], cube_size, if_fixed_radius=True)
    center_voxel = ct_data.world_to_voxel([coord_x, coord_y, coord_z])
    # 确保结节体积不为空
    if nodule_cube_data.size == 0:
        print(f"警告: 患者 {patient_id} 的结节体积为空，请检查坐标是否正确")
        return None, None
    # 打印原始数据形状
    # 创建NoduleCube实例
    nodule_cube = NoduleCube.from_array(
        pixel_data=nodule_cube_data,
        center_x=int(center_voxel[0]),
        center_y=int(center_voxel[1]),
        center_z=int(center_voxel[2]),
        radius=int(cube_size / 2),
        malignancy=mal_label
    )
    return nodule_cube


def process_nodule(nodule, cube_index,mhd_root_dir, label, if_aug, png_output, npy_output, check_output, check_count_limit):
    """
    处理单个结节的工作函数，适用于多进程

    Args:
        nodule: 结节信息（DataFrame的一行）
        mhd_root_dir: MHD文件的根目录
        label: 标签(0=良性, 1=恶性)
        if_aug: 是否需要做数据增强
        png_output: PNG输出目录
        npy_output: NPY输出目录
        check_output: 检查输出目录
        check_count_limit: 检查图像的最大数量
        cube_index: 结节索引

    Returns:
        处理结果信息字典
    """
    result = {
        'success': False,
        'patient_id': nodule['seriesuid'],
        'error': None,
        'check_saved': False
    }

    patient_id = nodule['seriesuid']
    # 获取MHD文件路径
    mhd_path = get_mhd_file_path(patient_id, mhd_root_dir)
    if mhd_path is None:
        result['error'] = "MHD文件未找到"
        return result

    patient_save_png_path = os.path.join(png_output, f"{patient_id}_mal={label}_{cube_index}.png")
    patient_save_npy_path = os.path.join(npy_output, f"{patient_id}_mal={label}_{cube_index}.npy")

    # 如果文件已存在，跳过处理
    if os.path.exists(patient_save_png_path):
        result['success'] = True
        result['error'] = "文件已存在，跳过处理"
        return result

    try:
        ct_data = CTData.from_mhd(mhd_path)
        ct_data.resample_pixel()
        ct_data.filter_lung_img_mask()

        # 处理结节
        one_nodule_cube = ctdata_annotation2nodule(ct_data, nodule, mal_label=label, cube_size=32)

        if one_nodule_cube is None:
            result['error'] = "结节体积为空"
            return result
        # 检查 良性结节的候选标注是否妥当。注意要先执行 resample_pixel 和 filter_lung_img_mask
        if label == 0:
            voxel_coord_x = one_nodule_cube.center_x
            voxel_coord_y = one_nodule_cube.center_y
            voxel_coord_z = one_nodule_cube.center_z
            this_nodule_valid = nodule_valid(ct_data, voxel_coord_x, voxel_coord_y, voxel_coord_z)
            if not this_nodule_valid:
                result['error'] = "结节不理想"
                return result
        # 保存为PNG拼接图和NPY文件
        one_nodule_cube.save_to_png(patient_save_png_path)
        one_nodule_cube.save_to_npy(patient_save_npy_path)
        if if_aug:
            # 3 次旋转
            rotation_nodule_cube1 = one_nodule_cube.augment(rotation=True)
            patient_save_png_path_rotatoion1 = patient_save_png_path.replace(".png", "_rotation1.png")
            patient_save_npy_path_rotatoion1 = patient_save_npy_path.replace(".npy", "_rotation1.npy")
            rotation_nodule_cube1.save_to_png(patient_save_png_path_rotatoion1)
            rotation_nodule_cube1.save_to_npy(patient_save_npy_path_rotatoion1)

            rotation_nodule_cube2 = one_nodule_cube.augment(rotation=True)
            patient_save_png_path_rotatoion2 = patient_save_png_path.replace(".png", "_rotation2.png")
            patient_save_npy_path_rotatoion2 = patient_save_npy_path.replace(".npy", "_rotation2.npy")
            rotation_nodule_cube2.save_to_png(patient_save_png_path_rotatoion2)
            rotation_nodule_cube2.save_to_npy(patient_save_npy_path_rotatoion2)

            rotation_nodule_cube3 = one_nodule_cube.augment(rotation=True)
            patient_save_png_path_rotatoion3 = patient_save_png_path.replace(".png", "_rotation3.png")
            patient_save_npy_path_rotatoion3 = patient_save_npy_path.replace(".npy", "_rotation3.npy")
            rotation_nodule_cube3.save_to_png(patient_save_png_path_rotatoion3)
            rotation_nodule_cube3.save_to_npy(patient_save_npy_path_rotatoion3)
            # 3次翻转
            flip_nodule_cube1 = one_nodule_cube.augment(rotation=False, flip_axis = 0)
            patient_save_png_path_flip1 = patient_save_png_path.replace(".png", "_flip1.png")
            patient_save_npy_path_flip1 = patient_save_npy_path.replace(".npy", "_flip1.npy")
            flip_nodule_cube1.save_to_png(patient_save_png_path_flip1)
            flip_nodule_cube1.save_to_npy(patient_save_npy_path_flip1)

            flip_nodule_cube2 = one_nodule_cube.augment(rotation=False, flip_axis=1)
            patient_save_png_path_flip2 = patient_save_png_path.replace(".png", "_flip2.png")
            patient_save_npy_path_flip2 = patient_save_npy_path.replace(".npy", "_flip2.npy")
            flip_nodule_cube2.save_to_png(patient_save_png_path_flip2)
            flip_nodule_cube2.save_to_npy(patient_save_npy_path_flip2)

            flip_nodule_cube3 = one_nodule_cube.augment(rotation=False, flip_axis=2)
            patient_save_png_path_flip3 = patient_save_png_path.replace(".png", "_flip3.png")
            patient_save_npy_path_flip3 = patient_save_npy_path.replace(".npy", "_flip3.npy")
            flip_nodule_cube3.save_to_png(patient_save_png_path_flip3)
            flip_nodule_cube3.save_to_npy(patient_save_npy_path_flip3)
            # 3次只加噪音
            noise_nodule_cube1 = one_nodule_cube.augment(rotation=False, flip_axis=-1, noise=True)
            patient_save_png_path_noise1 = patient_save_png_path.replace(".png", "_noise1.png")
            patient_save_npy_path_noise1 = patient_save_npy_path.replace(".npy", "_noise1.npy")
            noise_nodule_cube1.save_to_png(patient_save_png_path_noise1)
            noise_nodule_cube1.save_to_npy(patient_save_npy_path_noise1)

            noise_nodule_cube2 = one_nodule_cube.augment(rotation=False, flip_axis=-1, noise=True)
            patient_save_png_path_noise2 = patient_save_png_path.replace(".png", "_noise2.png")
            patient_save_npy_path_noise2 = patient_save_npy_path.replace(".npy", "_noise2.npy")
            noise_nodule_cube2.save_to_png(patient_save_png_path_noise2)
            noise_nodule_cube2.save_to_npy(patient_save_npy_path_noise2)

            noise_nodule_cube3 = one_nodule_cube.augment(rotation=False, flip_axis=-1, noise=True)
            patient_save_png_path_noise3 = patient_save_png_path.replace(".png", "_noise3.png")
            patient_save_npy_path_noise3 = patient_save_npy_path.replace(".npy", "_noise3.npy")
            noise_nodule_cube3.save_to_png(patient_save_png_path_noise3)
            noise_nodule_cube3.save_to_npy(patient_save_npy_path_noise3)

        # 判断是否需要保存检查图像
        if cube_index < check_count_limit:
            viz_path = os.path.join(check_output, f"{patient_id}_nodule_mal={label}_viz_{cube_index}.png")
            one_nodule_cube.visualize_3d(output_path=viz_path, show=False)
            result['check_saved'] = True

        result['success'] = True

    except Exception as e:
        result['error'] = str(e)
    return result

def prepare_cubes_mp(mhd_root_dir, annotation_csv, label, png_output, npy_output, check_output,if_aug = False, num_processes=None, max_samples = 60000):
    """
    多进程版本：从标注csv文件和mhd目录创建结节立方块

    Args:
        mhd_root_dir: mhd文件的根目录
        annotation_csv: 结节标注文件
        label: 当前标注文件良(0)/恶(1)性
        png_output: 结节立方块图片保存目录
        npy_output: 结节立方块数据保存目录
        check_output: 用于检查结节数据抽取是否准确的目录
        if_aug:     是否做数据增强，恶性结节数据较少，需要做增强
        num_processes: 进程数量，默认为CPU核心数的80%
        max_samples:   最大样本数，主要是针对负样本，因为正样本增强后也只有 9300，负样本有好几万
    """
    # 创建输出目录
    os.makedirs(png_output, exist_ok=True)
    os.makedirs(npy_output, exist_ok=True)
    os.makedirs(check_output, exist_ok=True)

    # 确定进程数
    if num_processes is None:
        num_processes = max(1, int(mp.cpu_count() * 0.8))

    # 加载标注数据
    annotations_df = pd.read_csv(annotation_csv, encoding="utf-8")
    print(annotations_df["class"].unique().tolist())
    # 确定只使用class=0 的记录，候选集里面还是有大量 class=1的数据
    if "class" in annotations_df.columns:
        annotations_df = annotations_df[annotations_df["class"] == 0].copy()

    annotations_df = annotations_df[:max_samples]
    total_nodules = len(annotations_df)

    print(f"开始处理 {total_nodules} 个{'恶性' if label == 1 else '良性'}结节，使用 {num_processes} 个进程")

    # 设置检查图像的数量限制
    check_count_limit = min(10, total_nodules)

    # 创建参数列表
    args_list = [
        (
            row,  # nodule
            i,  # cube_index
            mhd_root_dir,
            label,
            if_aug,
            png_output,
            npy_output,
            check_output,
            check_count_limit
        )
        for i, row in annotations_df.iterrows()
    ]
    # 使用进程池处理数据
    start_time = time.time()
    with mp.Pool(processes=num_processes) as pool:
        # 使用imap返回结果按顺序处理，并显示进度条
        results = list(tqdm(
            pool.starmap(process_nodule, args_list),
            total=total_nodules,
            desc=f"处理 {'恶性' if label == 1 else '良性'} 结节"
        ))

    # 统计处理结果
    success_count = sum(1 for r in results if r['success'])
    error_count = sum(1 for r in results if not r['success'])

    # 计算处理时间
    elapsed_time = time.time() - start_time
    avg_time_per_nodule = elapsed_time / total_nodules if total_nodules > 0 else 0

    print(
        f"处理完成！成功: {success_count}, 失败: {error_count}, 总共耗时: {elapsed_time:.2f}秒, 平均每个结节: {avg_time_per_nodule:.2f}秒")

    # 如果有错误，输出前5个错误
    if error_count > 0:
        print("错误样例:")
        error_samples = [r for r in results if not r['success']][:5]
        for i, sample in enumerate(error_samples):
            print(f"  {i + 1}. 患者ID: {sample['patient_id']}, 错误: {sample['error']}")

    return success_count, error_count


def main():
    """主函数"""
    # 设置路径
    positive_nodule_annotation_file = "H:/luna16/annotations.csv"
    negative_nodule_annotation_file = "H:/luna16/candidates_V2.csv"
    save_real_candidate_csv = "H:/luna16/candidates_clean.csv"
    luna16_root = "H:/luna16"
    # 从候选结节中筛选出 合适的 负样本，不要直接使用，里面存在大量有问题的数据
    get_real_candidate(luna16_root, positive_nodule_annotation_file, negative_nodule_annotation_file, save_real_candidate_csv)
    # 输出目录
    positive_png_save_dir = "J:/luna16_processed/positive_pngs/"
    positive_npy_save_dir = "J:/luna16_processed/positive_npys/"
    check_save_dir = "J:/luna16_processed/check_pngs/"
    # 设置进程数，默认为CPU核心数的80%
    num_processes = max(1, int(mp.cpu_count() * 0.8))
    print(f"系统检测到 {mp.cpu_count()} 个CPU核心，将使用 {num_processes} 个进程进行处理")
    # 处理恶性结节
    print("\n===== 处理恶性结节 =====")
    # success_pos, error_pos = prepare_cubes_mp(
    #     luna16_root,
    #     positive_nodule_annotation_file,
    #     1,
    #     positive_png_save_dir,
    #     positive_npy_save_dir,
    #     check_save_dir,
    #     True,
    #     num_processes
    # )
    # 处理良性结节
    print("\n===== 处理良性结节 =====")
    negative_png_save_dir = "J:/luna16_processed/negative_pngs/"
    negative_npy_save_dir = "J:/luna16_processed/negative_npys/"
    success_neg, error_neg = prepare_cubes_mp(
        luna16_root,
        save_real_candidate_csv,
        0,
        negative_png_save_dir,
        negative_npy_save_dir,
        check_save_dir,
        False,
        num_processes
    )
    # 总结处理结果
    print("\n===== 处理总结 =====")
    # print(f"恶性结节: 成功 {success_pos}, 失败 {error_pos}")
    print(f"良性结节: 成功 {success_neg}, 失败 {error_neg}")
    # print(f"总计: 成功 {success_pos + success_neg}, 失败 {error_pos + error_neg}")

if __name__ == "__main__":
    # 防止Windows多进程问题
    mp.freeze_support()
    main()