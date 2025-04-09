import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import torch
from threading import Thread, Lock
import json

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入肺结节检测模块
from inference.pytorch_nodule_detector import (
    load_ct_data, 
    load_model, 
    get_lung_bounds, 
    scan_ct_data, 
    reduce_overlapping_nodules, 
    filter_false_positives, 
    format_results
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检测会话状态
session_states = {}
session_locks = {}

# 检测状态常量
STATUS_LOADING = "loading"       # 加载数据
STATUS_PREPROCESSING = "preprocessing"  # 预处理
STATUS_SCANNING = "scanning"     # 扫描检测
STATUS_FILTERING = "filtering"   # 过滤假阳性
STATUS_COMPLETED = "completed"   # 完成
STATUS_ERROR = "error"           # 错误

class NoduleDetector:
    """肺结节检测器，包装pytorch_nodule_detector.py的功能"""
    
    def __init__(self, model_path=None, device='cuda'):
        """初始化检测器
        
        Args:
            model_path: 模型路径
            device: 使用设备 (cuda或cpu)
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model = None
        print("给出的模型路径是\t", model_path)
        # 如果提供了模型路径，加载模型
        if model_path and os.path.exists(model_path):
            print("模型已经完成加载!")
            self.load_model(model_path)
        
        # 添加完成回调
        self.completion_callback = None
            
    def load_model(self, model_path):
        """加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            加载是否成功
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return False
                
            self.model, self.device = load_model(model_path, self.device)
            self.model_path = model_path
            return True
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            return False
            
    def detect(self, file_path, session_id, patient_id=None):
        """启动肺结节检测
        
        Args:
            file_path: CT文件或文件夹路径
            session_id: 会话ID
            patient_id: 患者ID
            
        Returns:
            布尔值，表示检测是否成功启动
        """
        if not self.model:
            logger.error("模型未加载")
            return False
            
        if session_id in session_states:
            logger.warning(f"会话 {session_id} 已存在，将被覆盖")
        print("进入detect 中的 file_path\t", file_path)
        # 初始化会话状态
        session_states[session_id] = {
            "status": STATUS_LOADING,
            "progress": 0,
            "message": "正在加载CT数据...",
            "started_at": time.time(),
            "patient_id": patient_id,
            "file_path": file_path,
            "ct_data": None,
            "nodules": None,
            "lung_bounds": None,
            "error": None
        }
        
        # 创建锁
        if session_id not in session_locks:
            session_locks[session_id] = Lock()
            
        # 启动检测线程
        thread = Thread(target=self._detect_thread, args=(file_path, session_id, patient_id))
        thread.daemon = True
        thread.start()
        
        return True
        
    def _detect_thread(self, file_path, session_id, patient_id):
        """检测线程
        
        Args:
            file_path: CT文件或文件夹路径
            session_id: 会话ID
            patient_id: 患者ID
        """
        try:
            # 加载CT数据
            self._update_session(session_id, {
                "status": STATUS_LOADING,
                "progress": 0,
                "message": "正在加载CT数据..."
            })
            
            ct_data = load_ct_data(file_path)
            
            # 检查CT数据是否加载成功
            if ct_data is None:
                raise ValueError("CT数据加载失败")
                
            # 更新会话状态
            self._update_session(session_id, {
                "ct_data": ct_data,
                "progress": 10,
                "message": "CT数据加载完成，开始进行肺部分割..."
            })
            
            # 获取肺部边界信息
            self._update_session(session_id, {
                "status": STATUS_PREPROCESSING,
                "progress": 20,
                "message": "正在进行肺部分割..."
            })
            
            # 肺部分割在load_ct_data中已完成，这里获取肺部边界信息
            lung_bounds = get_lung_bounds(ct_data.lung_seg_mask)
            if lung_bounds is None:
                raise ValueError("未能找到有效的肺部区域")
                
            # 更新会话状态
            self._update_session(session_id, {
                "lung_bounds": lung_bounds,
                "progress": 30,
                "message": "肺部分割完成，开始检测结节..."
            })
            
            # 开始扫描检测
            self._update_session(session_id, {
                "status": STATUS_SCANNING,
                "progress": 40,
                "message": "正在扫描肺部区域，检测结节..."
            })
            # 创建logger用于接收进度更新
            progress_logger = self._create_progress_logger(session_id)
            # 执行扫描
            results_df = scan_ct_data(ct_data, self.model, self.device, progress_logger)
            # 更新会话状态
            self._update_session(session_id, {
                "progress": 80,
                "message": f"扫描完成，初步检测到 {len(results_df)} 个可能的结节，进行假阳性过滤..."
            })
            
            # 合并重叠结节
            self._update_session(session_id, {
                "status": STATUS_FILTERING,
                "progress": 85,
                "message": "正在合并重叠结节..."
            })
            reduced_df = reduce_overlapping_nodules(results_df)
            
            # 过滤假阳性
            self._update_session(session_id, {
                "progress": 90,
                "message": f"初步合并后剩余 {len(reduced_df)} 个结节，正在进行假阳性过滤..."
            })
            filtered_df = filter_false_positives(reduced_df, ct_data)
            
            # 格式化结果
            self._update_session(session_id, {
                "progress": 95,
                "message": f"过滤完成，最终检测到 {len(filtered_df)} 个结节，正在生成结果..."
            })
            final_df = format_results(filtered_df, ct_data, patient_id or session_id)
            
            # 提取结节立方体
            nodules = self._extract_nodule_cubes(ct_data, final_df)
            
            # 更新会话状态
            final_results = {
                "status": STATUS_COMPLETED,
                "progress": 100,
                "message": f"检测完成，共发现 {len(nodules)} 个结节",
                "nodules": nodules,
                "completed_at": time.time()
            }
            
            self._update_session(session_id, final_results)
            
            # 调用完成回调函数
            if self.completion_callback:
                try:
                    self.completion_callback(session_id, {"nodules": nodules})
                except Exception as callback_error:
                    logger.error(f"执行完成回调函数时出错: {str(callback_error)}", exc_info=True)
            
        except Exception as e:
            logger.error(f"检测过程出错: {str(e)}", exc_info=True)
            self._update_session(session_id, {
                "status": STATUS_ERROR,
                "message": f"检测失败: {str(e)}",
                "error": str(e)
            })
    
    def _create_progress_logger(self, session_id):
        """创建用于接收进度更新的logger
        
        Args:
            session_id: 会话ID
            
        Returns:
            自定义logger对象
        """
        class ProgressLogger:
            def __init__(self, outer_self, session_id):
                self.outer_self = outer_self
                self.session_id = session_id
                self.last_progress_time = time.time()
                
            def info(self, message):
                # 解析进度信息
                if "处理进度:" in message:
                    try:
                        # 从消息中提取进度百分比
                        progress_part = message.split("处理进度:")[1].split("%")[0].strip()
                        progress_parts = progress_part.split('/')
                        if len(progress_parts) == 2:
                            current, total = map(int, progress_parts)
                            progress = min(40 + int(current / total * 40), 80)  # 扫描进度从40%到80%
                            
                            # 每秒最多更新一次进度
                            current_time = time.time()
                            if current_time - self.last_progress_time > 1:
                                self.last_progress_time = current_time
                                self.outer_self._update_session(self.session_id, {
                                    "progress": progress,
                                    "message": message
                                })
                    except:
                        pass
                
                # 记录其他重要消息
                if "扫描完成" in message:
                    self.outer_self._update_session(self.session_id, {
                        "progress": 80,
                        "message": message
                    })
        
        return ProgressLogger(self, session_id)
    
    def _extract_nodule_cubes(self, ct_data, nodules_df, cube_size=32):
        """从CT数据中提取结节立方体
        
        Args:
            ct_data: CTData对象
            nodules_df: 结节DataFrame
            cube_size: 立方体大小
            
        Returns:
            结节列表，每个结节包含立方体数据和相关信息
        """
        nodule_list = []
        
        for _, row in nodules_df.iterrows():
            try:
                # 获取体素坐标
                x = int(row['voxel_x'])
                y = int(row['voxel_y'])
                z = int(row['voxel_z'])
                
                # 计算立方体区域边界
                half_size = cube_size // 2
                z_min = max(0, z - half_size)
                y_min = max(0, y - half_size)
                x_min = max(0, x - half_size)
                z_max = min(ct_data.lung_seg_img.shape[0], z + half_size)
                y_max = min(ct_data.lung_seg_img.shape[1], y + half_size)
                x_max = min(ct_data.lung_seg_img.shape[2], x + half_size)
                
                # 提取立方体
                cube = ct_data.lung_seg_img[z_min:z_max, y_min:y_max, x_min:x_max]
                
                # 如果立方体大小不符合要求，进行调整
                if cube.shape != (cube_size, cube_size, cube_size):
                    # 使用零填充调整大小
                    padded_cube = np.zeros((cube_size, cube_size, cube_size), dtype=cube.dtype)
                    padded_cube[:min(cube_size, cube.shape[0]), 
                               :min(cube_size, cube.shape[1]), 
                               :min(cube_size, cube.shape[2])] = cube[:min(cube_size, cube.shape[0]), 
                                                                      :min(cube_size, cube.shape[1]), 
                                                                      :min(cube_size, cube.shape[2])]
                    cube = padded_cube
                
                # 添加结节信息
                nodule_info = {
                    'id': int(row['nodule_id']),
                    'cube': cube.tolist(),  # 转换为列表以便JSON序列化
                    'voxel_coords': [x, y, z],
                    'world_coords': [float(row['world_x']), float(row['world_y']), float(row['world_z'])],
                    'diameter_mm': float(row['diameter_mm']),
                    'probability': float(row['prob'])
                }
                
                nodule_list.append(nodule_info)
                
            except Exception as e:
                logger.error(f"提取结节立方体时出错: {str(e)}", exc_info=True)
        
        return nodule_list
    
    def _update_session(self, session_id, updates):
        """更新会话状态
        
        Args:
            session_id: 会话ID
            updates: 要更新的字段
        """
        if session_id not in session_states:
            logger.warning(f"会话 {session_id} 不存在")
            return
            
        # 获取锁
        with session_locks[session_id]:
            for key, value in updates.items():
                session_states[session_id][key] = value
                
    def get_session_state(self, session_id):
        """获取会话状态
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话状态字典
        """
        if session_id not in session_states:
            return None
            
        # 获取锁
        with session_locks[session_id]:
            # 创建副本以避免修改原始状态
            state = session_states[session_id].copy()
            
            # 移除不可序列化的字段
            if 'ct_data' in state:
                del state['ct_data']
            if 'lung_bounds' in state:
                bounds_info = {}
                if state['lung_bounds']:
                    bounds_info = {
                        'z_min': state['lung_bounds']['z_min'],
                        'z_max': state['lung_bounds']['z_max'],
                        'region_count': len(state['lung_bounds']['regions'])
                    }
                state['lung_bounds'] = bounds_info
            
            return state
            
    def set_completion_callback(self, callback):
        """设置检测完成后的回调函数
        
        Args:
            callback: 回调函数，接收session_id和results参数
        """
        self.completion_callback = callback
            
def get_detector_instance(model_path=None):
    """获取检测器实例（单例模式）
    
    Args:
        model_path: 模型路径
        
    Returns:
        NoduleDetector实例
    """
    if not hasattr(get_detector_instance, 'instance'):
        get_detector_instance.instance = NoduleDetector(model_path)
    elif model_path and get_detector_instance.instance.model_path != model_path:
        # 如果提供了不同的模型路径，重新加载模型
        get_detector_instance.instance.load_model(model_path)
        
    return get_detector_instance.instance 