import os,cv2
import sys
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file, Response, make_response
from flask_cors import CORS
import logging
import json
import uuid
import threading
import zipfile
from PIL import Image
from datetime import datetime
import io
import struct
from deploy.backend.dataclass.CTData import CTData
from util.dicom_util import is_dicom_file
from io import BytesIO
import matplotlib
# 设置matplotlib后端为非交互式
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者你系统中的其他中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
import matplotlib.pyplot as plt

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 导入检测器模块
from detector import get_detector_instance, STATUS_COMPLETED, STATUS_ERROR

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 模型参数
CUBE_SIZE = 32
# 配置上传文件夹和模型文件夹
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'mhd', 'raw', 'nii', 'nii.gz', 'dcm', 'dicom', 'zip'}
# 会话数据存储 - 用于存储上传的文件和处理状态
SESSION_DATA = {}
# 创建Flask应用
app = Flask(__name__, static_folder='../frontend')
CORS(app)  # 启用跨域
# 配置应用
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB限制

# 设置默认模型路径
DEFAULT_MODEL_PATH = os.path.join(MODEL_FOLDER, 'c3d_nodule_detect.pth')
# 初始化检测器实例
detector = get_detector_instance(DEFAULT_MODEL_PATH)


# 添加检测完成回调函数
def on_detection_completed(session_id, results):
    """检测完成后的回调函数，更新会话状态"""
    logger.info(f"检测完成回调: 会话 {session_id}")
    if session_id in SESSION_DATA:
        # 更新会话状态为completed
        SESSION_DATA[session_id]['status'] = 'completed'
        SESSION_DATA[session_id]['results'] = results
        SESSION_DATA[session_id]['progress'] = 100
        SESSION_DATA[session_id]['message'] = f"检测完成，发现 {len(results.get('nodules', []))} 个结节"

        # 更新检测器中的状态
        detector.update_session_state(session_id, SESSION_DATA[session_id])

        # 生成结节图像
        try:
            nodules = results.get('nodules', [])
            if nodules:
                logger.info(f"检测完成，开始为 {len(nodules)} 个结节生成图像")
                lung_seg_path = os.path.join(UPLOAD_FOLDER, session_id, 'lung_seg.npy')
                if os.path.exists(lung_seg_path):
                    lung_img = np.load(lung_seg_path)
                    if lung_img is not None and lung_img.size > 0:
                        logger.info(f"成功加载肺部分割数据，形状: {lung_img.shape}")
                        success = generate_nodule_images(session_id, nodules, lung_img)
                        if success:
                            logger.info(f"结节图像生成完成")
                        else:
                            logger.error(f"结节图像生成失败")
                    else:
                        logger.error(f"肺部分割数据为空或无效")
                else:
                    logger.error(f"肺部分割数据文件不存在: {lung_seg_path}")
            else:
                logger.info(f"无结节数据，跳过图像生成")
        except Exception as e:
            logger.error(f"生成结节图像时出错: {str(e)}", exc_info=True)


# 检查检测器是否支持回调并添加
if hasattr(detector, 'set_completion_callback'):
    detector.set_completion_callback(on_detection_completed)
else:
    logger.warning("检测器不支持完成回调，状态更新可能不正确")

# 添加兼容性方法，确保detector支持update_session_state
if not hasattr(detector, 'update_session_state'):
    def update_session_state(session_id, state):
        """
        更新会话状态的兼容性方法
        确保session_states字典存在并更新状态
        """
        if not hasattr(detector, 'session_states'):
            detector.session_states = {}

        if not hasattr(detector, 'session_locks'):
            detector.session_locks = {}

        # 创建会话锁（如果不存在）
        if session_id not in detector.session_locks:
            detector.session_locks[session_id] = threading.Lock()

        # 更新会话状态
        with detector.session_locks.get(session_id, threading.Lock()):
            detector.session_states[session_id] = state.copy()  # 使用副本避免引用问题

        logger.info(f"更新会话 {session_id} 的状态: {state['status']}")


    # 将方法添加到检测器对象
    detector.update_session_state = update_session_state

    # 如果get_session_state方法也不存在，添加它
    if not hasattr(detector, 'get_session_state'):
        def get_session_state(session_id):
            """获取会话状态的兼容性方法"""
            if not hasattr(detector, 'session_states'):
                detector.session_states = {}

            return detector.session_states.get(session_id)


        detector.get_session_state = get_session_state


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """提供前端页面"""
    return send_from_directory('../frontend', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    """提供静态文件"""
    return send_from_directory('../frontend', path)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    处理CT文件上传
    支持上传包含DICOM或MHD/RAW文件的压缩包
    自动解压并检测文件类型
    """
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有找到文件'}), 400

        file = request.files['file']

        # 如果用户没有选择文件
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'}), 400

        # 检查文件是否为zip
        if not file.filename.lower().endswith('.zip'):
            return jsonify({'success': False, 'error': '请上传ZIP格式的文件'}), 400

        # 创建新的会话ID
        session_id = str(uuid.uuid4())

        # 创建会话目录
        session_dir = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_dir, exist_ok=True)

        # 保存压缩文件
        zip_path = os.path.join(session_dir, 'upload.zip')
        file.save(zip_path)
        app.logger.info(f"保存压缩文件到 {zip_path}")

        # 解压文件
        extract_dir = os.path.join(session_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        zip_filename = os.path.splitext(os.path.basename(file.filename))[0]
        real_extracted_path = os.path.join(extract_dir, zip_filename)
        print("真实的加压缩文件目录是\t", real_extracted_path)
        app.logger.info(f"解压文件到 {real_extracted_path}")
        # 检测文件类型
        file_type, file_paths = detect_file_type(real_extracted_path)
        print("检测到的类型\t", file_type)
        print("检测到的文件列表\t", file_paths)
        if not file_type:
            return jsonify({'success': False, 'error': '压缩包中未找到支持的CT数据文件(DICOM或MHD/RAW)'}), 400

        # 记录会话信息
        session_info = {
            'id': session_id,
            'timestamp': datetime.now().isoformat(),
            'file_type': file_type,
            'files': file_paths,
            'extract_dir': real_extracted_path,
            'status': 'uploaded'
        }

        # 可选：保存患者ID
        if 'patient_id' in request.form:
            session_info['patient_id'] = request.form['patient_id']

        # 保存会话信息到文件
        session_info_path = os.path.join(session_dir, 'session_info.json')
        with open(session_info_path, 'w') as f:
            json.dump(session_info, f)

        # 将会话ID添加到全局会话字典
        SESSION_DATA[session_id] = {
            'status': 'uploaded',
            'progress': 0,
            'message': f'已上传并解压{file_type}格式CT数据，等待检测',
            'file_type': file_type,
            'files': file_paths
        }

        # 如果配置了自动处理，启动预处理任务
        auto_detect = app.config.get('AUTO_DETECT', False)
        if auto_detect:
            # 启动异步任务进行预处理
            threading.Thread(target=start_preprocessing, args=(session_id,)).start()
            app.logger.info(f"启动自动预处理任务，会话ID: {session_id}")

        # 返回成功响应
        return jsonify({
            'success': True,
            'message': f'文件上传成功，检测到{file_type}格式CT数据',
            'session_id': session_id,
            'file_type': file_type,
            'auto_detect': auto_detect
        })

    except Exception as e:
        app.logger.error(f"文件上传失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f"文件上传失败: {str(e)}"
        }), 400


def detect_file_type(directory):
    """
    检测目录中的CT数据类型
    支持DICOM和MHD/RAW格式
    返回: (文件类型, 文件路径列表)
    """
    # 递归查找所有文件
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    print("检查的文件夹是\t", directory)
    print(file_list)
    # 检查是否有MHD文件
    mhd_files = [f for f in file_list if f.lower().endswith('.mhd')]
    if mhd_files:
        # 检查对应的RAW文件是否存在
        for mhd_file in mhd_files:
            # 获取对应的RAW文件名（替换扩展名）
            raw_file = os.path.splitext(mhd_file)[0] + '.raw'
            # 忽略大小写比较
            if any(f.lower() == raw_file.lower() for f in file_list):
                # 找到MHD和对应的RAW文件
                return 'MHD/RAW', [mhd_file]

    # 检查是否有DICOM文件
    dicom_files = [f for f in file_list if f.lower().endswith(('.dcm', '.dicom')) or
                   (os.path.isfile(f) and is_dicom_file(f))]

    if dicom_files:
        return 'DICOM', dicom_files

    # 如果没有找到支持的文件类型
    return None, []


def start_preprocessing(session_id):
    """
    启动CT数据预处理
    将CT数据转换为肺部分割数据
    """
    try:
        # 获取会话状态
        if session_id not in SESSION_DATA:
            logger.error(f"会话 {session_id} 不存在")
            return

        state = SESSION_DATA[session_id]
        state['status'] = 'preprocessing'
        state['progress'] = 0
        state['message'] = '正在开始预处理...'

        # 同步会话状态到检测器
        detector.update_session_state(session_id, state)

        # 获取会话目录
        session_dir = os.path.join(UPLOAD_FOLDER, session_id)
        # 加载会话信息
        with open(os.path.join(session_dir, 'session_info.json'), 'r') as f:
            session_info = json.load(f)
        print("加载检测的信息是\n", session_info)

        file_type = session_info['file_type']
        extract_dir = session_info['extract_dir']  # 使用保存的解压目录

        # 更新状态
        state['progress'] = 10
        state['message'] = f'正在加载{file_type}格式CT数据...'
        # 同步状态更新
        detector.update_session_state(session_id, state)

        # 根据文件类型进行处理
        ct_data = None

        print("使用的解压目录是\n", extract_dir)

        if file_type == 'MHD/RAW':
            # 使用MHD文件的目录路径，而不是单个文件路径
            ct_data = CTData.from_mhd(session_info['files'][0])

        elif file_type == 'DICOM':
            # 对于DICOM，使用包含所有DICOM文件的目录路径
            ct_data = CTData.from_dicom(extract_dir)

        # 检查是否成功加载
        if ct_data is None:
            state['status'] = 'error'
            state['message'] = '加载CT数据失败'
            SESSION_DATA[session_id] = state
            detector.update_session_state(session_id, state)
            logger.error(f"加载CT数据失败，会话ID: {session_id}")
            return

        # 更新状态
        state['progress'] = 30
        state['message'] = '正在进行肺部分割...'
        # 同步状态更新
        detector.update_session_state(session_id, state)

        ct_data.resample_pixel()
        ct_data.filter_lung_img_mask()
        # 执行肺部分割
        lung_seg = ct_data.lung_seg_img

        # 保存肺部分割结果
        lung_seg_path = os.path.join(session_dir, 'lung_seg.npy')
        np.save(lung_seg_path, lung_seg)

        # 更新状态 - 肺部分割完成
        state['progress'] = 40
        state['message'] = '肺部分割完成，准备检测'
        state['status'] = 'preprocessed'  # 在肺部分割完成时就设置为preprocessed
        # 添加肺部分割路径
        state['lung_seg_path'] = lung_seg_path
        SESSION_DATA[session_id] = state
        detector.update_session_state(session_id, state)

        # 自动启动检测 - 传递正确的路径
        if file_type == 'MHD/RAW':
            # 对于MHD使用文件路径
            detector.detect(session_info['files'][0], session_id, patient_id=None)
        else:
            # 对于DICOM使用目录路径
            detector.detect(extract_dir, session_id, patient_id=None)

        # 在保存肺部分割数据的时候，同时保存所有切片图像
        save_lung_segmentation_slices(session_id, lung_seg)

    except Exception as e:
        logger.error(f"预处理失败: {str(e)}", exc_info=True)

        # 更新状态为错误
        if session_id in SESSION_DATA:
            SESSION_DATA[session_id]['status'] = 'error'
            SESSION_DATA[session_id]['message'] = f'预处理失败: {str(e)}'
            detector.update_session_state(session_id, SESSION_DATA[session_id])


@app.route('/api/detect', methods=['POST'])
def start_detection():
    """启动检测过程"""
    try:
        # 获取请求参数
        data = request.json
        session_id = data.get('session_id')
        patient_id = data.get('patient_id', session_id)

        if not session_id:
            return jsonify({'success': False, 'error': '缺少会话ID'}), 400

        # 获取会话文件夹和文件
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        if not os.path.exists(session_folder):
            return jsonify({'success': False, 'error': f'会话 {session_id} 不存在'}), 404

        # 读取session_info.json获取正确的文件路径
        session_info_path = os.path.join(session_folder, 'session_info.json')
        if not os.path.exists(session_info_path):
            return jsonify({'success': False, 'error': '会话信息文件不存在'}), 404

        # 加载会话信息
        with open(session_info_path, 'r') as f:
            session_info = json.load(f)

        # 根据文件类型选择正确的路径
        file_type = session_info.get('file_type')
        extract_dir = session_info.get('extract_dir')

        if not file_type or not extract_dir:
            return jsonify({'success': False, 'error': '会话信息不完整'}), 400

        # 使用与start_preprocessing相同的逻辑
        if file_type == 'MHD/RAW':
            # 对于MHD使用文件路径
            file_path = session_info['files'][0]
        else:
            # 对于DICOM使用目录路径
            file_path = extract_dir

        logger.info(f"使用文件路径 {file_path} 开始检测")

        # 启动检测
        if not detector.detect(file_path, session_id, patient_id):
            return jsonify({'success': False, 'error': '检测启动失败，请确保模型已正确加载'}), 500

        # 返回会话ID，客户端将使用此ID轮询进度
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': '检测已启动'
        })

    except Exception as e:
        logger.error(f"启动检测错误: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/progress/<session_id>', methods=['GET'])
def get_progress(session_id):
    """获取检测进度"""
    try:
        # 获取会话状态 - 首先尝试从检测器获取
        state = detector.get_session_state(session_id)

        # 如果检测器中没有，则尝试从SESSION_DATA获取
        if not state and session_id in SESSION_DATA:
            state = SESSION_DATA[session_id]
            logger.info(f"从SESSION_DATA中获取会话 {session_id} 的状态")

        # 如果两处都没有，则返回错误
        if not state:
            return jsonify({'success': False, 'error': f'会话 {session_id} 不存在'}), 404

        # 创建响应
        response = {
            'success': True,
            'session_id': session_id,
            'status': state['status'],
            'progress': state['progress'],
            'message': state['message'],
            'started_at': state.get('started_at'),
            'completed_at': state.get('completed_at')
        }

        # 如果检测完成，添加结节信息
        if state['status'] == STATUS_COMPLETED and 'nodules' in state:
            response['nodules_count'] = len(state['nodules'])

        # 如果发生错误，添加错误信息
        if state['status'] == STATUS_ERROR and 'error' in state:
            response['error'] = state['error']

        return jsonify(response)

    except Exception as e:
        logger.error(f"获取进度错误: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/results/<session_id>', methods=['GET'])
def get_results(session_id):
    """获取检测结果"""
    try:
        # 获取会话状态
        state = detector.get_session_state(session_id)
        if not state:
            return jsonify({'success': False, 'error': f'会话 {session_id} 不存在'}), 404
        print("结节api/result 返回状态", state['status'])
        # 检查检测是否完成
        if state['status'] != STATUS_COMPLETED:
            return jsonify({
                'success': False,
                'error': '检测尚未完成',
                'status': state['status'],
                'progress': state['progress']
            }), 400

        # 检查是否有结节数据
        if 'nodules' not in state or not state['nodules']:
            print('nodule 不在最终结果里面')
            return jsonify({
                'success': True,
                'session_id': session_id,
                'nodules_count': 0,
                'nodules': [],
                'message': '未检测到结节'
            })

        # 返回结节信息
        # 注意：为了减少传输数据量，这里不返回完整的立方体数据
        nodules_info = []
        for nodule in state['nodules']:
            print('尝试在返回结果增加每个结节信息')
            # 创建不包含立方体数据的结节信息副本
            nodule_info = {
                'id': nodule['id'],
                'voxel_coords': nodule['voxel_coords'],
                'world_coords': nodule['world_coords'],
                'diameter_mm': nodule['diameter_mm'],
                'probability': nodule['probability']
            }
            nodules_info.append(nodule_info)
            lung_seg_path = os.path.join(UPLOAD_FOLDER, session_id, 'lung_seg.npy')
            lung_seg_img = np.load(lung_seg_path)
            print('开始保存 结节的bbox...')
            save_slice_box(session_id, lung_seg_img, nodule['voxel_coords'], radius=32)
            print('保存一个结节的bbox 完成。。。')

        return jsonify({
            'success': True,
            'session_id': session_id,
            'nodules_count': len(nodules_info),
            'nodules': nodules_info,
            'message': f'检测完成，共发现 {len(nodules_info)} 个结节'
        })

    except Exception as e:
        logger.error(f"获取结果错误: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/nodule/<session_id>/<int:nodule_id>', methods=['GET'])
def get_nodule_data(session_id, nodule_id):
    try:
        # 检查会话是否存在
        if session_id not in SESSION_DATA:
            return jsonify({"success": False, "error": "会话不存在"}), 404

        # 获取会话数据
        session = SESSION_DATA[session_id]
        print('返回 api nodule 结果状态\t', session['status'])
        # 检查是否已完成检测
        if session['status'] not in ['preprocessed', 'completed']:
            return jsonify({"success": False, "error": "CT数据检测未完成"}), 400
        # 获取检测结果
        results = session.get('results', {})
        nodules = results.get('nodules', [])
        # 查找指定的结节
        target_nodule = None
        for nodule in nodules:
            if nodule.get('id') == nodule_id:
                target_nodule = nodule
                break
        if not target_nodule:
            return jsonify({"success": False, "error": "找不到指定的结节"}), 404
        # 返回结节数据
        return jsonify({
            "success": True,
            "nodule": target_nodule
        })

    except Exception as e:
        app.logger.error(f"获取结节数据时出错: {str(e)}")
        return jsonify({"success": False, "error": f"获取结节数据时出错: {str(e)}"}), 500


# 添加新函数: 在检测完成后为所有结节生成并保存图像
def generate_nodule_images(session_id, nodules, lung_img):
    """
    为所有结节生成并保存切片图像

    Args:
        session_id: 会话ID
        nodules: 结节列表
        lung_img: 肺部分割数据
    """
    try:
        # 确保结节图像目录存在
        nodule_images_dir = os.path.join(UPLOAD_FOLDER, session_id, 'nodule_images')
        os.makedirs(nodule_images_dir, exist_ok=True)
        app.logger.info(f"开始为 {len(nodules)} 个结节生成切片图像")

        # 打印肺部图像数据统计信息以进行调试
        app.logger.info(f"肺部图像数据统计: 形状={lung_img.shape}, 类型={lung_img.dtype}, "
                        f"最小值={lung_img.min()}, 最大值={lung_img.max()}, "
                        f"平均值={lung_img.mean()}, 中位数={np.median(lung_img)}")

        # 检查肺部图像数据是否全为0或接近于0
        if np.max(lung_img) < 0.01:
            app.logger.warning("警告: 肺部图像数据几乎全为零，可能导致结节图像显示为黑色")
            # 尝试自动增强对比度 - 将较小的值映射到0，将较大的值映射到255
            lung_img = (lung_img * 100).clip(0, 1)
            app.logger.info(
                f"增强对比度后统计: 最小值={lung_img.min()}, 最大值={lung_img.max()}, 平均值={lung_img.mean()}")

        # 处理每个结节
        for nodule in nodules:
            nodule_id = nodule.get('id')
            # 获取结节中心坐标
            voxel_coords = nodule.get('voxel_coords', [0, 0, 0])
            app.logger.info(f"处理结节 {nodule_id}, 原始坐标 (xyz格式): {voxel_coords}")
            # 将坐标转换为整数，注意从xyz转换为zyx顺序
            x, y, z = [int(coord) for coord in voxel_coords]
            # 交换坐标顺序为z,y,x以匹配lung_img的顺序
            center_voxel_zyx = [z, y, x]
            app.logger.info(f"转换后坐标 (zyx格式): {center_voxel_zyx}")
            # 检查坐标是否在肺部图像范围内
            if (center_voxel_zyx[0] < 0 or center_voxel_zyx[0] >= lung_img.shape[0] or
                    center_voxel_zyx[1] < 0 or center_voxel_zyx[1] >= lung_img.shape[1] or
                    center_voxel_zyx[2] < 0 or center_voxel_zyx[2] >= lung_img.shape[2]):
                app.logger.error(
                    f"结节 {nodule_id} 坐标超出肺部图像范围: {center_voxel_zyx}, 肺部图像尺寸: {lung_img.shape}")
                continue
            # 定义立方体的半尺寸
            half_size = 16  # CUBE_SIZE/2 = 16
            # 提取立方体数据
            z_min = max(0, center_voxel_zyx[0] - half_size)
            y_min = max(0, center_voxel_zyx[1] - half_size)
            x_min = max(0, center_voxel_zyx[2] - half_size)

            z_max = min(lung_img.shape[0], center_voxel_zyx[0] + half_size)
            y_max = min(lung_img.shape[1], center_voxel_zyx[1] + half_size)
            x_max = min(lung_img.shape[2], center_voxel_zyx[2] + half_size)

            # 检查提取的范围是否有效
            if z_min >= z_max or y_min >= y_max or x_min >= x_max:
                app.logger.error(
                    f"结节 {nodule_id} 提取的范围无效: z({z_min}-{z_max}), y({y_min}-{y_max}), x({x_min}-{x_max})")
                continue

            # 提取子体积
            cube = lung_img[z_min:z_max, y_min:y_max, x_min:x_max]
            # 检查立方体数据是否为空
            if cube.size == 0:
                app.logger.error(f"结节 {nodule_id} 提取的立方体数据为空")
                continue

            app.logger.info(f"成功提取结节 {nodule_id} 立方体数据, 形状: {cube.shape}")
            app.logger.info(f"立方体数据统计: 最小值={cube.min()}, 最大值={cube.max()}, 平均值={cube.mean()}")

            # 为每个平面类型生成图像
            for plane_type in ['axial', 'coronal', 'sagittal']:
                try:
                    # 根据平面类型获取中心切片
                    if plane_type == 'axial':  # Z轴切片
                        if cube.shape[0] == 0:
                            app.logger.error(f"结节 {nodule_id} 在轴向切片上的维度为0")
                            continue
                        center_index = min(cube.shape[0] // 2, cube.shape[0] - 1)
                        slice_data = cube[center_index, :, :]
                    elif plane_type == 'coronal':  # Y轴切片
                        if cube.shape[1] == 0:
                            app.logger.error(f"结节 {nodule_id} 在冠状切片上的维度为0")
                            continue
                        center_index = min(cube.shape[1] // 2, cube.shape[1] - 1)
                        slice_data = cube[:, center_index, :]
                    elif plane_type == 'sagittal':  # X轴切片
                        if cube.shape[2] == 0:
                            app.logger.error(f"结节 {nodule_id} 在矢状切片上的维度为0")
                            continue
                        center_index = min(cube.shape[2] // 2, cube.shape[2] - 1)
                        slice_data = cube[:, :, center_index]

                    # 检查切片数据是否为空
                    if slice_data.size == 0:
                        app.logger.error(f"结节 {nodule_id} 在{plane_type}切片上的数据为空")
                        continue

                    app.logger.info(
                        f"切片数据统计: 最小值={slice_data.min()}, 最大值={slice_data.max()}, 平均值={slice_data.mean()}")

                    # 增强对比度
                    # 如果所有值都很小（接近0），进行强化对比度处理
                    if slice_data.max() < 0.1:
                        # 将数据放大100倍以增强可见度
                        slice_data = np.clip(slice_data * 100, 0, 1)
                        app.logger.info(f"对比度增强后: 最小值={slice_data.min()}, 最大值={slice_data.max()}")
                    # 归一化数据
                    if np.all(slice_data == 0):  # 检查是否全为0
                        app.logger.warning(f"结节切片数据全为0")
                        normalized_slice = np.zeros(slice_data.shape, dtype=np.uint8)
                    elif slice_data.min() >= 0 and slice_data.max() <= 1:
                        # 0-1范围数据，转为0-255
                        normalized_slice = (slice_data * 255).astype(np.uint8)
                    elif slice_data.min() >= 0 and slice_data.max() <= 255:
                        normalized_slice = slice_data.astype(np.uint8)
                    else:
                        min_val = slice_data.min()
                        max_val = slice_data.max()
                        if max_val > min_val:
                            normalized_slice = ((slice_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        else:
                            normalized_slice = np.zeros_like(slice_data, dtype=np.uint8)

                    app.logger.info(
                        f"归一化后数据统计: 最小值={normalized_slice.min()}, 最大值={normalized_slice.max()}, 平均值={normalized_slice.mean()}")

                    # 应用额外的图像增强（如果需要）
                    if normalized_slice.max() < 50:  # 如果最大值仍然很小
                        # 应用CLAHE（对比度受限的自适应直方图均衡化）
                        app.logger.info("应用CLAHE增强对比度")
                        try:
                            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                            normalized_slice = clahe.apply(normalized_slice)
                        except ImportError:
                            # 如果没有cv2，使用简单的线性拉伸
                            normalized_slice = np.clip(normalized_slice * 5, 0, 255).astype(np.uint8)

                    # 创建图像
                    plt.figure(figsize=(6, 6))
                    plt.imshow(normalized_slice, cmap='gray', vmin=0, vmax=255)

                    # 添加标题
                    view_names = {
                        'axial': '轴向视图 (XY)',
                        'coronal': '冠状视图 (XZ)',
                        'sagittal': '矢状视图 (YZ)'
                    }
                    plt.title(f"结节 {nodule_id} - {view_names.get(plane_type, plane_type)}")

                    # 添加中心标记
                    center_y, center_x = normalized_slice.shape[0] // 2, normalized_slice.shape[1] // 2
                    plt.plot(center_x, center_y, 'r+', markersize=10)

                    # 添加直径标记
                    diameter = nodule.get('diameter_mm', 10)
                    radius_pixels = diameter / 2
                    circle = plt.Circle((center_x, center_y), radius_pixels,
                                        color='r', fill=False, linestyle='--')
                    plt.gca().add_patch(circle)

                    # 添加颜色条以直观显示像素值
                    plt.colorbar(label='像素值')

                    # 关闭坐标轴
                    plt.axis('off')
                    plt.tight_layout()

                    # 保存图像到文件
                    img_filename = f"nodule_{nodule_id}_{plane_type}.png"
                    img_path = os.path.join(nodule_images_dir, img_filename)

                    # 保存图像
                    plt.savefig(img_path, format='png', dpi=100)
                    plt.close('all')  # 确保关闭所有图形

                    app.logger.info(f"已保存结节图像: {img_path}")
                except Exception as slice_error:
                    app.logger.error(f"生成结节 {nodule_id} 的 {plane_type} 图像时出错: {str(slice_error)}",
                                     exc_info=True)

        app.logger.info(f"结节图像生成完成")
        return True

    except Exception as e:
        app.logger.error(f"生成结节图像时出错: {str(e)}", exc_info=True)
        return False


# 修改检测完成时的代码，在检测完成后生成结节图像
def update_session_state(session_id, state):
    """更新会话状态"""
    if session_id in SESSION_DATA:
        SESSION_DATA[session_id]['status'] = state

        # 如果检测已完成，生成所有结节的图像
        if state == 'completed' and 'results' in SESSION_DATA[session_id]:
            nodules = SESSION_DATA[session_id]['results'].get('nodules', [])
            if nodules:
                app.logger.info(f"检测完成，开始生成结节图像")
                # 加载肺部分割数据用于生成图像
                try:
                    lung_seg_path = os.path.join(UPLOAD_FOLDER, session_id, 'lung_seg.npy')
                    if os.path.exists(lung_seg_path):
                        lung_img = np.load(lung_seg_path)
                        # 生成结节图像
                        generate_nodule_images(session_id, nodules, lung_img)
                    else:
                        app.logger.error(f"无法生成结节图像: 肺部分割数据不存在")
                except Exception as e:
                    app.logger.error(f"生成结节图像时出错: {str(e)}", exc_info=True)
    else:
        # 创建新会话
        SESSION_DATA[session_id] = {
            'status': state,
            'progress': 0
        }

def save_slice_box(session_id,lung_seg,voxel_coords,radius = 32):
    print("传入的体素是\t", voxel_coords)
    x, y, z = [int(coord) for coord in voxel_coords]
    from_z,end_z = z- int(radius/2), z + int(radius/2)
    from_x,end_x = x - int(radius/2), x + int(radius/2)
    from_y, end_y = y - int(radius / 2), y + int(radius / 2)
    slices_dir = os.path.join(UPLOAD_FOLDER, session_id, 'lung_slices')
    for z_index in range(from_z, end_z + 1):
        slice_data = lung_seg[z_index, :, :]
        # 归一化数据
        normalized_slice = (slice_data * 255).astype(np.uint8)
        # 在图像上绘制红色边界框
        # 将灰度图转换为彩色图像，以便绘制彩色边界框
        color_slice = cv2.cvtColor(normalized_slice, cv2.COLOR_GRAY2BGR)
        # 确保边界框坐标在图像范围内
        from_y_safe = max(0, from_y)
        end_y_safe = min(normalized_slice.shape[0], end_y)
        from_x_safe = max(0, from_x)
        end_x_safe = min(normalized_slice.shape[1], end_x)
        # 绘制红色边界框，线宽为2
        cv2.rectangle(color_slice, (from_x_safe, from_y_safe), (end_x_safe, end_y_safe), (0, 0, 255), 2)
        # 使用绘制了边界框的彩色图像替换原来的灰度图像
        normalized_slice = color_slice
        # 创建图像文件名和路径
        img_filename = f"z_slice_{z_index:04d}.png"
        img_path = os.path.join(slices_dir, img_filename)
        # 保存图像
        cv2.imwrite(img_path, normalized_slice)
        print("保存一个结节的 bbox 到\t", img_path)

# 在保存肺部分割数据的时候，同时保存所有切片图像
def save_lung_segmentation_slices(session_id, lung_seg):
    """
    保存肺部分割的所有切片图像

    Args:
        session_id: 会话ID
        lung_seg: 肺部分割数据 (3D数组)
    """
    try:
        # 创建切片图像目录
        slices_dir = os.path.join(UPLOAD_FOLDER, session_id, 'lung_slices')
        os.makedirs(slices_dir, exist_ok=True)
        app.logger.info(f"开始保存肺部切片图像，形状: {lung_seg.shape}")
        # 获取各轴切片数量
        z_slices = lung_seg.shape[0]
        y_slices = lung_seg.shape[1]
        x_slices = lung_seg.shape[2]

        # 创建索引文件，用于前端加载
        slice_info = {
            "dimensions": lung_seg.shape,
            "z_slices": z_slices,
            "y_slices": y_slices,
            "x_slices": x_slices,
            "z_axis": [],
            "y_axis": [],
            "x_axis": []
        }
        # 保存Z轴切片 (横断面)
        app.logger.info(f"正在保存Z轴切片，共{z_slices}个...")
        for z in range(z_slices):
            # 获取切片
            slice_data = lung_seg[z, :, :]
            # 归一化数据
            normalized_slice = (slice_data * 255).astype(np.uint8)
            # 创建图像文件名和路径
            img_filename = f"z_slice_{z:04d}.png"
            img_path = os.path.join(slices_dir, img_filename)
            # 保存图像
            cv2.imwrite(img_path, normalized_slice)
            # 添加切片信息到索引
            slice_info["z_axis"].append({
                "index": z,
                "filename": img_filename,
                "path": f"/api/lung_slice/{session_id}/z/{z}"
            })
            
            # 每保存10个切片打印一次进度
            if z % 10 == 0 or z == z_slices - 1:
                app.logger.info(f"已保存Z轴切片 {z + 1}/{z_slices}")
        # 保存索引文件
        index_path = os.path.join(slices_dir, 'slices_index.json')
        with open(index_path, 'w') as f:
            json.dump(slice_info, f)

        app.logger.info(f"肺部切片图像保存完成，共 {z_slices} 个Z轴切片, {y_slices} 个Y轴切片, {x_slices} 个X轴切片")
        return True

    except Exception as e:
        app.logger.error(f"保存肺部切片图像时出错: {str(e)}", exc_info=True)
        return False

# 添加API端点获取肺部切片图像
@app.route('/api/lung_slice/<session_id>/z/<int:slice_index>', methods=['GET'])
def get_lung_z_slice(session_id, slice_index):
    """获取肺部的Z轴切片图像"""
    try:
        app.logger.info(f"请求Z轴切片 - 会话ID: {session_id}, 切片索引: {slice_index}")
        # 检查会话是否存在
        if session_id not in SESSION_DATA:
            app.logger.error(f"会话不存在: {session_id}")
            return jsonify({"success": False, "error": "会话不存在"}), 404
        
        # 图像文件路径
        img_filename = f"z_slice_{slice_index:04d}.png"
        img_path = os.path.join(UPLOAD_FOLDER, session_id, 'lung_slices', img_filename)
        app.logger.info(f"切片图像路径: {img_path}")
        
        # 检查文件是否存在
        if os.path.exists(img_path):
            app.logger.info(f"切片图像文件已存在，直接返回: {img_path}")
            return send_file(img_path, mimetype='image/png')

    except Exception as e:
        app.logger.error(f"获取肺部Z轴切片时出错: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"获取肺部Z轴切片时出错: {str(e)}"}), 500

# 添加API端点获取肺部切片信息
@app.route('/api/lung_slices_info/<session_id>', methods=['GET'])
def get_lung_slices_info(session_id):
    """获取肺部切片信息"""
    print('/api/lung_slices_info 会话内容\t', SESSION_DATA, session_id)
    # 检查会话是否存在
    if session_id not in SESSION_DATA:
        return jsonify({"success": False, "error": "会话不存在"}), 404
    # 索引文件路径
    index_path = os.path.join(UPLOAD_FOLDER, session_id, 'lung_slices', 'slices_index.json')
    print('索引文件路径\t', index_path)
    # 检查索引文件是否存在
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            slices_info = json.load(f)
        return jsonify({"success": True, "slices_info": slices_info})
    print('索引文件不存在，重新加载 npy 文件\t')
    # 如果索引文件不存在，尝试从肺部分割数据创建基本信息
    lung_seg_path = os.path.join(UPLOAD_FOLDER, session_id, 'lung_seg.npy')
    if not os.path.exists(lung_seg_path):
        return jsonify({"success": False, "error": "肺部分割数据不存在"}), 404

    # 加载数据并创建基本信息
    lung_seg = np.load(lung_seg_path)

    # 获取各轴切片数量
    z_slices = lung_seg.shape[0]
    y_slices = lung_seg.shape[1]
    x_slices = lung_seg.shape[2]

    # 创建与save_lung_segmentation_slices函数相同格式的切片信息
    slices_info = {
        "dimensions": lung_seg.shape,
        "z_slices": z_slices,
        "y_slices": y_slices,
        "x_slices": x_slices,
        "z_axis": [],
        "y_axis": [],
        "x_axis": []
    }
    # 添加Z轴切片信息
    for z in range(z_slices):
        slices_info["z_axis"].append({
            "index": z,
            "filename": f"z_slice_{z:04d}.png",
            "path": f"/api/lung_slice/{session_id}/z/{z}"
        })
    return jsonify({"success": True, "slices_info": slices_info})

# 启动服务器
if __name__ == '__main__':
    # 确保目录存在
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    # 启用自动检测
    app.config['AUTO_DETECT'] = True
    # 启动应用
    app.run(host='0.0.0.0', port=5000, debug=True)
