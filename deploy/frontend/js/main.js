/**
 * CT 肺结节检测系统 主JS文件
 */

// 全局变量
let currentSessionId = null;
let currentSliceIndex = 0;
let maxSliceIndex = 0;
let lungSegmentationLoaded = false;
let progressInterval = null;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('CT肺结节检测系统初始化');
    
    // 初始化事件监听器
    initializeEventListeners();
    // 初始化步骤UI
    initializeSteps();
    // 设置文件上传
    setupFileUpload();
    // 检查是否有会话ID保存在sessionStorage中
    const savedSessionId = sessionStorage.getItem('currentSessionId');
});

// 初始化步骤显示
function initializeSteps() {
    // 初始状态下激活上传步骤
    updateUIState('initial');
}

// 设置文件上传
function setupFileUpload() {
    const fileInput = document.getElementById('ct-file');
    const selectFileBtn = document.getElementById('select-file-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const fileInfo = document.getElementById('file-info');
    
    // 选择文件按钮点击事件
    selectFileBtn.addEventListener('click', function() {
        fileInput.click();
    });
    
    // 文件选择变化事件
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            const file = this.files[0];
            
            // 检查文件类型
            const fileExt = getFileExtension(file.name).toLowerCase();
            if (!['zip', 'mhd', 'dcm', 'dicom', 'nii', 'nii.gz'].includes(fileExt)) {
                showMessage('请上传支持的格式: MHD, DICOM, NIfTI 或 ZIP文件', 'error');
                fileInfo.textContent = '请选择支持的CT文件格式';
                uploadBtn.disabled = true;
                return;
            }
            
            fileInfo.textContent = `已选择: ${file.name} (${formatFileSize(file.size)})`;
            uploadBtn.disabled = false;
        } else {
            fileInfo.textContent = '请选择CT文件';
            uploadBtn.disabled = true;
        }
    });
    
    // 上传按钮点击事件
    uploadBtn.addEventListener('click', function() {
        if (fileInput.files.length === 0) {
            showMessage('请先选择文件', 'error');
            return;
        }
        
        // 上传文件
        uploadFile(fileInput.files[0]);
    });
}

// 初始化事件监听器
function initializeEventListeners() {
    // 重置视图按钮
    document.getElementById('reset-view-btn').addEventListener('click', function() {
        if (maxSliceIndex > 0) {
            // 将切片重置到中间位置
            currentSliceIndex = Math.floor(maxSliceIndex / 2);
            loadSlice(currentSliceIndex);
            showMessage('已重置视图至中间切片', 'info');
        } else {
            showMessage('无法重置视图，未加载数据', 'warning');
        }
    });

    // 开始检测按钮
    document.getElementById('start-detection-btn').addEventListener('click', startDetection);
    // 结节列表点击事件委托
    document.getElementById('nodule-list').addEventListener('click', function(e) {
        if (e.target.closest('.nodule-item')) {
            const noduleItem = e.target.closest('.nodule-item');
            const noduleId = noduleItem.dataset.id;
            // 移除其他项目的选中状态
            document.querySelectorAll('.nodule-item').forEach(item => {
                item.classList.remove('selected');
            });
            
            // 添加选中状态
            noduleItem.classList.add('selected');
            
            // 加载结节详情
            console.log(`选中结节 ${noduleId}`);
            loadNoduleDetails(noduleId);
        }
    });
}

// 获取文件扩展名
function getFileExtension(filename) {
    return filename.split('.').pop();
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 上传文件
function uploadFile(file) {
    console.log('开始上传文件:', file.name);
    
    // 显示进度区域
    document.getElementById('upload-area').style.display = 'none';
    document.getElementById('progress-area').style.display = 'block';
    
    // 更新进度条初始状态
    updateProgress(10, '正在上传文件...');
    
    // 创建FormData对象
    const formData = new FormData();
    formData.append('file', file);
    
    // 添加患者ID (如果有)
    const patientId = document.getElementById('patient-id').value;
    if (patientId) {
        formData.append('patient_id', patientId);
        document.getElementById('patient-info').textContent = `患者ID: ${patientId}`;
    }
    
    // 发送上传请求
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log('服务器响应状态:', response.status);
        if (!response.ok) {
            throw new Error(`网络响应错误: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('上传响应:', data);
        if (data.success) {
            // 保存会话ID
            currentSessionId = data.session_id;
            console.log('保存会话ID:', currentSessionId);
            
            // 同时保存到sessionStorage
            sessionStorage.setItem('currentSessionId', currentSessionId);
            
            // 完成上传步骤
            updateUIState('uploaded');
            
            // 更新进度
            updateProgress(100, '文件上传成功');
            
            // 显示文件类型信息
            let typeInfo = data.file_type ? `检测到${data.file_type}格式CT数据` : '文件已上传';
            
            // 根据是否自动检测显示不同消息
            if (data.auto_detect) {
                document.getElementById('progress-message').textContent = `${typeInfo}，正在自动开始检测...`;
                document.getElementById('start-detection-btn').disabled = true;
                
                // 开始轮询检测进度
                startProgressPolling();
            } else {
                document.getElementById('progress-message').textContent = `${typeInfo}，可以开始检测`;
                document.getElementById('start-detection-btn').disabled = false;
            }
            
            // 显示成功消息
            showMessage('文件上传成功', 'success');
        } else {
            throw new Error(data.error || '上传失败');
        }
    })
    .catch(error => {
        console.error('文件上传错误:', error);
        updateUIState('initial');
        document.getElementById('progress-message').textContent = `上传失败: ${error.message}`;
        showMessage(`上传失败: ${error.message}`, 'error');
        
        // 重新显示上传区域
        setTimeout(() => {
            document.getElementById('progress-area').style.display = 'none';
            document.getElementById('upload-area').style.display = 'block';
        }, 3000);
    });
}

// 开始检测
async function startDetection() {
        if (!currentSessionId) {
            showMessage('请先上传CT文件', 'error');
            return;
        }
        
    try {
        // 更新UI状态
        updateUIState('detecting');
        // 清除之前的数据
        resetResults();
        // 显示消息
        showMessage('正在开始检测...', 'info');
        // 发送检测请求
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: currentSessionId
            })
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || '启动检测失败');
        }
        
        // 显示成功消息
        showMessage('检测已启动', 'success');
        
        // 开始轮询检测进度
        startProgressPolling();
        
    } catch (error) {
        console.error('启动检测时出错:', error);
        showMessage(`启动检测失败: ${error.message}`, 'error');
        updateUIState('uploaded');
    }
}

// 开始轮询检测进度
function startProgressPolling() {
    // 清除之前的轮询
    if (progressInterval) {
        clearInterval(progressInterval);
    }
    
    // 初始化进度条
    updateProgress(0, '正在初始化检测...');
    
    // 显示进度区域
    document.getElementById('progress-area').style.display = 'block';
    
    // 重置肺部分割加载状态
    lungSegmentationLoaded = false;
    
    // 开始轮询
    progressInterval = setInterval(async function() {
        try {
            if (!currentSessionId) {
                clearInterval(progressInterval);
                return;
            }
            
            const response = await fetch(`/api/progress/${currentSessionId}`);
            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.error || '获取进度失败');
            }
            
            // 更新进度条
            updateProgress(data.progress, data.message);
            
            // 当进度达到40%时，可以开始尝试加载肺部切片数据
            if (data.progress >= 40 && !lungSegmentationLoaded) {
                loadLungSegmentation();
            }
            
            // 如果状态为已完成或错误，停止轮询
            if (data.status === 'completed') {
                clearInterval(progressInterval);
                progressInterval = null;
                
                // 获取检测结果
                fetchResults();
            } else if (data.status === 'error') {
                clearInterval(progressInterval);
                progressInterval = null;
                showMessage(`检测失败: ${data.error || '未知错误'}`, 'error');
                updateUIState('uploaded');
            }
        } catch (error) {
            console.error('获取进度时出错:', error);
            showMessage(`获取进度失败: ${error.message}`, 'error');
        }
    }, 1000); // 每秒轮询一次
}

// 更新进度条
function updateProgress(progress, message) {
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const progressMessage = document.getElementById('progress-message');
    
    if (progressBar && progressText) {
        progressBar.style.width = `${progress}%`;
        progressText.textContent = `${progress}%`;
    }
    
    if (progressMessage && message) {
        progressMessage.textContent = message;
    }
}

// 加载肺部切片数据
function loadLungSegmentation() {
    console.log('加载肺部切片数据...');
    
    // 检查会话ID是否存在
    if (!currentSessionId) {
        console.error('无法加载肺部切片: 没有会话ID');
        return;
    }
    
    // 显示加载中消息
    document.getElementById('progress-message').textContent = '加载肺部切片数据...';
    
    // 加载切片信息
    fetch(`/api/lung_slices_info/${currentSessionId}`)
        .then(response => {
            console.log('请求lung sliced info 返回了什么\t', response);
            if (!response.ok) {
                throw new Error(`服务器返回错误状态: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (!data.success) {
                throw new Error(data.error || '获取切片信息失败');
            }
            
            console.log('获取到切片信息:', data.slices_info);
            
            // 设置全局变量
            maxSliceIndex = data.slices_info.z_slices - 1;
            // 将当前切片设置为中间位置
            currentSliceIndex = Math.floor(maxSliceIndex / 2);
            
            // 初始化CT查看器
            initCTViewer();
            
            // 加载初始切片
            loadSlice(currentSliceIndex);
            
            // 设置状态
            document.getElementById('progress-message').textContent = '肺部切片数据加载完成';
            lungSegmentationLoaded = true;
        })
        .catch(error => {
            console.error('获取肺部切片信息失败:', error);
            document.getElementById('progress-message').textContent = `加载失败: ${error.message}`;
            showMessage(`肺部切片数据加载失败: ${error.message}`, 'error');
        });
}

// 初始化CT查看器
function initCTViewer() {
    console.log('初始化CT查看器...');
    
    // 获取查看器容器
    const viewerContainer = document.getElementById('ct-viewer');
    if (!viewerContainer) {
        console.error('找不到CT查看器容器元素');
        return;
    }
    
    // 清空容器
    viewerContainer.innerHTML = '';
    
    // 创建CT切片视图 - 只显示Z轴切片
    const sliceView = document.createElement('div');
    sliceView.className = 'slice-view';
    sliceView.id = 'main-slice-view';
    sliceView.innerHTML = `
        <div class="slice-view-header">横断面 (Z轴切片 - XY平面)</div>
        <div class="slice-view-content">
            <div class="slice-image-container">
                <img id="slice-img" src="" alt="CT切片">
            </div>
        </div>
        <div class="slice-view-controls">
            <button class="slice-btn prev-slice" id="prev-btn"><i class="fas fa-chevron-left"></i></button>
            <span class="slice-index" id="slice-index">0 / 0</span>
            <button class="slice-btn next-slice" id="next-btn"><i class="fas fa-chevron-right"></i></button>
        </div>
    `;
    
    viewerContainer.appendChild(sliceView);
    
    // 添加滚动事件监听器
    viewerContainer.addEventListener('wheel', handleSliceScroll);
    
    // 添加按钮事件监听器
    document.getElementById('prev-btn').addEventListener('click', () => changeSlice(-1));
    document.getElementById('next-btn').addEventListener('click', () => changeSlice(1));
}

// 处理滚轮事件
function handleSliceScroll(event) {
    event.preventDefault();
    
    // 确定滚动方向
    const delta = Math.sign(event.deltaY);
    
    // 切换切片
    changeSlice(delta);
}

// 切换切片
function changeSlice(delta) {
    // 计算新的切片索引
    let newIndex = currentSliceIndex + delta;
    
    // 确保索引在有效范围内
    newIndex = Math.max(0, Math.min(maxSliceIndex, newIndex));
    
    // 如果索引没有变化，不更新
    if (newIndex === currentSliceIndex) {
        return;
    }
    
    // 更新当前索引
    currentSliceIndex = newIndex;
    
    // 加载新切片
    loadSlice(currentSliceIndex);
}

// 加载切片
function loadSlice(sliceIndex) {
    console.log(`加载Z轴切片，索引: ${sliceIndex}`);
    
    // 获取图像元素
    const imgElement = document.getElementById('slice-img');
    if (!imgElement) {
        console.error('找不到切片图像元素');
        return;
    }
    
    // 更新切片索引显示
    const indexElement = document.getElementById('slice-index');
    if (indexElement) {
        indexElement.textContent = `${sliceIndex} / ${maxSliceIndex}`;
    }
    
    // 构建API URL
    const url = `/api/lung_slice/${currentSessionId}/z/${sliceIndex}`;
    // 添加时间戳防止缓存
    const finalUrl = `${url}?t=${Date.now()}`;
    // 设置加载事件
    imgElement.onload = function() {
        console.log('切片图像加载成功');
    };
    
    imgElement.onerror = function() {
        console.error('切片图像加载失败');
        imgElement.src = ''; // 清空错误的图像
        showMessage('切片图像加载失败', 'error');
    };
    
    // 加载图像
    imgElement.src = finalUrl;
    
    // 如果有结节标记功能，更新结节标记
    if (window.renderNoduleMarkers) {
        window.renderNoduleMarkers(sliceIndex);
    }
}

// 获取检测结果
function fetchResults() {
    console.log('获取检测结果...');
    
    // 检查会话ID是否存在
    if (!currentSessionId) {
        console.error('无法获取结果: 没有会话ID');
        return;
    }
    
    // 更新UI状态
    document.getElementById('progress-message').textContent = '加载检测结果...';
    
    // 加载肺部切片数据（如果尚未加载）
    if (!lungSegmentationLoaded) {
        loadLungSegmentation();
    }
    
    // 获取结果
    fetch(`/api/results/${currentSessionId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`服务器返回错误状态: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('检测结果:', data);
            
            // 更新UI状态
            if (data.nodules && data.nodules.length > 0) {
                updateUIState('detected');
                document.getElementById('progress-message').textContent = `检测到 ${data.nodules.length} 个结节`;
                
                // 保存结节数据到全局变量
                window.nodules = data.nodules;
                
                // 更新结节列表
                updateNoduleList(data.nodules);
                
                // 创建结节标记
                createNoduleMarkers();
            } else {
                updateUIState('detected');
                document.getElementById('progress-message').textContent = '未检测到结节';
                updateNoduleList([]);
            }
        })
        .catch(error => {
            console.error('获取结果出错:', error);
            document.getElementById('progress-message').textContent = `获取结果失败: ${error.message}`;
            showMessage(`获取结果出错: ${error.message}`, 'error');
        });
}

// 更新结节列表
function updateNoduleList(nodules) {
    console.log('更新结节列表，数量:', nodules ? nodules.length : 0);
    
    const noduleListContainer = document.getElementById('nodule-list');
    if (!noduleListContainer) {
        console.error('找不到结节列表容器');
        return;
    }
    
    // 清空现有列表
    noduleListContainer.innerHTML = '';
    
        // 更新结节计数
    const noduleCountElement = document.getElementById('nodules-count');
    if (noduleCountElement) {
        noduleCountElement.textContent = nodules ? nodules.length : '0';
    }
    
    // 如果没有结节
    if (!nodules || nodules.length === 0) {
        noduleListContainer.innerHTML = '<div class="no-nodules">未检测到结节</div>';
        return;
    }
    
    // 添加结节到列表
    nodules.forEach((nodule, index) => {
        try {
            const noduleItem = document.createElement('div');
            noduleItem.className = 'nodule-item';
            noduleItem.dataset.id = nodule.id || index;
            
            // 提取结节信息
            const diameter = (nodule.diameter_mm || 10).toFixed(1);
            const probability = ((nodule.probability || 0.5) * 100).toFixed(0);
            
            // 检查坐标数据
            let coordsStr = '[未知]';
            if (nodule.voxel_coords && Array.isArray(nodule.voxel_coords)) {
                coordsStr = `[${nodule.voxel_coords.map(v => Math.round(v)).join(', ')}]`;
            }
            
            // 结节名称和详细信息
            noduleItem.innerHTML = `
                <div class="nodule-header">
                    <span class="nodule-name">结节 #${index + 1}</span>
                    <span class="nodule-probability ${probability > 70 ? 'high' : (probability > 30 ? 'medium' : 'low')}">
                        ${probability}%
                    </span>
                </div>
                <div class="nodule-details">
                    <div class="nodule-detail">
                        <span class="detail-label">直径:</span>
                        <span class="detail-value">${diameter} mm</span>
                    </div>
                    <div class="nodule-detail">
                        <span class="detail-label">位置:</span>
                        <span class="detail-value">${coordsStr}</span>
                    </div>
                </div>
            `;
            
            // 添加到列表
            noduleListContainer.appendChild(noduleItem);
        } catch (error) {
            console.error(`处理结节 #${index+1} 时出错:`, error);
        }
    });
    
    // 默认选中第一个结节
        if (nodules.length > 0) {
            const firstNoduleItem = noduleListContainer.querySelector('.nodule-item');
            if (firstNoduleItem) {
                firstNoduleItem.classList.add('selected');
                const noduleId = firstNoduleItem.dataset.id;
                loadNoduleDetails(noduleId);
            }
        }
}

// 创建结节标记
function createNoduleMarkers() {
    if (!window.nodules || !window.nodules.length) {
        console.log('没有结节数据可用于创建标记');
        return;
    }
    
    console.log(`为 ${window.nodules.length} 个结节创建标记`);
    
    // 定义全局渲染函数
    window.renderNoduleMarkers = function(currentZIndex) {
        const imageContainer = document.querySelector('.slice-image-container');
        const sliceImg = document.getElementById('slice-img');
        
        if (!imageContainer || !sliceImg || !sliceImg.complete) {
            console.log('图像容器或图像未准备好，无法渲染结节标记');
                return;
        }
        
        // 清除所有现有标记
        const existingMarkers = document.querySelectorAll('.nodule-marker');
        existingMarkers.forEach(marker => marker.remove());
        
        // 获取图像尺寸
        const imgWidth = sliceImg.width;
        const imgHeight = sliceImg.height;
        
        if (imgWidth <= 0 || imgHeight <= 0) {
            console.log('图像尺寸无效，无法渲染标记');
            return;
        }
        
        // 获取容器尺寸
        const containerRect = imageContainer.getBoundingClientRect();
        
        // 为当前切片中的每个结节创建标记
        window.nodules.forEach((nodule, index) => {
            try {
                // 检查结节是否有有效坐标
                if (!nodule.voxel_coords || !Array.isArray(nodule.voxel_coords) || nodule.voxel_coords.length < 3) {
                    console.warn(`结节 #${index+1} 没有有效坐标`);
                    return;
                }
                
                // 提取坐标
                const [x, y, z] = nodule.voxel_coords.map(Math.round);
                
                // 仅显示当前Z切片上或附近的结节（容许一定误差）
                const zTolerance = 1; // 允许的Z轴误差范围
                if (Math.abs(z - currentZIndex) > zTolerance) {
                    return;
                }
                
                // 计算标记在图像上的位置（相对于图像尺寸）
                // 注意：根据实际图像坐标系统可能需要调整
                const relativeX = x / maxSliceIndex; // 假设x坐标范围与切片数量相关
                const relativeY = y / maxSliceIndex; // 假设y坐标范围与切片数量相关
                
                // 将相对坐标转换为绝对像素坐标
                const markerX = relativeX * imgWidth;
                const markerY = relativeY * imgHeight;
                
                // 创建标记元素
                const marker = document.createElement('div');
                marker.className = 'nodule-marker';
                marker.dataset.id = nodule.id || index;
                
                // 设置标记大小（根据结节直径调整）
                const size = Math.max(16, (nodule.diameter_mm || 5) * 2);
                marker.style.width = `${size}px`;
                marker.style.height = `${size}px`;
                
                // 设置标记位置
                marker.style.left = `${markerX}px`;
                marker.style.top = `${markerY}px`;
                
                // 根据结节概率设置样式
                const probability = (nodule.probability || 0.5) * 100;
                if (probability > 70) {
                    marker.classList.add('high-prob');
                } else if (probability > 30) {
                    marker.classList.add('medium-prob');
                } else {
                    marker.classList.add('low-prob');
                }
                // 添加标记编号
                marker.innerHTML = `<span class="marker-label">${index + 1}</span>`;
                // 添加点击事件
                marker.addEventListener('click', function() {
                    // 高亮显示对应的结节列表项
                    highlightNoduleInList(this.dataset.id);
                    // 加载结节详情
                    loadNoduleDetails(this.dataset.id);
                });
                
                // 添加到容器
                imageContainer.appendChild(marker);
                
            } catch (error) {
                console.error(`为结节 #${index+1} 创建标记时出错:`, error);
            }
        });
    };
    
    // 为当前切片渲染标记
    window.renderNoduleMarkers(currentSliceIndex);
}

// 高亮显示结节列表中的特定结节
function highlightNoduleInList(noduleId) {
    // 移除所有当前选中项
    document.querySelectorAll('.nodule-item.selected').forEach(item => {
        item.classList.remove('selected');
    });
    
    // 找到并选中指定结节
    const noduleItem = document.querySelector(`.nodule-item[data-id="${noduleId}"]`);
    if (noduleItem) {
        noduleItem.classList.add('selected');
        
        // 确保选中项在视图中可见
        noduleItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

// 加载结节详情
function loadNoduleDetails(noduleId) {
    console.log(`加载结节详情，ID: ${noduleId}`);
    
    if (!window.nodules) {
        console.error('没有结节数据可用');
        return;
    }
    
    // 找到指定结节
    const noduleIndex = parseInt(noduleId);
    console.log('window nodule 有哪些数据\n', window.nodules);
    // 通过ID查找结节数据
    let nodule = null;
    // 将noduleId转换为整数
    const targetId = parseInt(noduleId);
    
    // 遍历结节数组查找匹配的ID
    for (let i = 0; i < window.nodules.length; i++) {
        // 确保结节的id属性转换为整数进行比较
        if (parseInt(window.nodules[i].id) === targetId) {
            nodule = window.nodules[i];
            console.log(`找到匹配的结节，ID: ${targetId}`);
            break;
        }
    }
    
    // 如果找不到匹配的结节，则使用索引方式获取
    if (!nodule) {
        console.warn(`未找到ID为${targetId}的结节，尝试使用索引方式获取`);
        // 尝试使用索引方式获取
        nodule = window.nodules[noduleIndex];
    }

    if (!nodule) {
        console.error(`找不到ID为 ${noduleId} 的结节`);
        return;
    }
    
    // 获取详情容器
    const detailsContainer = document.getElementById('nodule-detail-area');
    if (!detailsContainer) {
        console.error('找不到结节详情容器');
        return;
    }
    
    // 清空当前详情
    detailsContainer.innerHTML = '';
    
    try {
        // 提取结节数据
        const diameter = (nodule.diameter_mm || 10).toFixed(1);
        const probability = ((nodule.probability || 0.5) * 100).toFixed(0);
        const coords = nodule.voxel_coords || [0, 0, 0];
        const [x, y, z] = coords.map(v => Math.round(v));
        
        // 如果结节有Z坐标，跳转到对应切片
        if (typeof z === 'number' && z >= 0 && z <= maxSliceIndex) {
            currentSliceIndex = z;
            loadSlice(z);
        }
        
        // 创建详情内容
        detailsContainer.innerHTML = `
            <h3>结节 #${noduleIndex + 1} 详情</h3>
            <div class="detail-row">
                <span class="detail-label">直径:</span>
                <span class="detail-value">${diameter} mm</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">恶性概率:</span>
                <span class="detail-value probability-value ${probability > 70 ? 'high' : (probability > 30 ? 'medium' : 'low')}">
                    ${probability}%
                </span>
            </div>
            <div class="detail-row">
                <span class="detail-label">位置坐标:</span>
                <span class="detail-value">[${x}, ${y}, ${z}]</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">类型:</span>
                <span class="detail-value">${nodule.type || '未分类'}</span>
            </div>
            
            <div class="actions-row">
                <button id="goto-nodule-btn" class="btn btn-primary">定位到结节</button>
                <button id="nodule-report-btn" class="btn btn-secondary">查看报告</button>
            </div>
        `;
        
        // 添加按钮事件监听器
        document.getElementById('goto-nodule-btn').addEventListener('click', function() {
            // 定位到结节所在切片
            if (typeof z === 'number' && z >= 0 && z <= maxSliceIndex) {
                currentSliceIndex = z;
                loadSlice(z);
                
                // 高亮显示结节标记
                setTimeout(() => {
                    const marker = document.querySelector(`.nodule-marker[data-id="${noduleId}"]`);
                    if (marker) {
                        marker.classList.add('highlight');
                        // 移除高亮状态
                        setTimeout(() => {
                            marker.classList.remove('highlight');
                        }, 2000);
                    }
                }, 100);
            }
        });
        
        document.getElementById('nodule-report-btn').addEventListener('click', function() {
            showMessage('结节报告功能尚未实现', 'info');
        });
        
    } catch (error) {
        console.error('加载结节详情出错:', error);
        detailsContainer.innerHTML = `<div class="error-message">加载结节详情失败: ${error.message}</div>`;
    }
}

// 重置视图
function resetView() {
    console.log('重置视图');

    // 如果没有数据，不执行任何操作
    if (!lungSegmentationLoaded || maxSliceIndex <= 0) {
        console.log('没有数据可重置');
        return;
    }

    // 将当前切片索引重置为中间位置
    currentSliceIndex = Math.floor(maxSliceIndex / 2);

    // 加载中间切片
    loadSlice(currentSliceIndex);

    // 显示消息
    showMessage('视图已重置', 'info');
}

// 显示消息
function showMessage(message, type = 'info') {
    console.log(`显示消息: ${message} (${type})`);

    // 创建消息容器（如果不存在）
    let msgContainer = document.getElementById('message-container');
    if (!msgContainer) {
        msgContainer = document.createElement('div');
        msgContainer.id = 'message-container';
        document.body.appendChild(msgContainer);
    }

    // 创建消息元素
    const msgElement = document.createElement('div');
    msgElement.className = `message ${type}`;
    msgElement.innerHTML = `
        <span class="message-icon">
            <i class="fas ${type === 'error' ? 'fa-exclamation-circle' : (type === 'success' ? 'fa-check-circle' : 'fa-info-circle')}"></i>
        </span>
        <span class="message-text">${message}</span>
    `;

    // 添加到容器
    msgContainer.appendChild(msgElement);

    // 显示消息
    setTimeout(() => {
        msgElement.classList.add('show');
    }, 10);

    // 设置自动消失
    setTimeout(() => {
        msgElement.classList.remove('show');
        setTimeout(() => {
            msgElement.remove();
        }, 300);
    }, 3000);
}

// 更新UI状态
function updateUIState(state) {
    // 移除所有状态类
    document.body.classList.remove(
        'state-initial',
        'state-uploading',
        'state-detecting',
        'state-detected'
    );

    // 添加新状态类
    document.body.classList.add(`state-${state}`);

    // 更新UI元素可见性
    switch (state) {
        case 'initial':
            document.getElementById('step-upload').classList.add('active');
            document.getElementById('step-detect').classList.remove('active');
            document.getElementById('step-visualize').classList.remove('active');
            break;
        case 'uploading':
            document.getElementById('step-upload').classList.add('active');
            document.getElementById('step-detect').classList.remove('active');
            document.getElementById('step-visualize').classList.remove('active');
            break;
        case 'detecting':
            document.getElementById('step-upload').classList.add('completed');
            document.getElementById('step-detect').classList.add('active');
            document.getElementById('step-visualize').classList.remove('active');
            break;
        case 'detected':
            document.getElementById('step-upload').classList.add('completed');
            document.getElementById('step-detect').classList.add('completed');
            document.getElementById('step-visualize').classList.add('active');
            break;
    }
}