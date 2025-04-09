import os
import sys
import subprocess
import platform
import webbrowser
import time

def get_python_command():
    """获取Python命令"""
    if platform.system() == "Windows":
        return "python"
    else:
        return "python3"

def check_dependencies():
    """检查必要的依赖是否已安装"""
    try:
        import flask
        import numpy
        import tensorflow
        import SimpleITK
        print("✓ 所有必要的依赖已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请安装所需依赖: pip install flask flask-cors numpy tensorflow SimpleITK")
        return False

def create_directories():
    """创建必要的目录"""
    os.makedirs("backend/uploads", exist_ok=True)
    os.makedirs("backend/models", exist_ok=True)
    print("✓ 目录创建完成")

def run_backend_server():
    """运行后端服务器"""
    python_cmd = get_python_command()
    
    # 构建命令
    cmd = [python_cmd, "backend/app.py"]
    
    # 启动后端服务器
    print("\n启动后端服务器...")
    process = subprocess.Popen(cmd)
    
    # 等待服务器启动
    time.sleep(2)
    
    # 打开浏览器
    print("正在打开浏览器...")
    webbrowser.open("http://localhost:5000")
    
    print("\n服务器已启动!\n")
    print("在浏览器中访问: http://localhost:5000")
    print("按 Ctrl+C 停止服务器")
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务器...")
        process.terminate()
        process.wait()
        print("服务器已停止")

def main():
    """主函数"""
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("CT图像分析系统启动工具")
    print("=" * 30)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 创建必要的目录
    create_directories()
    
    # 运行后端服务器
    run_backend_server()

if __name__ == "__main__":
    main() 