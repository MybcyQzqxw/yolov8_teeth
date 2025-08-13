"""
File and directory utilities for YOLOv8 training project
"""

import os
from datetime import datetime


def create_output_dirs(model_name, epochs, base_output_dir="outputs", enable_logs=True):
    """
    创建训练输出目录结构
    
    Args:
        model_name (str): 模型名称
        epochs (int): 训练轮数
        base_output_dir (str): 基础输出目录
        enable_logs (bool): 是否启用日志目录
        
    Returns:
        tuple: (base_dir, weights_dir, logs_dir)
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base_dir = os.path.join(base_output_dir, f"train_{model_name}_{epochs}ep_{timestamp}")
    weights_dir = os.path.join(base_dir, "weights")
    logs_dir = os.path.join(base_dir, "logs") if enable_logs else None
    
    os.makedirs(weights_dir, exist_ok=True)
    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)
    
    return base_dir, weights_dir, logs_dir


def validate_files(model_file, data_yaml):
    """
    验证必要文件是否存在
    
    Args:
        model_file (str): 模型文件路径 (YOLOv8会自动下载预训练模型)
        data_yaml (str): 数据配置文件路径
        
    Raises:
        FileNotFoundError: 当数据文件不存在时抛出异常
    """
    # YOLOv8模型文件会自动下载，不需要验证
    # 只验证数据集配置文件
    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(f"数据集配置文件不存在: {data_yaml}")
    
    # 打印模型信息
    if os.path.isfile(model_file):
        print(f"📦 使用本地模型: {model_file}")
    else:
        print(f"📦 模型将自动下载: {model_file}")


def ensure_model_extension(model_name):
    """
    确保模型名称包含.pt扩展名，并返回models/yolov8文件夹中的完整路径
    
    Args:
        model_name (str): 模型名称
        
    Returns:
        str: models/yolov8文件夹中的完整模型文件路径
    """
    # 确保模型名有.pt扩展名
    if not model_name.endswith('.pt'):
        model_name = model_name + '.pt'
    
    # 返回models/yolov8文件夹中的路径
    models_dir = os.path.join(os.getcwd(), 'models', 'yolov8')
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, model_name)
