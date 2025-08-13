"""
Configuration management for YOLOv8 teeth detection project
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """训练配置类"""
    model: str = "yolov8n"  # 默认模型
    epochs: int = 30
    batch_size: str = "16"
    data_dir: str = "./preprocessed_dataset/yolov8"
    output_dir: str = "./outputs"
    enable_logging: bool = True
    imgsz: int = 640
    device: str = "auto"  # auto, cpu, 0, 1, 2, 3...
    
    def __post_init__(self):
        """验证配置参数"""
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not os.path.exists(self.data_dir):
            print(f"Warning: data_dir '{self.data_dir}' does not exist")


@dataclass  
class DatasetConfig:
    """数据集配置类"""
    source_dir: str = "./dataset/dentalai"
    target_dir: str = "./preprocessed_dataset/yolov8"
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    def __post_init__(self):
        """验证配置参数"""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Train/Val/Test ratios must sum to 1.0, got {total_ratio}")


# 默认配置实例
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_DATASET_CONFIG = DatasetConfig()
