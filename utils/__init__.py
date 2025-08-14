#!/usr/bin/env python3
"""
Utils package for YOLOv8 teeth detection project
"""

# 导入主要模块
try:
    from .file_utils import *
except ImportError:
    pass

try:
    from .visualization import *
except ImportError:
    pass

try:
    from .metrics import *
except ImportError:
    pass

from .demo_utils import DentalDetectionDemo

# 版本信息
__version__ = "1.0.0"
__author__ = "YOLOv8 Teeth Detection Project"

# 导出的主要类和函数
__all__ = [
    'DentalDetectionDemo',
]