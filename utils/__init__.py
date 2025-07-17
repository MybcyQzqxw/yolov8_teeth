"""
Utils package for YOLOv8 teeth detection project
"""

from .visualization import plot_loss_curve, plot_simple_loss_curve
from .file_utils import create_output_dirs, validate_files, ensure_model_extension
from .config import TrainingConfig, DatasetConfig, DEFAULT_TRAINING_CONFIG, DEFAULT_DATASET_CONFIG

__all__ = [
    'plot_loss_curve', 
    'plot_simple_loss_curve',
    'create_output_dirs', 
    'validate_files', 
    'ensure_model_extension',
    'TrainingConfig',
    'DatasetConfig', 
    'DEFAULT_TRAINING_CONFIG', 
    'DEFAULT_DATASET_CONFIG'
]
