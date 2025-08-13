"""
Visualization utilities for YOLOv8 training
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_training_metrics(results_csv_path, save_path):
    """
    绘制完整的训练指标可视化图表
    
    Args:
        results_csv_path (str): results.csv文件路径
        save_path (str): 保存图片的路径
    """
    try:
        df = pd.read_csv(results_csv_path)
        # 清理列名，去除多余的空格
        df.columns = df.columns.str.strip()
        
        # 计算F1-Score
        df['f1_score'] = df.apply(lambda row: calculate_f1_score(
            row.get('metrics/precision(B)', 0), 
            row.get('metrics/recall(B)', 0)
        ), axis=1)
        
        # 创建2x3的子图布局，显示所有重要指标
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('YOLOv8 Training Metrics Analysis', fontsize=16, fontweight='bold')
        
        # 1. 训练损失曲线
        ax1 = axes[0, 0]
        if 'train/box_loss' in df.columns:
            ax1.plot(df['epoch'], df['train/box_loss'], label='Box Loss', linewidth=2, color='blue')
        if 'train/cls_loss' in df.columns:
            ax1.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', linewidth=2, color='red')
        if 'train/dfl_loss' in df.columns:
            ax1.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', linewidth=2, color='green')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss Value')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 验证损失曲线
        ax2 = axes[0, 1]
        if 'val/box_loss' in df.columns:
            ax2.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linewidth=2, color='orange')
        if 'val/cls_loss' in df.columns:
            ax2.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', linewidth=2, color='purple')
        if 'val/dfl_loss' in df.columns:
            ax2.plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss', linewidth=2, color='brown')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Value')
        ax2.set_title('Validation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 精确率、召回率和F1-Score
        ax3 = axes[0, 2]
        ax3.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2, color='green')
        ax3.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2, color='red')
        ax3.plot(df['epoch'], df['f1_score'], label='F1-Score', linewidth=2, color='purple')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.set_title('Precision, Recall & F1-Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. mAP指标
        ax4 = axes[1, 0]
        if 'metrics/mAP50(B)' in df.columns:
            ax4.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2, color='blue')
        if 'metrics/mAP50-95(B)' in df.columns:
            ax4.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=2, color='orange')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('mAP')
        ax4.set_title('Mean Average Precision')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 学习率调度
        ax5 = axes[1, 1]
        if 'lr/pg0' in df.columns:
            ax5.plot(df['epoch'], df['lr/pg0'], label='Learning Rate', linewidth=2, color='darkgreen')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Learning Rate')
            ax5.set_title('Learning Rate Schedule')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No LR data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Learning Rate (N/A)')
        
        # 6. 精确率vs召回率散点图
        ax6 = axes[1, 2]
        scatter = ax6.scatter(df['metrics/recall(B)'], df['metrics/precision(B)'], 
                             c=df['epoch'], cmap='viridis', s=50, alpha=0.7)
        ax6.set_xlabel('Recall')
        ax6.set_ylabel('Precision')
        ax6.set_title('Precision vs Recall Evolution')
        ax6.grid(True, alpha=0.3)
        # 添加颜色条显示epoch进展
        plt.colorbar(scatter, ax=ax6, label='Epoch')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[✓] 训练指标可视化图表已保存至: {save_path}")
        
    except Exception as e:
        print(f"[!] 绘制训练指标图表失败：{e}")


def calculate_f1_score(precision, recall):
    """
    计算F1-Score
    
    Args:
        precision (float): 精确率
        recall (float): 召回率
    
    Returns:
        float: F1-Score
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
