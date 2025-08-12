"""
Visualization utilities for YOLOv8 training
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_loss_curve(results_csv_path, save_path):
    """
    绘制训练损失曲线图
    
    Args:
        results_csv_path (str): results.csv文件路径
        save_path (str): 保存图片的路径
    """
    try:
        df = pd.read_csv(results_csv_path)
        # 清理列名，去除多余的空格
        df.columns = df.columns.str.strip()
        plt.figure(figsize=(12, 8))
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        ax1.plot(df['epoch'], df['train/box_loss'], label='Box Loss', linewidth=2)
        if 'train/cls_loss' in df.columns:
            ax1.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', linewidth=2)
        if 'train/dfl_loss' in df.columns:
            ax1.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss Value')
        ax1.set_title('Training Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 精度和召回率
        ax2.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2, color='green')
        ax2.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2, color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Value')
        ax2.set_title('Precision & Recall')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # mAP指标（如果存在）
        if 'metrics/mAP50(B)' in df.columns:
            ax3.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2, color='blue')
        if 'metrics/mAP50-95(B)' in df.columns:
            ax3.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=2, color='orange')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('mAP')
        ax3.set_title('Mean Average Precision')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 学习率（如果存在）
        if 'lr/pg0' in df.columns:
            ax4.plot(df['epoch'], df['lr/pg0'], label='Learning Rate', linewidth=2, color='purple')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No LR data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Learning Rate (N/A)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[✓] 训练可视化图表已保存至: {save_path}")
        
    except Exception as e:
        print(f"[!] 绘制训练曲线图失败：{e}")


def plot_simple_loss_curve(results_csv_path, save_path):
    """
    绘制简单的损失曲线图（原版本）
    
    Args:
        results_csv_path (str): results.csv文件路径
        save_path (str): 保存图片的路径
    """
    try:
        df = pd.read_csv(results_csv_path)
        # 清理列名，去除多余的空格
        df.columns = df.columns.str.strip()
        plt.figure(figsize=(8, 5))
        plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
        if 'train/cls_loss' in df.columns:
            plt.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss')
        if 'train/dfl_loss' in df.columns:
            plt.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
        plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
        plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Loss & Metrics Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[✓] Loss 曲线图已保存至: {save_path}")
    except Exception as e:
        print(f"[!] 绘制 loss 曲线图失败：{e}")
