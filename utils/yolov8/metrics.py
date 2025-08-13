"""
Enhanced metrics and evaluation utilities for YOLOv8 training
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from ultralytics import YOLO
import os
import json


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


def calculate_iou_from_results(results_csv_path):
    """
    从训练结果中计算平均IoU
    
    Args:
        results_csv_path (str): results.csv文件路径
    
    Returns:
        dict: IoU统计信息
    """
    try:
        df = pd.read_csv(results_csv_path)
        df.columns = df.columns.str.strip()
        
        # YOLOv8的mAP指标基于IoU计算，可以从中推算IoU信息
        iou_stats = {}
        
        if 'metrics/mAP50(B)' in df.columns:
            # mAP@0.5 反映了IoU>=0.5的检测质量
            latest_map50 = df['metrics/mAP50(B)'].iloc[-1]
            iou_stats['avg_iou_at_0.5'] = latest_map50
            
        if 'metrics/mAP50-95(B)' in df.columns:
            # mAP@0.5:0.95 反映了不同IoU阈值下的平均性能
            latest_map50_95 = df['metrics/mAP50-95(B)'].iloc[-1]
            iou_stats['avg_iou_0.5_to_0.95'] = latest_map50_95
            
        return iou_stats
        
    except Exception as e:
        print(f"[!] 计算IoU统计失败: {e}")
        return {}


def enhanced_metrics_analysis(results_csv_path, class_names):
    """
    增强的指标分析
    
    Args:
        results_csv_path (str): results.csv文件路径
        class_names (list): 类别名称列表
    
    Returns:
        dict: 增强的指标统计
    """
    try:
        df = pd.read_csv(results_csv_path)
        df.columns = df.columns.str.strip()
        
        # 获取最新的指标
        latest_metrics = {}
        if len(df) > 0:
            latest_row = df.iloc[-1]
            
            # 基础指标
            precision = latest_row.get('metrics/precision(B)', 0)
            recall = latest_row.get('metrics/recall(B)', 0)
            map50 = latest_row.get('metrics/mAP50(B)', 0)
            map50_95 = latest_row.get('metrics/mAP50-95(B)', 0)
            
            # 计算F1-Score
            f1 = calculate_f1_score(precision, recall)
            
            latest_metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'map50': map50,
                'map50_95': map50_95,
                'epoch': latest_row.get('epoch', 0)
            }
            
            # IoU统计
            iou_stats = calculate_iou_from_results(results_csv_path)
            latest_metrics.update(iou_stats)
        
        return latest_metrics
        
    except Exception as e:
        print(f"[!] 增强指标分析失败: {e}")
        return {}


def plot_enhanced_metrics(results_csv_path, save_path, class_names):
    """
    绘制增强的指标可视化图表，包括F1-Score、IoU等
    
    Args:
        results_csv_path (str): results.csv文件路径
        save_path (str): 保存图片的路径
        class_names (list): 类别名称列表
    """
    try:
        df = pd.read_csv(results_csv_path)
        df.columns = df.columns.str.strip()
        
        # 计算F1-Score
        df['f1_score'] = df.apply(lambda row: calculate_f1_score(
            row.get('metrics/precision(B)', 0), 
            row.get('metrics/recall(B)', 0)
        ), axis=1)
        
        # 创建2x3的子图布局
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced YOLOv8 Training Metrics Analysis', fontsize=16, fontweight='bold')
        
        # 1. 损失曲线
        ax1 = axes[0, 0]
        if 'train/box_loss' in df.columns:
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
        
        # 2. 精确率、召回率和F1-Score
        ax2 = axes[0, 1]
        ax2.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2, color='green')
        ax2.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2, color='red')
        ax2.plot(df['epoch'], df['f1_score'], label='F1-Score', linewidth=2, color='purple')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Value')
        ax2.set_title('Precision, Recall & F1-Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. mAP指标 (作为IoU质量的代理)
        ax3 = axes[0, 2]
        if 'metrics/mAP50(B)' in df.columns:
            ax3.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@IoU0.5', linewidth=2, color='blue')
        if 'metrics/mAP50-95(B)' in df.columns:
            ax3.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@IoU0.5:0.95', linewidth=2, color='orange')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('mAP (IoU Quality)')
        ax3.set_title('Mean Average Precision (IoU Quality)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 验证集损失
        ax4 = axes[1, 0]
        if 'val/box_loss' in df.columns:
            ax4.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linewidth=2, color='red')
        if 'val/cls_loss' in df.columns:
            ax4.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', linewidth=2, color='orange')
        if 'val/dfl_loss' in df.columns:
            ax4.plot(df['epoch'], df['val/dfl_loss'], label='Val DFL Loss', linewidth=2, color='purple')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation Loss')
        ax4.set_title('Validation Loss Curves')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 学习率
        ax5 = axes[1, 1]
        if 'lr/pg0' in df.columns:
            ax5.plot(df['epoch'], df['lr/pg0'], label='Learning Rate', linewidth=2, color='green')
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
        
        # 6. 指标概览表格
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # 创建最终指标表格
        if len(df) > 0:
            latest_row = df.iloc[-1]
            metrics_data = [
                ['Precision', f"{latest_row.get('metrics/precision(B)', 0):.3f}"],
                ['Recall', f"{latest_row.get('metrics/recall(B)', 0):.3f}"],
                ['F1-Score', f"{latest_row.get('f1_score', 0):.3f}"],
                ['mAP@0.5', f"{latest_row.get('metrics/mAP50(B)', 0):.3f}"],
                ['mAP@0.5:0.95', f"{latest_row.get('metrics/mAP50-95(B)', 0):.3f}"],
                ['Classes', f"{len(class_names)}"],
            ]
            
            table = ax6.table(cellText=metrics_data,
                            colLabels=['Metric', 'Value'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            ax6.set_title('Final Metrics Summary')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[✓] 增强指标可视化图表已保存至: {save_path}")
        
        # 返回增强的指标分析
        return enhanced_metrics_analysis(results_csv_path, class_names)
        
    except Exception as e:
        print(f"[!] 绘制增强指标图表失败: {e}")
        return {}


def generate_metrics_report(results_csv_path, class_names, save_path):
    """
    生成详细的指标报告
    
    Args:
        results_csv_path (str): results.csv文件路径
        class_names (list): 类别名称列表
        save_path (str): 报告保存路径
    """
    try:
        # 分析指标
        metrics = enhanced_metrics_analysis(results_csv_path, class_names)
        
        if not metrics:
            print("[!] 无法生成指标报告：没有有效的指标数据")
            return
        
        # 生成报告内容
        report = f"""# YOLOv8 牙齿检测模型指标报告

## 模型性能概览

### 核心指标
- **精确率 (Precision)**: {metrics.get('precision', 0):.4f}
- **召回率 (Recall)**: {metrics.get('recall', 0):.4f}
- **F1-Score**: {metrics.get('f1_score', 0):.4f}
- **mAP@IoU0.5**: {metrics.get('map50', 0):.4f}
- **mAP@IoU0.5:0.95**: {metrics.get('map50_95', 0):.4f}

### IoU质量分析
- **IoU@0.5阈值质量**: {metrics.get('avg_iou_at_0.5', 0):.4f}
- **IoU综合质量(0.5:0.95)**: {metrics.get('avg_iou_0.5_to_0.95', 0):.4f}

### 检测类别
"""
        
        for i, class_name in enumerate(class_names):
            report += f"- **类别 {i}**: {class_name}\n"
        
        report += f"""
### 性能评估
- **训练轮次**: {int(metrics.get('epoch', 0))}
- **模型平衡性**: {"良好" if metrics.get('f1_score', 0) > 0.5 else "需要改进"}
- **定位精度**: {"优秀" if metrics.get('map50', 0) > 0.7 else "良好" if metrics.get('map50', 0) > 0.5 else "需要改进"}
- **综合性能**: {"优秀" if metrics.get('map50_95', 0) > 0.5 else "良好" if metrics.get('map50_95', 0) > 0.3 else "需要改进"}

### 改进建议
"""
        
        # 根据指标给出改进建议
        if metrics.get('precision', 0) < 0.6:
            report += "- 考虑调整置信度阈值或增加负样本训练\n"
        if metrics.get('recall', 0) < 0.6:
            report += "- 考虑数据增强或增加正样本训练数据\n"
        if metrics.get('f1_score', 0) < 0.5:
            report += "- 模型精确率和召回率需要平衡，考虑调整损失函数权重\n"
        if metrics.get('map50_95', 0) < 0.3:
            report += "- IoU质量较低，考虑调整锚框设置或增加训练轮数\n"
            
        report += f"""
---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存报告
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(f"[✓] 指标报告已保存至: {save_path}")
        
        return metrics
        
    except Exception as e:
        print(f"[!] 生成指标报告失败: {e}")
        return {}
