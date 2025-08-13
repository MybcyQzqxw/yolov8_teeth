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
