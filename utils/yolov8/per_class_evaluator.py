"""
Per-class metrics calculator for YOLOv8 evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import json


class PerClassMetrics:
    """
    YOLOv8 每类别指标计算器
    """
    
    def __init__(self, model_path, data_yaml_path, class_names):
        """
        初始化每类别指标计算器
        
        Args:
            model_path (str): 训练好的模型路径
            data_yaml_path (str): 数据配置文件路径
            class_names (list): 类别名称列表
        """
        self.model_path = model_path
        self.data_yaml_path = data_yaml_path
        self.class_names = class_names
        self.model = None
        
    def load_model(self):
        """加载训练好的模型"""
        try:
            self.model = YOLO(self.model_path)
            print(f"[✓] 模型加载成功: {self.model_path}")
            return True
        except Exception as e:
            print(f"[!] 模型加载失败: {e}")
            return False
    
    def evaluate_per_class(self, split='val'):
        """
        计算每类别的详细指标
        
        Args:
            split (str): 数据分割 ('train', 'val', 'test')
        
        Returns:
            dict: 每类别指标结果
        """
        if not self.model:
            if not self.load_model():
                return {}
        
        try:
            # 运行验证评估
            print(f"[🔄] 正在评估 {split} 数据集...")
            results = self.model.val(data=self.data_yaml_path, split=split, verbose=False)
            
            # 提取每类别指标
            per_class_metrics = {}
            
            if hasattr(results, 'box') and results.box is not None:
                # AP@0.5 每类别
                if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
                    ap50_per_class = results.box.ap50.cpu().numpy()
                    
                # AP@0.5:0.95 每类别  
                if hasattr(results.box, 'ap') and results.box.ap is not None:
                    ap_per_class = results.box.ap.cpu().numpy()
                    
                # 精确率每类别
                if hasattr(results.box, 'p') and results.box.p is not None:
                    precision_per_class = results.box.p.cpu().numpy()
                    
                # 召回率每类别
                if hasattr(results.box, 'r') and results.box.r is not None:
                    recall_per_class = results.box.r.cpu().numpy()
                
                # 组织每类别数据
                for i, class_name in enumerate(self.class_names):
                    if i < len(ap50_per_class):
                        per_class_metrics[class_name] = {
                            'precision': float(precision_per_class[i]) if i < len(precision_per_class) else 0.0,
                            'recall': float(recall_per_class[i]) if i < len(recall_per_class) else 0.0,
                            'ap50': float(ap50_per_class[i]) if i < len(ap50_per_class) else 0.0,
                            'ap50_95': float(np.mean(ap_per_class[i])) if i < len(ap_per_class) and len(ap_per_class[i]) > 0 else 0.0,
                            'f1_score': 0.0
                        }
                        
                        # 计算F1-Score
                        p = per_class_metrics[class_name]['precision']
                        r = per_class_metrics[class_name]['recall']
                        if p + r > 0:
                            per_class_metrics[class_name]['f1_score'] = 2 * p * r / (p + r)
            
            return per_class_metrics
            
        except Exception as e:
            print(f"[!] 每类别指标计算失败: {e}")
            return {}
    
    def plot_per_class_metrics(self, metrics_dict, save_path):
        """
        绘制每类别指标对比图
        
        Args:
            metrics_dict (dict): 每类别指标字典
            save_path (str): 保存路径
        """
        try:
            if not metrics_dict:
                print("[!] 没有可用的每类别指标数据")
                return
            
            # 准备数据
            classes = list(metrics_dict.keys())
            precision_vals = [metrics_dict[c]['precision'] for c in classes]
            recall_vals = [metrics_dict[c]['recall'] for c in classes]
            f1_vals = [metrics_dict[c]['f1_score'] for c in classes]
            ap50_vals = [metrics_dict[c]['ap50'] for c in classes]
            ap50_95_vals = [metrics_dict[c]['ap50_95'] for c in classes]
            
            # 创建2x2子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Per-Class Metrics Analysis', fontsize=16, fontweight='bold')
            
            # 1. 精确率和召回率对比
            ax1 = axes[0, 0]
            x = np.arange(len(classes))
            width = 0.35
            ax1.bar(x - width/2, precision_vals, width, label='Precision', color='green', alpha=0.7)
            ax1.bar(x + width/2, recall_vals, width, label='Recall', color='red', alpha=0.7)
            ax1.set_xlabel('Classes')
            ax1.set_ylabel('Score')
            ax1.set_title('Precision vs Recall by Class')
            ax1.set_xticks(x)
            ax1.set_xticklabels(classes, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. F1-Score对比
            ax2 = axes[0, 1]
            bars = ax2.bar(classes, f1_vals, color='purple', alpha=0.7)
            ax2.set_xlabel('Classes')
            ax2.set_ylabel('F1-Score')
            ax2.set_title('F1-Score by Class')
            ax2.set_xticklabels(classes, rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # 在柱状图上添加数值标签
            for bar, val in zip(bars, f1_vals):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom')
            
            # 3. mAP@0.5对比
            ax3 = axes[1, 0]
            bars = ax3.bar(classes, ap50_vals, color='blue', alpha=0.7)
            ax3.set_xlabel('Classes')
            ax3.set_ylabel('mAP@0.5')
            ax3.set_title('mAP@0.5 by Class')
            ax3.set_xticklabels(classes, rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, val in zip(bars, ap50_vals):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom')
            
            # 4. mAP@0.5:0.95对比
            ax4 = axes[1, 1]
            bars = ax4.bar(classes, ap50_95_vals, color='orange', alpha=0.7)
            ax4.set_xlabel('Classes')
            ax4.set_ylabel('mAP@0.5:0.95')
            ax4.set_title('mAP@0.5:0.95 by Class')
            ax4.set_xticklabels(classes, rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, val in zip(bars, ap50_95_vals):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[✓] 每类别指标图表已保存至: {save_path}")
            
        except Exception as e:
            print(f"[!] 绘制每类别指标图表失败: {e}")
    
    def save_per_class_report(self, metrics_dict, save_path):
        """
        保存每类别指标详细报告
        
        Args:
            metrics_dict (dict): 每类别指标字典
            save_path (str): 保存路径
        """
        try:
            if not metrics_dict:
                print("[!] 没有可用的每类别指标数据")
                return
            
            report = "# 每类别详细指标报告\n\n"
            report += "## 指标概览\n\n"
            
            # 创建表格
            report += "| 类别 | 精确率 | 召回率 | F1-Score | mAP@0.5 | mAP@0.5:0.95 |\n"
            report += "|------|--------|--------|----------|---------|---------------|\n"
            
            total_precision = 0
            total_recall = 0
            total_f1 = 0
            total_ap50 = 0
            total_ap50_95 = 0
            
            for class_name, metrics in metrics_dict.items():
                report += f"| {class_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['ap50']:.4f} | {metrics['ap50_95']:.4f} |\n"
                
                total_precision += metrics['precision']
                total_recall += metrics['recall'] 
                total_f1 += metrics['f1_score']
                total_ap50 += metrics['ap50']
                total_ap50_95 += metrics['ap50_95']
            
            # 平均值
            num_classes = len(metrics_dict)
            avg_precision = total_precision / num_classes
            avg_recall = total_recall / num_classes
            avg_f1 = total_f1 / num_classes
            avg_ap50 = total_ap50 / num_classes
            avg_ap50_95 = total_ap50_95 / num_classes
            
            report += f"| **平均值** | **{avg_precision:.4f}** | **{avg_recall:.4f}** | **{avg_f1:.4f}** | **{avg_ap50:.4f}** | **{avg_ap50_95:.4f}** |\n\n"
            
            # 详细分析
            report += "## 详细分析\n\n"
            
            # 找出最佳和最差表现的类别
            best_f1_class = max(metrics_dict.items(), key=lambda x: x[1]['f1_score'])
            worst_f1_class = min(metrics_dict.items(), key=lambda x: x[1]['f1_score'])
            
            report += f"### 🏆 最佳F1-Score类别\n"
            report += f"**{best_f1_class[0]}**: F1-Score = {best_f1_class[1]['f1_score']:.4f}\n\n"
            
            report += f"### ⚠️ 最差F1-Score类别\n"
            report += f"**{worst_f1_class[0]}**: F1-Score = {worst_f1_class[1]['f1_score']:.4f}\n\n"
            
            # 性能建议
            report += "### 💡 改进建议\n\n"
            for class_name, metrics in metrics_dict.items():
                if metrics['f1_score'] < 0.5:
                    report += f"- **{class_name}**: F1-Score较低({metrics['f1_score']:.3f})，建议增加该类别的训练样本或调整数据增强策略\n"
                if metrics['precision'] < 0.6:
                    report += f"- **{class_name}**: 精确率较低({metrics['precision']:.3f})，可能存在较多误检，建议提高置信度阈值\n"
                if metrics['recall'] < 0.6:
                    report += f"- **{class_name}**: 召回率较低({metrics['recall']:.3f})，可能存在较多漏检，建议降低置信度阈值或增加训练数据\n"
            
            report += f"\n---\n*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
            
            # 保存报告
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"[✓] 每类别指标报告已保存至: {save_path}")
            
        except Exception as e:
            print(f"[!] 保存每类别指标报告失败: {e}")


def evaluate_and_visualize_per_class(model_path, data_yaml_path, class_names, output_dir):
    """
    便捷函数：评估并可视化每类别指标
    
    Args:
        model_path (str): 模型路径
        data_yaml_path (str): 数据配置文件路径
        class_names (list): 类别名称列表
        output_dir (str): 输出目录
    """
    try:
        # 创建评估器
        evaluator = PerClassMetrics(model_path, data_yaml_path, class_names)
        
        # 评估验证集
        print("[🔄] 开始每类别指标评估...")
        metrics = evaluator.evaluate_per_class('val')
        
        if not metrics:
            print("[!] 每类别指标评估失败")
            return
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成可视化图表
        plot_path = os.path.join(output_dir, "per_class_metrics.png")
        evaluator.plot_per_class_metrics(metrics, plot_path)
        
        # 生成详细报告
        report_path = os.path.join(output_dir, "per_class_report.md")
        evaluator.save_per_class_report(metrics, report_path)
        
        print(f"[✅] 每类别指标评估完成!")
        print(f"📊 可视化图表: {plot_path}")
        print(f"📋 详细报告: {report_path}")
        
        return metrics
        
    except Exception as e:
        print(f"[!] 每类别指标评估过程失败: {e}")
        return None
