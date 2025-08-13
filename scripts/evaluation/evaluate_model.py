#!/usr/bin/env python3
"""
YOLOv8 牙齿检测模型评估脚本
支持增强指标分析，包括F1-Score、IoU、混淆矩阵、每类别mAP等
"""

import argparse
import os
import sys
import yaml

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.yolov8.metrics import (
    plot_enhanced_metrics, 
    generate_metrics_report, 
    enhanced_metrics_analysis
)
from utils.yolov8.per_class_evaluator import evaluate_and_visualize_per_class


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 牙齿检测模型增强评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用训练结果进行评估
  python scripts/evaluation/evaluate_model.py --results results.csv --output ./evaluation_output
  
  # 使用模型和数据集进行完整评估
  python scripts/evaluation/evaluate_model.py --model best.pt --data data.yaml --output ./evaluation_output
  
  # 仅分析已有的训练结果
  python scripts/evaluation/evaluate_model.py --results outputs/train_xxx/weights/results.csv
        """)
    
    # 输入参数
    parser.add_argument('--results', '-r', type=str, 
                        help="训练结果CSV文件路径 (results.csv)")
    parser.add_argument('--model', '-m', type=str, 
                        help="训练好的模型文件路径 (.pt文件)")
    parser.add_argument('--data', '-d', type=str, 
                        help="数据配置文件路径 (data.yaml)")
    
    # 输出控制
    parser.add_argument('--output', '-o', type=str, default="./evaluation_output",
                        help="评估结果输出目录 (默认: ./evaluation_output)")
    parser.add_argument('--split', type=str, default="val", choices=['train', 'val', 'test'],
                        help="评估数据集分割 (默认: val)")
    
    # 功能选项
    parser.add_argument('--skip-per-class', action='store_true',
                        help="跳过每类别详细评估 (需要模型和数据文件)")
    parser.add_argument('--classes', type=str, nargs='+', 
                        default=['Caries', 'Cavity', 'Crack', 'Tooth'],
                        help="类别名称列表 (默认: Caries Cavity Crack Tooth)")
    
    args = parser.parse_args()
    
    # 参数验证
    if not args.results and not (args.model and args.data):
        print("❌ 错误: 必须提供以下参数之一:")
        print("   1. --results: 训练结果CSV文件")
        print("   2. --model 和 --data: 模型文件和数据配置文件")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    print(f"📁 评估结果将保存至: {args.output}")
    
    # 处理类别名称
    class_names = args.classes
    
    # 如果提供了数据文件，尝试从中读取类别名称
    if args.data and os.path.exists(args.data):
        try:
            with open(args.data, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
                if 'names' in data_config:
                    class_names = data_config['names']
                    print(f"📋 从数据文件读取类别: {class_names}")
        except Exception as e:
            print(f"⚠️ 读取数据文件类别失败，使用默认类别: {e}")
    
    print(f"🏷️ 评估类别: {class_names}")
    
    # 1. 如果提供了results.csv，进行基于训练结果的评估
    if args.results:
        if not os.path.exists(args.results):
            print(f"❌ 找不到结果文件: {args.results}")
            return
        
        print("📊 开始基于训练结果的增强指标分析...")
        
        # 生成增强指标图表
        enhanced_plot_path = os.path.join(args.output, "enhanced_metrics_analysis.png")
        metrics = plot_enhanced_metrics(args.results, enhanced_plot_path, class_names)
        
        # 生成详细报告
        report_path = os.path.join(args.output, "metrics_report.md")
        generate_metrics_report(args.results, class_names, report_path)
        
        print(f"✅ 基于训练结果的评估完成:")
        print(f"   - 增强指标图表: {enhanced_plot_path}")
        print(f"   - 详细报告: {report_path}")
        
        # 显示关键指标摘要
        if metrics:
            print(f"🎯 关键指标摘要:")
            print(f"   - F1-Score: {metrics.get('f1_score', 0):.3f}")
            print(f"   - Precision: {metrics.get('precision', 0):.3f}")
            print(f"   - Recall: {metrics.get('recall', 0):.3f}")
            print(f"   - mAP@0.5: {metrics.get('map50', 0):.3f}")
            print(f"   - IoU质量: {metrics.get('avg_iou_at_0.5', 0):.3f}")
    
    # 2. 如果提供了模型和数据文件，进行完整的每类别评估
    if args.model and args.data and not args.skip_per_class:
        if not os.path.exists(args.model):
            print(f"❌ 找不到模型文件: {args.model}")
            return
        if not os.path.exists(args.data):
            print(f"❌ 找不到数据文件: {args.data}")
            return
        
        print(f"🔍 开始每类别详细评估 ({args.split}数据集)...")
        
        # 运行每类别评估
        per_class_metrics = evaluate_and_visualize_per_class(
            args.model, args.data, class_names, args.output
        )
        
        if per_class_metrics:
            print("✅ 每类别详细评估完成:")
            print(f"   - 每类别图表: {os.path.join(args.output, 'per_class_metrics.png')}")
            print(f"   - 每类别报告: {os.path.join(args.output, 'per_class_report.md')}")
            
            # 显示每类别F1-Score摘要
            print("🏆 每类别F1-Score:")
            for class_name, metrics in per_class_metrics.items():
                print(f"   - {class_name}: {metrics.get('f1_score', 0):.3f}")
    
    print(f"\n🎉 评估完成! 所有结果已保存至: {args.output}")


if __name__ == '__main__':
    main()
