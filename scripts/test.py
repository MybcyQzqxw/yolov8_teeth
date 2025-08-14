#!/usr/bin/env python3
"""
YOLOv8 牙齿检测模型测试脚本
在测试集上评估训练好的模型性能，生成详细的评估报告和可视化结果
"""

import argparse
import os
import sys
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
import pandas as pd
from ultralytics import YOLO
import yaml
from datetime import datetime

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.file_utils import create_output_dirs, validate_files
from utils.visualization import plot_training_metrics
from utils.metrics import generate_metrics_report, enhanced_metrics_analysis

# 设置matplotlib中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def find_latest_model(output_dir):
    """
    在输出目录中查找最新训练的模型
    
    Args:
        output_dir (str): 输出目录路径
        
    Returns:
        str: 最新模型的路径，如果未找到返回None
    """
    if not os.path.exists(output_dir):
        return None
    
    # 查找所有训练目录
    train_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith('train_'):
            train_dirs.append((item, item_path))
    
    if not train_dirs:
        return None
    
    # 按时间戳排序，取最新的
    train_dirs.sort(key=lambda x: x[0], reverse=True)
    
    # 查找第一个有效的模型文件
    for _, train_dir in train_dirs:
        possible_models = [
            os.path.join(train_dir, "weights", "best.pt"),
            os.path.join(train_dir, "weights", "last.pt")
        ]
        
        for model_path in possible_models:
            if os.path.exists(model_path):
                return model_path
    
    return None

def load_ground_truth_labels(label_path):
    """
    加载真实标签
    
    Args:
        label_path (str): 标签文件路径
        
    Returns:
        list: 标签列表，每个元素为[class_id, x_center, y_center, width, height]
    """
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([float(x) for x in parts])
    return labels

def denormalize_bbox(bbox, img_width, img_height):
    """
    将YOLO格式的归一化边界框转换为像素坐标
    
    Args:
        bbox (list): [x_center, y_center, width, height] (归一化)
        img_width (int): 图像宽度
        img_height (int): 图像高度
        
    Returns:
        tuple: (x_min, y_min, x_max, y_max) 像素坐标
    """
    x_center, y_center, width, height = bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    
    return int(x_min), int(y_min), int(x_max), int(y_max)

def visualize_predictions_vs_labels(image_paths, label_dir, model, class_names, output_dir, num_samples=10):
    """
    可视化预测结果与真实标签的对比
    
    Args:
        image_paths (list): 图像路径列表
        label_dir (str): 标签目录
        model: YOLO模型
        class_names (list): 类别名称列表
        output_dir (str): 输出目录
        num_samples (int): 采样数量
    """
    # 随机选择图像
    selected_images = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    # 创建子图
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    fig.suptitle('测试集预测结果对比 (绿色: 真实标签, 红色: 预测结果)', fontsize=16, fontweight='bold')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for idx, image_path in enumerate(selected_images):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]
        
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        # 获取对应的标签文件
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(label_dir, f"{image_name}.txt")
        
        # 加载真实标签
        gt_labels = load_ground_truth_labels(label_path)
        
        # 预测
        results = model.predict(image_path, verbose=False)
        
        # 显示图像
        ax.imshow(image)
        ax.set_title(f'图片 {idx + 1}: {os.path.basename(image_path)}', fontsize=10)
        ax.axis('off')
        
        # 绘制真实标签 (绿色)
        for label in gt_labels:
            class_id, x_center, y_center, width, height = label
            x_min, y_min, x_max, y_max = denormalize_bbox(
                [x_center, y_center, width, height], img_width, img_height
            )
            
            # 绘制边界框
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='green', facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)
            
            # 添加类别标签
            class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f'Class{int(class_id)}'
            ax.text(x_min, y_min - 5, f'GT: {class_name}', 
                   fontsize=8, color='green', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # 绘制预测结果 (红色)
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # 获取边界框坐标
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # 只显示置信度较高的预测
                if confidence > 0.3:
                    # 绘制边界框
                    rect = patches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=2, edgecolor='red', facecolor='none', alpha=0.8
                    )
                    ax.add_patch(rect)
                    
                    # 添加类别标签和置信度
                    class_name = class_names[class_id] if class_id < len(class_names) else f'Class{class_id}'
                    ax.text(x_min, y_max + 15, f'Pred: {class_name} ({confidence:.2f})', 
                           fontsize=8, color='red', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # 保存图像
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'test_predictions_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] 预测对比可视化已保存至: {save_path}")

def plot_training_metrics(csv_path, output_path):
    """绘制训练指标图表"""
    try:
        import pandas as pd
        
        # 读取CSV数据
        df = pd.read_csv(csv_path)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 指标映射
        metrics_map = {
            'metrics/precision(B)': 'Precision',
            'metrics/recall(B)': 'Recall', 
            'metrics/mAP50(B)': 'mAP@0.5',
            'metrics/mAP50-95(B)': 'mAP@0.5:0.95'
        }
        
        epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
        
        for idx, (col_name, label) in enumerate(metrics_map.items()):
            row = idx // 2
            col = idx % 2
            
            if col_name in df.columns:
                axes[row, col].plot(epochs, df[col_name], marker='o', linewidth=2, markersize=4)
                axes[row, col].set_title(f'{label}', fontsize=12, fontweight='bold')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel(label)
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   [✓] 测试指标可视化已保存: {output_path}")
        
    except Exception as e:
        print(f"   ⚠️ 绘制测试指标图表时出错: {e}")


def _save_complete_evaluation_csv(metrics, per_class_metrics, class_names, csv_path):
    """保存完整的评估指标到CSV文件"""
    import pandas as pd
    
    # 创建整体指标数据
    overall_data = {
        '类别': ['Overall'],
        '精确率': [metrics['precision']],
        '召回率': [metrics['recall']],
        'F1-Score': [metrics['f1_score']],
        'mAP@0.5': [metrics['map50']],
        'mAP@0.5:0.95': [metrics['map50_95']]
    }
    
    # 添加每类别指标
    if per_class_metrics:
        for i, class_name in enumerate(class_names):
            if i < len(per_class_metrics['precision']):
                overall_data['类别'].append(class_name)
                overall_data['精确率'].append(per_class_metrics['precision'][i])
                overall_data['召回率'].append(per_class_metrics['recall'][i])
                overall_data['F1-Score'].append(per_class_metrics['f1'][i])
                overall_data['mAP@0.5'].append(per_class_metrics['ap50'][i])
                overall_data['mAP@0.5:0.95'].append(per_class_metrics['ap'][i])
    
    # 保存到CSV
    df = pd.DataFrame(overall_data)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    

def run_test_evaluation(model_path, data_yaml, output_dir):
    """
    运行完整的测试评估
    
    Args:
        model_path (str): 模型文件路径
        data_yaml (str): 数据配置文件路径
        output_dir (str): 输出目录
        
    Returns:
        dict: 评估结果
    """
    print("🔄 正在加载模型...")
    model = YOLO(model_path)
    print("✅ 模型加载成功!")
    
    # 读取数据配置
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    test_dir = data_config.get('test', 'test')
    class_names = data_config.get('names', ['Unknown'])
    
    print(f"🔍 开始在测试集上评估模型...")
    print(f"   📂 测试数据: {test_dir}")
    print(f"   🏷️  类别数量: {len(class_names)}")
    print(f"   📊 类别名称: {class_names}")
    
    # 创建分析目录
    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 运行验证并保存详细结果
    results = model.val(
        data=data_yaml, 
        split='test', 
        save_json=True, 
        save_hybrid=True,
        plots=True,  # 生成所有分析图表
        save_dir=analysis_dir,  # 保存到分析目录
        name='test_analysis'  # 指定子目录名
    )
    
    if results is None:
        print("❌ 模型评估失败")
        return None
    
    # 提取关键指标
    metrics = {
        'precision': float(results.box.mp),  # 平均精确率
        'recall': float(results.box.mr),     # 平均召回率
        'map50': float(results.box.map50),   # mAP@0.5
        'map50_95': float(results.box.map),  # mAP@0.5:0.95
        'f1_score': 2 * float(results.box.mp) * float(results.box.mr) / (float(results.box.mp) + float(results.box.mr)) if (float(results.box.mp) + float(results.box.mr)) > 0 else 0,
    }
    
    print(f"📊 测试集评估结果:")
    print(f"   - 精确率 (Precision): {metrics['precision']:.4f}")
    print(f"   - 召回率 (Recall): {metrics['recall']:.4f}")
    print(f"   - F1-Score: {metrics['f1_score']:.4f}")
    print(f"   - mAP@0.5: {metrics['map50']:.4f}")
    print(f"   - mAP@0.5:0.95: {metrics['map50_95']:.4f}")
    
    # 复制YOLOv8生成的分析图表到主分析目录
    # YOLOv8会将结果保存到outputs目录下，我们需要找到实际的保存位置
    possible_yolo_dirs = [
        os.path.join(analysis_dir, 'test_analysis'),
        os.path.join('outputs', 'dentalai', 'detect', 'test_analysis'),
        os.path.join('outputs', 'dentalai', 'detect', 'test_analysis2'),
        os.path.join('outputs', 'dentalai', 'detect', 'test_analysis3'),
    ]
    
    yolo_analysis_dir = None
    for possible_dir in possible_yolo_dirs:
        if os.path.exists(possible_dir):
            yolo_analysis_dir = possible_dir
            break
    
    if yolo_analysis_dir and os.path.exists(yolo_analysis_dir):
        print(f"📊 正在整理分析图表...")
        print(f"   从 {yolo_analysis_dir} 复制到 {analysis_dir}")
        
        # 需要复制的文件
        analysis_files = [
            'confusion_matrix.png',
            'confusion_matrix_normalized.png', 
            'BoxF1_curve.png',
            'BoxPR_curve.png',
            'BoxP_curve.png',
            'BoxR_curve.png',
            'val_batch0_labels.jpg',
            'val_batch0_pred.jpg'
        ]
        
        for file_name in analysis_files:
            src_file = os.path.join(yolo_analysis_dir, file_name)
            dst_file = os.path.join(analysis_dir, file_name)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                print(f"   [✓] {file_name}")
            else:
                print(f"   [×] {file_name} (未找到)")
    else:
        print(f"⚠️ 未找到YOLOv8生成的分析图表目录")
        
        # 生成训练指标可视化（如果有results.csv）
        results_csv = os.path.join(yolo_analysis_dir, 'results.csv')
        if os.path.exists(results_csv):
            metrics_plot_path = os.path.join(analysis_dir, "test_metrics.png")
            plot_training_metrics(results_csv, metrics_plot_path)
            print(f"   [✓] test_metrics.png")
    
    # 生成每类别详细评估（复用训练时的功能）
    print(f"🔍 开始每类别详细指标评估...")
    try:
        from utils.per_class_evaluator import evaluate_and_visualize_per_class
        per_class_metrics = evaluate_and_visualize_per_class(
            model_path, data_yaml, class_names, analysis_dir, split='test'
        )
        
        if per_class_metrics:
            # 生成完整的评估数据CSV文件
            evaluation_csv_path = os.path.join(output_dir, "complete_test_evaluation.csv")
            _save_complete_evaluation_csv(metrics, per_class_metrics, class_names, evaluation_csv_path)
            print(f"[✓] 完整评估指标CSV已保存: {evaluation_csv_path}")
    except Exception as e:
        print(f"⚠️ 每类别评估出现问题: {e}")
        per_class_metrics = None
    
    # 获取测试图像列表进行可视化对比
    base_dir = os.path.dirname(data_yaml)
    test_images_dir = os.path.join(base_dir, 'test', 'images')
    test_labels_dir = os.path.join(base_dir, 'test', 'labels')
    
    if os.path.exists(test_images_dir):
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            image_paths.extend(Path(test_images_dir).glob(f'*{ext}'))
            image_paths.extend(Path(test_images_dir).glob(f'*{ext.upper()}'))
        
        image_paths = [str(p) for p in image_paths]
        print(f"   📸 测试图像数量: {len(image_paths)}")
        
        if len(image_paths) > 0:
            print(f"🎨 生成预测对比可视化...")
            visualize_predictions_vs_labels(
                image_paths, test_labels_dir, model, class_names, analysis_dir
            )
    
    # 生成详细的测试指标报告
    print(f"📋 生成详细的测试指标报告...")
    try:
        from utils.metrics import generate_metrics_report, enhanced_metrics_analysis
        
        # 使用YOLOv8生成的results.csv
        results_csv = os.path.join(yolo_analysis_dir, 'results.csv') if os.path.exists(os.path.join(yolo_analysis_dir, 'results.csv')) else None
        
        if results_csv and os.path.exists(results_csv):
            # 计算增强指标分析
            enhanced_metrics = enhanced_metrics_analysis(results_csv, class_names)
            
            # 生成详细报告
            report_path = os.path.join(output_dir, "detailed_test_report.md") 
            generate_metrics_report(results_csv, class_names, report_path)
            print(f"[✓] 详细测试报告已保存: {report_path}")
        else:
            print(f"⚠️ 未找到results.csv，跳过详细报告生成")
            
    except Exception as e:
        print(f"⚠️ 生成详细报告时出现问题: {e}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 牙齿检测模型测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python scripts/test.py                                    # 使用默认参数测试
  python scripts/test.py -m ./outputs/train_xxx/weights/best.pt  # 指定模型文件
  python scripts/test.py -d ./my_dataset                   # 指定数据集目录
  python scripts/test.py -o ./test_results                 # 指定输出目录
        """)
    
    # 模型参数
    parser.add_argument('--model', '-m', type=str, default=None,
                        help="模型文件路径，默认自动查找最新训练的模型")
    
    # 数据参数
    parser.add_argument('--data', '-d', type=str, default="./preprocessed_datasets/dentalai/data.yaml",
                        help="数据配置文件路径 (默认: ./preprocessed_datasets/dentalai/data.yaml)")
    parser.add_argument('--output_dir', '-o', type=str, default="./test_results",
                        help="测试结果输出目录 (默认: ./test_results)")
    
    # 可视化参数
    parser.add_argument('--samples', '-s', type=int, default=10,
                        help="可视化对比的样本数量 (默认: 10)")
    parser.add_argument('--conf_threshold', '-c', type=float, default=0.3,
                        help="预测置信度阈值 (默认: 0.3)")
    
    args = parser.parse_args()
    
    # 验证数据配置文件
    data_yaml = args.data
    if not os.path.exists(data_yaml):
        print(f"❌ 数据配置文件不存在: {data_yaml}")
        return
    
    # 确定模型文件
    if args.model:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"❌ 指定的模型文件不存在: {model_path}")
            return
    else:
        # 自动查找最新模型
        print("🔍 正在查找最新训练的模型...")
        model_path = find_latest_model("./outputs/dentalai")
        if not model_path:
            print("❌ 未找到训练好的模型文件")
            print("💡 请先训练模型或使用 -m 参数指定模型文件路径")
            return
    
    print(f"🎯 使用模型: {model_path}")
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    timestamped_output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(timestamped_output_dir, exist_ok=True)
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    print(f"🚀 开始测试 YOLOv8 牙齿检测模型")
    print(f"   📦 模型文件: {model_path}")
    print(f"   📁 数据配置: {data_yaml}")
    print(f"   💾 输出目录: {timestamped_output_dir}")
    print(f"   🎨 可视化样本: {args.samples}")
    print(f"   🎯 置信度阈值: {args.conf_threshold}")
    
    try:
        # 运行测试评估
        metrics = run_test_evaluation(model_path, data_yaml, timestamped_output_dir)
        
        if metrics:
            # 保存测试结果
            results_file = os.path.join(timestamped_output_dir, 'test_results.json')
            import json
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"[✓] 测试结果已保存至: {results_file}")
            
            # 生成测试报告
            report_file = os.path.join(timestamped_output_dir, 'test_report.md')
            generate_test_report(metrics, model_path, data_yaml, report_file)
            print(f"[✓] 测试报告已保存至: {report_file}")
            
            print(f"\n✅ 测试完成! 结果保存至: {timestamped_output_dir}")
            print(f"📊 关键指标:")
            print(f"   - F1-Score: {metrics['f1_score']:.4f}")
            print(f"   - 精确率: {metrics['precision']:.4f}")
            print(f"   - 召回率: {metrics['recall']:.4f}")
            print(f"   - mAP@0.5: {metrics['map50']:.4f}")
        else:
            print("❌ 测试失败")
    
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def generate_test_report(metrics, model_path, data_yaml, report_path):
    """
    生成测试报告
    
    Args:
        metrics (dict): 测试指标
        model_path (str): 模型路径
        data_yaml (str): 数据配置路径
        report_path (str): 报告保存路径
    """
    import datetime
    
    report_content = f"""# YOLOv8 牙齿检测模型测试报告

## 基本信息
- **测试时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **模型文件**: `{model_path}`
- **数据配置**: `{data_yaml}`

## 测试结果

### 整体性能指标
| 指标 | 数值 |
|------|------|
| 精确率 (Precision) | {metrics['precision']:.4f} |
| 召回率 (Recall) | {metrics['recall']:.4f} |
| F1-Score | {metrics['f1_score']:.4f} |
| mAP@0.5 | {metrics['map50']:.4f} |
| mAP@0.5:0.95 | {metrics['map50_95']:.4f} |

### 性能分析

#### 📊 指标解读
- **精确率 ({metrics['precision']:.4f})**: 在所有预测为正例的样本中，真正为正例的比例
- **召回率 ({metrics['recall']:.4f})**: 在所有真正为正例的样本中，被正确预测为正例的比例  
- **F1-Score ({metrics['f1_score']:.4f})**: 精确率和召回率的调和平均数，综合评价指标
- **mAP@0.5 ({metrics['map50']:.4f})**: IoU阈值为0.5时的平均精度
- **mAP@0.5:0.95 ({metrics['map50_95']:.4f})**: IoU阈值从0.5到0.95的平均精度

#### 🎯 性能评估
"""

    # 性能等级评估
    f1_score = metrics['f1_score']
    if f1_score >= 0.9:
        performance_level = "🏆 优秀"
    elif f1_score >= 0.8:
        performance_level = "🥈 良好"
    elif f1_score >= 0.7:
        performance_level = "🥉 一般"
    else:
        performance_level = "⚠️ 需要改进"
    
    report_content += f"- **整体性能**: {performance_level} (F1-Score: {f1_score:.4f})\n"
    
    # 添加建议
    report_content += f"""
#### 💡 优化建议
"""
    
    if metrics['precision'] > metrics['recall']:
        report_content += "- 精确率高于召回率，模型倾向于保守预测，可考虑降低置信度阈值以提高召回率\n"
    elif metrics['recall'] > metrics['precision']:
        report_content += "- 召回率高于精确率，模型可能存在过拟合，可考虑提高置信度阈值或增加正则化\n"
    else:
        report_content += "- 精确率和召回率较为平衡，模型性能稳定\n"
    
    if f1_score < 0.7:
        report_content += "- F1-Score较低，建议:\n"
        report_content += "  - 增加训练数据量\n"
        report_content += "  - 调整模型架构或超参数\n"
        report_content += "  - 检查数据质量和标注准确性\n"
    
    report_content += """
## 可视化结果

测试过程中生成了以下可视化结果：
- `test_predictions_comparison.png`: 随机选取10张图片的预测结果对比

## 文件说明

- `test_results.json`: 详细的数值结果
- `test_report.md`: 本测试报告  
- `test_predictions_comparison.png`: 预测结果可视化

---
*报告由 YOLOv8 牙齿检测测试脚本自动生成*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

if __name__ == '__main__':
    main()
