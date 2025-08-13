"""
File and directory utilities for YOLOv8 training project
"""

import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime


def create_output_dirs(model_name, epochs, base_output_dir="outputs", enable_logs=True):
    """
    创建训练输出目录结构
    
    按照新的专业结构组织:
    - weights/: 模型文件 (best.pt, last.pt, epoch*.pt)
    - logs/: 过程性文件 (训练指标、记录图片)
    - analysis/: 结果分析 (评估报告、分析图表)
    - meta/: 训练基本情况 (配置、数据集信息)
    
    Args:
        model_name (str): 模型名称
        epochs (int): 训练轮数
        base_output_dir (str): 基础输出目录
        enable_logs (bool): 是否启用日志目录
        
    Returns:
        dict: 包含所有目录路径的字典
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base_dir = os.path.join(base_output_dir, f"train_{model_name}_{epochs}ep_{timestamp}")
    
    # 创建四个主要目录
    dirs = {
        'base': base_dir,
        'weights': os.path.join(base_dir, "weights"),
        'logs': os.path.join(base_dir, "logs"),
        'logs_records': os.path.join(base_dir, "logs", "records"),
        'analysis': os.path.join(base_dir, "analysis"), 
        'meta': os.path.join(base_dir, "meta")
    }
    
    # 创建所有目录
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def validate_files(model_file, data_yaml):
    """
    验证必要文件是否存在
    
    Args:
        model_file (str): 模型文件路径 (YOLOv8会自动下载预训练模型)
        data_yaml (str): 数据配置文件路径
        
    Raises:
        FileNotFoundError: 当数据文件不存在时抛出异常
    """
    # YOLOv8模型文件会自动下载，不需要验证
    # 只验证数据集配置文件
    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(f"数据集配置文件不存在: {data_yaml}")
    
    # 打印模型信息
    if os.path.isfile(model_file):
        print(f"📦 使用本地模型: {model_file}")
    else:
        print(f"📦 模型将自动下载: {model_file}")


def ensure_model_extension(model_name):
    """
    确保模型名称包含.pt扩展名，并返回models文件夹中的完整路径
    
    Args:
        model_name (str): 模型名称
        
    Returns:
        str: models文件夹中的完整模型文件路径
    """
    # 确保模型名有.pt扩展名
    if not model_name.endswith('.pt'):
        model_name = model_name + '.pt'
    
    # 返回models文件夹中的路径
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, model_name)


def reorganize_training_outputs(yolo_weights_dir, dirs, class_names):
    """
    重新组织训练输出文件到新的目录结构
    
    Args:
        yolo_weights_dir (str): YOLOv8原始weights目录路径
        dirs (dict): 新的目录结构字典
        class_names (list): 类别名称列表
    """
    print("🔄 正在重新组织输出文件...")
    
    try:
        # 1. 移动模型文件到 weights/
        weights_subdir = os.path.join(yolo_weights_dir, 'weights')
        if os.path.exists(weights_subdir):
            for pt_file in ['best.pt', 'last.pt']:
                src = os.path.join(weights_subdir, pt_file)
                dst = os.path.join(dirs['weights'], pt_file)
                if os.path.exists(src):
                    shutil.move(src, dst)
            
            # 移动epoch文件
            for file in os.listdir(weights_subdir):
                if file.startswith('epoch') and file.endswith('.pt'):
                    src = os.path.join(weights_subdir, file)
                    dst = os.path.join(dirs['weights'], file)
                    shutil.move(src, dst)
        
        # 2. 移动训练记录图片到 logs/records/
        record_files = [
            'train_batch0.jpg', 'train_batch1.jpg', 'train_batch2.jpg',
            'val_batch0_labels.jpg', 'val_batch0_pred.jpg',
            'val_batch1_labels.jpg', 'val_batch1_pred.jpg', 
            'val_batch2_labels.jpg', 'val_batch2_pred.jpg'
        ]
        
        for file in record_files:
            src = os.path.join(yolo_weights_dir, file)
            dst = os.path.join(dirs['logs_records'], file)
            if os.path.exists(src):
                shutil.move(src, dst)
        
        # 3. 移动分析文件到 analysis/
        analysis_files = [
            'BoxF1_curve.png', 'BoxPR_curve.png', 'BoxP_curve.png', 'BoxR_curve.png',
            'confusion_matrix.png', 'confusion_matrix_normalized.png'
        ]
        
        for file in analysis_files:
            src = os.path.join(yolo_weights_dir, file)
            dst = os.path.join(dirs['analysis'], file)
            if os.path.exists(src):
                shutil.move(src, dst)
        
        # 4. 移动meta文件到 meta/
        meta_files = ['args.yaml', 'labels.jpg', 'labels_correlogram.jpg']
        
        for file in meta_files:
            src = os.path.join(yolo_weights_dir, file)
            dst = os.path.join(dirs['meta'], file)
            if os.path.exists(src):
                shutil.move(src, dst)
        
        # 5. 合并训练指标文件
        merge_training_metrics(yolo_weights_dir, dirs)
        
        # 6. 合并训练指标图片
        merge_training_images(dirs)
        
        # 7. 创建分析README文档
        create_analysis_readme(dirs, class_names)
        
        # 8. 清理空的嵌套weights目录
        nested_weights = os.path.join(yolo_weights_dir, 'weights')
        if os.path.exists(nested_weights) and not os.listdir(nested_weights):
            os.rmdir(nested_weights)
            
        print("✅ 输出文件重组织完成!")
        
    except Exception as e:
        print(f"❌ 文件重组织失败: {e}")


def merge_training_metrics(yolo_weights_dir, dirs):
    """合并训练指标CSV文件"""
    try:
        results_csv = os.path.join(yolo_weights_dir, 'results.csv')
        complete_csv = os.path.join(dirs['base'], 'logs', 'complete_evaluation_metrics.csv')
        output_csv = os.path.join(dirs['logs'], 'training_metrics.csv')
        
        # 读取YOLOv8的results.csv
        results_data = None
        if os.path.exists(results_csv):
            results_data = pd.read_csv(results_csv)
            # 清理列名的空格
            results_data.columns = results_data.columns.str.strip()
        
        # 读取我们的complete_evaluation_metrics.csv  
        complete_data = None
        if os.path.exists(complete_csv):
            complete_data = pd.read_csv(complete_csv)
        
        # 合并数据并保存
        if results_data is not None:
            results_data.to_csv(output_csv, index=False)
            if complete_data is not None:
                # 在文件末尾添加分隔和完整评估数据
                with open(output_csv, 'a', encoding='utf-8') as f:
                    f.write('\n\n# Complete Evaluation Metrics\n')
                complete_data.to_csv(output_csv, mode='a', index=False)
        
        # 删除原始文件
        for file in [results_csv, complete_csv]:
            if os.path.exists(file):
                os.remove(file)
                
    except Exception as e:
        print(f"⚠️ 合并训练指标失败: {e}")


def merge_training_images(dirs):
    """处理训练指标图片 - 现在直接使用统一生成的图片，无需合并"""
    try:
        # 现在直接使用统一生成的training_metrics.png
        source_img = os.path.join(dirs['base'], 'logs', 'training_metrics.png')
        target_img = os.path.join(dirs['logs'], 'training_metrics.png')
        
        # 如果源图片存在且与目标不是同一个文件，则移动
        if os.path.exists(source_img) and source_img != target_img:
            import shutil
            shutil.move(source_img, target_img)
        
        # 清理可能存在的旧的分离图片文件
        old_files = [
            os.path.join(dirs['base'], 'logs', 'training_analysis.png'),
            os.path.join(dirs['base'], 'logs', 'enhanced_metrics_analysis.png')
        ]
        for old_file in old_files:
            if os.path.exists(old_file):
                os.remove(old_file)
        
    except Exception as e:
        print(f"⚠️ 处理训练图片失败: {e}")


def create_analysis_readme(dirs, class_names):
    """创建分析README文档"""
    try:
        # 读取原始报告
        metrics_report = os.path.join(dirs['base'], 'logs', 'metrics_report.md')
        per_class_report = os.path.join(dirs['base'], 'logs', 'per_class_report.md')
        output_readme = os.path.join(dirs['analysis'], 'README.md')
        
        # 移动per_class_metrics.png到analysis目录
        per_class_img = os.path.join(dirs['base'], 'logs', 'per_class_metrics.png')
        if os.path.exists(per_class_img):
            shutil.move(per_class_img, os.path.join(dirs['analysis'], 'per_class_metrics.png'))
        
        # 创建README内容
        readme_content = f"""# 训练结果分析报告

## 如何得到的训练结果分析

本分析基于YOLOv8训练完成后在验证集上的最终评估结果。具体来源：

### 指标计算方法

1. **整体指标** (精确率、召回率、F1-Score、mAP等)
   - 来源：训练完成后，使用最佳模型(`best.pt`)在验证集上进行评估
   - 计算方式：基于所有类别的平均值
   - 数据文件：`../logs/training_metrics.csv`

2. **每类别指标**
   - 来源：使用每类别评估器对最佳模型进行详细分析
   - 包括每个类别的精确率、召回率、F1-Score、AP@0.5等
   - 可视化：`per_class_metrics.png`

3. **混淆矩阵与曲线分析**
   - 基于验证集预测结果与真实标签的对比
   - 包括F1曲线、PR曲线、精确率曲线、召回率曲线
   - 反映模型在不同置信度阈值下的表现

## 训练结果综合分析

"""
        
        # 读取并整合原始报告内容
        if os.path.exists(metrics_report):
            with open(metrics_report, 'r', encoding='utf-8') as f:
                content = f.read()
                # 移除标题，只保留内容
                content = content.replace('# YOLOv8 牙齿检测模型训练报告', '').strip()
                readme_content += content + "\n\n"
        
        if os.path.exists(per_class_report):
            with open(per_class_report, 'r', encoding='utf-8') as f:
                content = f.read()
                # 移除标题，只保留内容
                content = content.replace('# 每类别详细指标报告', '### 每类别详细分析').strip()
                readme_content += content + "\n\n"
        
        # 添加文件说明
        readme_content += f"""
## 分析文件说明

### 图表文件
- `per_class_metrics.png` - 各类别指标对比图
- `BoxF1_curve.png` - F1-Score曲线 (不同置信度阈值)
- `BoxPR_curve.png` - 精确率-召回率曲线 
- `BoxP_curve.png` - 精确率曲线
- `BoxR_curve.png` - 召回率曲线
- `confusion_matrix.png` - 混淆矩阵 (原始数量)
- `confusion_matrix_normalized.png` - 标准化混淆矩阵 (百分比)

### 相关数据
- 训练过程数据：`../logs/training_metrics.csv`
- 训练过程图表：`../logs/training_metrics.png`
- 模型文件：`../weights/best.pt`

---
*分析报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        # 保存README
        with open(output_readme, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # 删除原始报告文件
        for file in [metrics_report, per_class_report]:
            if os.path.exists(file):
                os.remove(file)
                
    except Exception as e:
        print(f"⚠️ 创建分析README失败: {e}")
