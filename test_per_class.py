#!/usr/bin/env python3
"""
测试每类别评估功能的修复
"""
import os
import sys
sys.path.append('.')
from utils.yolov8.per_class_evaluator import evaluate_and_visualize_per_class

def test_per_class_evaluation():
    # 使用已有的训练结果进行测试
    model_path = 'outputs/yolov8/train_yolov8m_1ep_2025_08_13_13_49_34/weights/weights/best.pt'
    data_path = 'preprocessed_datasets/yolov8/data.yaml'
    class_names = ['Caries', 'Cavity', 'Crack', 'Tooth']
    output_dir = 'outputs/yolov8/train_yolov8m_1ep_2025_08_13_13_49_34/logs'

    print('🧪 测试每类别评估功能...')
    print(f'📦 模型路径: {model_path}')
    print(f'📊 数据路径: {data_path}')
    print(f'📁 输出目录: {output_dir}')

    if os.path.exists(model_path):
        print('✅ 模型文件存在，开始评估...')
        try:
            metrics = evaluate_and_visualize_per_class(model_path, data_path, class_names, output_dir)
            if metrics:
                print('✅ 每类别评估成功完成!')
                for class_name, class_metrics in metrics.items():
                    f1_score = class_metrics.get('f1_score', 0)
                    print(f'   {class_name}: F1={f1_score:.3f}')
                return True
            else:
                print('⚠️ 评估返回空结果')
                return False
        except Exception as e:
            print(f'❌ 评估失败: {e}')
            return False
    else:
        print('❌ 模型文件不存在')
        return False

if __name__ == '__main__':
    success = test_per_class_evaluation()
    exit(0 if success else 1)
