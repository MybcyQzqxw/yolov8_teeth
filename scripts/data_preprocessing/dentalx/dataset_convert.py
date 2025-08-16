#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dentalx 疾病检测数据集转换脚本
将 quadrant-enumeration-disease 变体的 COCO 格式数据转换为 YOLO 格式
支持训练集、验证集、测试集的 7:2:1 比例划分
"""

import os
import json
import shutil
import random
import glob
from pathlib import Path

def coco_to_yolo_bbox(bbox, img_width, img_height):
    """将COCO格式的bbox转换为YOLO格式"""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return [x_center, y_center, w_norm, h_norm]

def convert_disease_dataset():
    """
    转换 quadrant-enumeration-disease 数据集为 YOLO 格式
    专门用于疾病检测任务
    """
    print("开始转换 dentalx 疾病检测数据集...")
    
    # 路径设置
    source_dir = "datasets/dentalx/quadrant-enumeration-disease"
    output_dir = "preprocessed_datasets/dentalx"
    
    if not os.path.exists(source_dir):
        print(f"错误: 源目录不存在: {source_dir}")
        print("请先运行 dataset_extract.py 解压数据集")
        return False
    
    # 查找JSON文件和图像目录
    json_file = os.path.join(source_dir, "train_quadrant_enumeration_disease.json")
    images_dir = os.path.join(source_dir, "xrays")
    
    if not os.path.exists(json_file):
        print(f"错误: JSON文件不存在: {json_file}")
        return False
        
    if not os.path.exists(images_dir):
        print(f"错误: 图像目录不存在: {images_dir}")
        return False
    
    print(f"- JSON文件: {json_file}")
    print(f"- 图像目录: {images_dir}")
    
    # 读取COCO数据
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取JSON文件: {e}")
        return False
    
    # 按照 COCO 格式处理疾病分类数据（使用 categories_3）
    if 'categories_3' not in coco_data:
        print("错误: JSON文件中没有 categories_3 字段")
        return False
    
    # 提取类别信息并排序
    categories = {cat['id']: cat['name'] for cat in coco_data['categories_3']}
    category_ids = list(categories.keys())
    category_ids.sort()  # 确保类别ID顺序一致
    class_names = [categories[cat_id] for cat_id in category_ids]
    
    print(f"- 发现 {len(class_names)} 个疾病类别: {class_names}")
    print(f"- 类别ID顺序: {category_ids}")
    
    # 创建输出目录结构
    train_img_dir = os.path.join(output_dir, 'train', 'images')
    train_label_dir = os.path.join(output_dir, 'train', 'labels')
    val_img_dir = os.path.join(output_dir, 'val', 'images')
    val_label_dir = os.path.join(output_dir, 'val', 'labels')
    test_img_dir = os.path.join(output_dir, 'test', 'images')
    test_label_dir = os.path.join(output_dir, 'test', 'labels')
    
    for directory in [train_img_dir, train_label_dir, val_img_dir, val_label_dir, test_img_dir, test_label_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # 建立图像信息映射
    images_info = {img['id']: img for img in coco_data['images']}
    
    # 整理标注数据（按图像ID分组）
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # 随机分割数据集（7:2:1 = 训练:验证:测试）
    image_ids = list(images_info.keys())
    random.seed(42)  # 固定随机种子，确保结果可复现
    random.shuffle(image_ids)
    
    total_count = len(image_ids)
    train_split = int(total_count * 0.7)
    val_split = int(total_count * 0.2)
    
    train_ids = image_ids[:train_split]
    val_ids = image_ids[train_split:train_split + val_split]
    test_ids = image_ids[train_split + val_split:]
    
    print(f"- 数据集分割: 训练集 {len(train_ids)} 张，验证集 {len(val_ids)} 张，测试集 {len(test_ids)} 张")
    
    # 转换函数
    def convert_split(ids, img_dir, label_dir, split_name):
        converted_count = 0
        for image_id in ids:
            image_info = images_info[image_id]
            img_width = image_info['width']
            img_height = image_info['height']
            image_filename = image_info['file_name']
            
            # 源图像路径
            src_img_path = os.path.join(images_dir, image_filename)
            if not os.path.exists(src_img_path):
                print(f"    警告: 图像文件不存在: {image_filename}")
                continue
            
            # 目标路径
            dst_img_path = os.path.join(img_dir, image_filename)
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            label_path = os.path.join(label_dir, label_filename)
            
            # 复制图像
            shutil.copy2(src_img_path, dst_img_path)
            
            # 生成 YOLO 格式标签
            yolo_annotations = []
            if image_id in annotations_by_image:
                for ann in annotations_by_image[image_id]:
                    bbox = ann['bbox']
                    x, y, w, h = bbox
                    center_x = (x + w / 2) / img_width
                    center_y = (y + h / 2) / img_height
                    norm_width = w / img_width
                    norm_height = h / img_height
                    
                    # 使用疾病分类的 category_id_3
                    category_id = ann['category_id_3']
                    class_index = category_ids.index(category_id)
                    
                    # YOLO 格式：class_id x_center y_center width height
                    yolo_annotations.append(f"{class_index} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
            
            # 写入标签文件
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            converted_count += 1
        
        print(f"  - {split_name}: {converted_count} 张图像")
        return converted_count
    
    # 转换训练集、验证集和测试集
    print("- 转换训练集...")
    train_count = convert_split(train_ids, train_img_dir, train_label_dir, "训练集")
    
    print("- 转换验证集...")
    val_count = convert_split(val_ids, val_img_dir, val_label_dir, "验证集")
    
    print("- 转换测试集...")
    test_count = convert_split(test_ids, test_img_dir, test_label_dir, "测试集")
    
    # 生成 YOLO 数据集配置文件
    yaml_content = f"""# dentalx 疾病检测数据集配置
path: {os.path.abspath(output_dir)}
train: train/images
val: val/images
test: test/images

nc: {len(class_names)}  # number of classes
names: {class_names}  # class names
"""
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    # 生成类别名称文件
    classes_path = os.path.join(output_dir, 'classes.txt')
    with open(classes_path, 'w', encoding='utf-8') as f:
        for cat_id in category_ids:
            f.write(f"{categories[cat_id]}\n")
    
    total_count = train_count + val_count + test_count
    print(f"\n转换完成！")
    print(f"- 总计处理: {total_count} 张图像")
    print(f"- 训练集: {train_count} 张 ({train_count/total_count*100:.1f}%)")
    print(f"- 验证集: {val_count} 张 ({val_count/total_count*100:.1f}%)")
    print(f"- 测试集: {test_count} 张 ({test_count/total_count*100:.1f}%)")
    print(f"- 输出目录: {output_dir}")
    print(f"- 配置文件: {yaml_path}")
    print(f"- 类别文件: {classes_path}")
    print(f"\n可以使用以下命令开始训练:")
    print(f"python scripts/train.py --data {yaml_path}")
    
    return True

if __name__ == "__main__":
    convert_disease_dataset()
