import json
import os
import shutil
import argparse

def get_category_id_map(categories):
    """获取类别ID映射"""
    return {cat['id']: idx for idx, cat in enumerate(categories)}

def convert_coco_to_yolo(coco_json_path, output_dir):
    """将COCO格式的标注转换为YOLO格式"""
    print(f"转换标注文件：{coco_json_path} -> {output_dir}")
    
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    
    images = {img['id']: img for img in coco['images']}
    categories = coco['categories']
    cat_id_map = get_category_id_map(categories)
    
    # 构建图像ID到标注的映射
    img_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for img_id, img in images.items():
        file_name = os.path.splitext(img['file_name'])[0] + '.txt'
        label_path = os.path.join(output_dir, file_name)
        width = img['width']
        height = img['height']
        
        anns = img_to_anns.get(img_id, [])
        lines = []
        
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in cat_id_map:
                print(f"警告: 未知的category_id {cat_id}，已跳过该标注。")
                continue
            
            yolo_cat_id = cat_id_map[cat_id]
            bbox = ann['bbox']
            
            # COCO格式: [x_min, y_min, width, height]
            # YOLO格式: [x_center, y_center, width, height] (归一化)
            x_center = (bbox[0] + bbox[2] / 2) / width
            y_center = (bbox[1] + bbox[3] / 2) / height
            w = bbox[2] / width
            h = bbox[3] / height
            
            lines.append(f"{yolo_cat_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    return categories

def convert_dataset(src_dir, dest_dir):
    """转换整个数据集"""
    print(f"开始转换数据集：{src_dir} -> {dest_dir}")
    
    # 检查源目录结构
    annotations_dir = os.path.join(src_dir, "annotations")
    train_images_dir = os.path.join(src_dir, "train2017")
    val_images_dir = os.path.join(src_dir, "val2017")
    
    required_files = [
        os.path.join(annotations_dir, "instances_train2017.json"),
        os.path.join(annotations_dir, "instances_val2017.json"),
        train_images_dir,
        val_images_dir
    ]
    
    for required_file in required_files:
        if not os.path.exists(required_file):
            raise FileNotFoundError(f"找不到必需的文件或目录: {required_file}")
    
    # 创建目标目录结构
    os.makedirs(os.path.join(dest_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "val", "labels"), exist_ok=True)
    
    # 复制图像文件
    print("复制训练图像...")
    dest_train_images = os.path.join(dest_dir, "train", "images")
    if os.path.exists(dest_train_images):
        shutil.rmtree(dest_train_images)
    shutil.copytree(train_images_dir, dest_train_images)
    
    print("复制验证图像...")
    dest_val_images = os.path.join(dest_dir, "val", "images")
    if os.path.exists(dest_val_images):
        shutil.rmtree(dest_val_images)
    shutil.copytree(val_images_dir, dest_val_images)
    
    # 转换标注文件
    print("转换训练标注...")
    train_ann_path = os.path.join(annotations_dir, "instances_train2017.json")
    train_labels_dir = os.path.join(dest_dir, "train", "labels")
    categories = convert_coco_to_yolo(train_ann_path, train_labels_dir)
    
    print("转换验证标注...")
    val_ann_path = os.path.join(annotations_dir, "instances_val2017.json")
    val_labels_dir = os.path.join(dest_dir, "val", "labels")
    convert_coco_to_yolo(val_ann_path, val_labels_dir)
    
    # 生成data.yaml配置文件
    print("生成data.yaml配置文件...")
    names = [cat['name'] for cat in categories]
    nc = len(names)
    
    yaml_content = "train: train/images\n"
    yaml_content += "val: val/images\n\n"
    yaml_content += f"nc: {nc}\n"
    yaml_content += "names:\n"
    for name in names:
        yaml_content += f"  - {name}\n"
    
    yaml_path = os.path.join(dest_dir, "data.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"已生成配置文件: {yaml_path}")
    
    # 统计信息
    train_images_count = len([f for f in os.listdir(dest_train_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    val_images_count = len([f for f in os.listdir(dest_val_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    train_labels_count = len([f for f in os.listdir(train_labels_dir) if f.endswith('.txt')])
    val_labels_count = len([f for f in os.listdir(val_labels_dir) if f.endswith('.txt')])
    
    print(f"\n转换完成！统计信息：")
    print(f"├── 训练集: {train_images_count} 张图片, {train_labels_count} 个标签文件")
    print(f"├── 验证集: {val_images_count} 张图片, {val_labels_count} 个标签文件")
    print(f"├── 类别数量: {nc}")
    print(f"└── 类别名称: {', '.join(names)}")

def main():
    parser = argparse.ArgumentParser(description="将oralxrays9数据集从COCO格式转换为YOLO格式")
    parser.add_argument("--src_dir", type=str, default="./datasets/oralxrays9",
                        help="源数据集目录")
    parser.add_argument("--dest_dir", type=str, default="./preprocessed_datasets/oralxrays9",
                        help="目标数据集目录")
    args = parser.parse_args()

    try:
        convert_dataset(args.src_dir, args.dest_dir)
        print(f"\n数据集转换完成! 目标目录: {args.dest_dir}")
        print("\n现在可以使用以下命令训练模型：")
        print(f"python scripts/train.py --data_config {args.dest_dir}/data.yaml --dataset_name oralxrays9")
    except Exception as e:
        print(f"转换失败: {e}")

if __name__ == "__main__":
    main()
