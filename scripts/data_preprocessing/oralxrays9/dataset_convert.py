import json
import os
import shutil
import argparse
import random
from sklearn.model_selection import train_test_split

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

def convert_coco_subset_to_yolo(coco_data, image_names, img_name_to_id, img_id_to_info, output_dir, categories):
    """将COCO格式的子集转换为YOLO格式"""
    cat_id_map = get_category_id_map(categories)
    
    # 构建图像ID到标注的映射
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name in image_names:
        img_id = img_name_to_id.get(img_name)
        if img_id is None:
            print(f"警告: 找不到图像 {img_name} 的ID")
            continue
            
        img_info = img_id_to_info[img_id]
        file_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(output_dir, file_name)
        width = img_info['width']
        height = img_info['height']
        
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

def convert_dataset(src_dir, dest_dir, test_split=0.3):
    """转换整个数据集，并将验证集分割为验证集和测试集"""
    print(f"开始转换数据集：{src_dir} -> {dest_dir}")
    print(f"测试集比例：{test_split * 100:.1f}% (从验证集中分割)")
    
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
    
    # 创建目标目录结构（包含测试集）
    os.makedirs(os.path.join(dest_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "val", "labels"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "test", "images"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "test", "labels"), exist_ok=True)
    
    # 复制训练图像文件
    print("复制训练图像...")
    dest_train_images = os.path.join(dest_dir, "train", "images")
    if os.path.exists(dest_train_images):
        shutil.rmtree(dest_train_images)
    shutil.copytree(train_images_dir, dest_train_images)
    
    # 获取验证集图像列表，进行分割
    print("分割验证集为验证集和测试集...")
    val_images = [f for f in os.listdir(val_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 随机分割验证集
    random.seed(42)  # 设置随机种子确保可重复性
    val_images_new, test_images = train_test_split(val_images, test_size=test_split, random_state=42)
    
    print(f"验证集图片数量: {len(val_images_new)}")
    print(f"测试集图片数量: {len(test_images)}")
    
    # 复制验证集图像
    print("复制验证集图像...")
    dest_val_images = os.path.join(dest_dir, "val", "images")
    if os.path.exists(dest_val_images):
        shutil.rmtree(dest_val_images)
    os.makedirs(dest_val_images, exist_ok=True)
    
    for img_name in val_images_new:
        src_path = os.path.join(val_images_dir, img_name)
        dst_path = os.path.join(dest_val_images, img_name)
        shutil.copy2(src_path, dst_path)
    
    # 复制测试集图像
    print("复制测试集图像...")
    dest_test_images = os.path.join(dest_dir, "test", "images")
    if os.path.exists(dest_test_images):
        shutil.rmtree(dest_test_images)
    os.makedirs(dest_test_images, exist_ok=True)
    
    for img_name in test_images:
        src_path = os.path.join(val_images_dir, img_name)
        dst_path = os.path.join(dest_test_images, img_name)
        shutil.copy2(src_path, dst_path)
    
    # 转换训练标注文件
    print("转换训练标注...")
    train_ann_path = os.path.join(annotations_dir, "instances_train2017.json")
    train_labels_dir = os.path.join(dest_dir, "train", "labels")
    categories = convert_coco_to_yolo(train_ann_path, train_labels_dir)
    
    # 转换验证集标注文件
    print("转换验证集标注...")
    val_ann_path = os.path.join(annotations_dir, "instances_val2017.json")
    
    # 加载验证集标注文件
    with open(val_ann_path, 'r', encoding='utf-8') as f:
        val_coco = json.load(f)
    
    # 创建图像名到ID的映射
    val_img_name_to_id = {}
    val_img_id_to_info = {}
    for img in val_coco['images']:
        val_img_name_to_id[img['file_name']] = img['id']
        val_img_id_to_info[img['id']] = img
    
    # 分别处理验证集和测试集的标注
    val_labels_dir = os.path.join(dest_dir, "val", "labels")
    test_labels_dir = os.path.join(dest_dir, "test", "labels")
    
    # 处理验证集标注
    convert_coco_subset_to_yolo(val_coco, val_images_new, val_img_name_to_id, val_img_id_to_info, val_labels_dir, categories)
    
    # 处理测试集标注  
    convert_coco_subset_to_yolo(val_coco, test_images, val_img_name_to_id, val_img_id_to_info, test_labels_dir, categories)
    
    # 生成data.yaml配置文件
    print("生成data.yaml配置文件...")
    names = [cat['name'] for cat in categories]
    nc = len(names)
    
    yaml_content = "train: train/images\n"
    yaml_content += "val: val/images\n"
    yaml_content += "test: test/images\n\n"
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
    test_images_count = len([f for f in os.listdir(dest_test_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    train_labels_count = len([f for f in os.listdir(train_labels_dir) if f.endswith('.txt')])
    val_labels_count = len([f for f in os.listdir(val_labels_dir) if f.endswith('.txt')])
    test_labels_count = len([f for f in os.listdir(test_labels_dir) if f.endswith('.txt')])
    
    print(f"\n转换完成！统计信息：")
    print(f"├── 训练集: {train_images_count} 张图片, {train_labels_count} 个标签文件")
    print(f"├── 验证集: {val_images_count} 张图片, {val_labels_count} 个标签文件")
    print(f"├── 测试集: {test_images_count} 张图片, {test_labels_count} 个标签文件")
    print(f"├── 类别数量: {nc}")
    print(f"└── 类别名称: {', '.join(names)}")

def main():
    parser = argparse.ArgumentParser(description="将oralxrays9数据集从COCO格式转换为YOLO格式")
    parser.add_argument("--src_dir", type=str, default="./datasets/oralxrays9",
                        help="源数据集目录")
    parser.add_argument("--dest_dir", type=str, default="./preprocessed_datasets/oralxrays9",
                        help="目标数据集目录")
    parser.add_argument("--test_split", type=float, default=0.3,
                        help="测试集比例 (从验证集中分割，默认: 0.3)")
    args = parser.parse_args()

    try:
        convert_dataset(args.src_dir, args.dest_dir, args.test_split)
        print(f"\n数据集转换完成! 目标目录: {args.dest_dir}")
    except Exception as e:
        print(f"转换失败: {e}")

if __name__ == "__main__":
    main()
