import shutil
from os import path
import os
import json
import argparse

def convert_dataset(src_dir, dest_dir, classes_path):
    # 构建目标数据集目录结构
    os.makedirs(path.join(dest_dir, "train", "images"), exist_ok=True)
    os.makedirs(path.join(dest_dir, "train", "labels"), exist_ok=True)
    os.makedirs(path.join(dest_dir, "val", "images"), exist_ok=True)
    os.makedirs(path.join(dest_dir, "val", "labels"), exist_ok=True)
    os.makedirs(path.join(dest_dir, "test", "images"), exist_ok=True)
    os.makedirs(path.join(dest_dir, "test", "labels"), exist_ok=True)

    # 读取 meta.json 获取类别信息
    if not path.isfile(classes_path):
        raise FileNotFoundError(f"找不到 meta.json 文件: {classes_path}")
    with open(classes_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    classes = {entry["title"]: idx for idx, entry in enumerate(meta["classes"])}

    # 写 data.yaml
    with open(path.join(dest_dir, "data.yaml"), "w", encoding="utf-8") as fp:
        fp.write("train: train/images\n")
        fp.write("val: val/images\n")
        fp.write("test: test/images\n\n")
        fp.write(f"nc: {len(classes)}\n")
        fp.write("names: ['{}']\n".format("','".join(classes.keys())))

    # 目录映射
    dirs_map = {"train": "train", "valid": "val", "test": "test"}

    for src_subdir, dest_subdir in dirs_map.items():
        # 复制图像
        src_img_dir = path.join(src_dir, src_subdir, "img")
        dest_img_dir = path.join(dest_dir, dest_subdir, "images")
        if path.exists(src_img_dir):
            shutil.copytree(src_img_dir, dest_img_dir, dirs_exist_ok=True)
        else:
            print(f"警告：未找到图像目录 {src_img_dir}")

        # 转换注释
        src_ann_dir = path.join(src_dir, src_subdir, "ann")
        dest_label_dir = path.join(dest_dir, dest_subdir, "labels")
        if not path.exists(src_ann_dir):
            print(f"警告：未找到注释目录 {src_ann_dir}")
            continue

        for ann_file in os.listdir(src_ann_dir):
            ann_path = path.join(src_ann_dir, ann_file)
            with open(ann_path, "r", encoding="utf-8") as af:
                ann = json.load(af)

            img_width = ann["size"]["width"]
            img_height = ann["size"]["height"]
            file_name = ann_file.replace(".jpg.json", ".txt")

            label_path = path.join(dest_label_dir, file_name)
            with open(label_path, "w", encoding="utf-8") as lf:
                for obj in ann["objects"]:
                    class_id = classes[obj["classTitle"]]
                    points = obj["points"]["exterior"]

                    left = min(p[0] for p in points)
                    right = max(p[0] for p in points)
                    top = min(p[1] for p in points)
                    bottom = max(p[1] for p in points)

                    width = right - left
                    height = bottom - top
                    x_center = (left + width / 2) / img_width
                    y_center = (top + height / 2) / img_height
                    width /= img_width
                    height /= img_height

                    lf.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def main():
    parser = argparse.ArgumentParser(description="Convert dataset to YOLOv8 format")
    parser.add_argument("--src_dir", type=str, default="./datasets/dentalai",
                        help="源数据集目录")
    parser.add_argument("--dest_dir", type=str, default="./preprocessed_datasets/yolov8",
                        help="目标数据集目录")
    parser.add_argument("--classes_path", type=str, default="./datasets/dentalai/meta.json",
                        help="类别定义文件路径")
    args = parser.parse_args()

    convert_dataset(args.src_dir, args.dest_dir, args.classes_path)
    print(f"数据集转换完成! 目标目录: {args.dest_dir}")

if __name__ == "__main__":
    main()
