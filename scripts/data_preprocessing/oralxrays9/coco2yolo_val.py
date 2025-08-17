import json
import os

def get_category_id_map(categories):
    return {cat['id']: idx for idx, cat in enumerate(categories)}

def convert_coco_to_yolo(coco_json_path, output_dir):
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    images = {img['id']: img for img in coco['images']}
    categories = coco['categories']
    cat_id_map = get_category_id_map(categories)
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
            x_center = (bbox[0] + bbox[2] / 2) / width
            y_center = (bbox[1] + bbox[3] / 2) / height
            w = bbox[2] / width
            h = bbox[3] / height
            lines.append(f"{yolo_cat_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    print(f"转换完成: {coco_json_path} -> {output_dir}")

if __name__ == "__main__":
    convert_coco_to_yolo('instances_val2017.json', 'yolo/labels_val2017')
