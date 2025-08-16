#!/usr/bin/env python3
"""
YOLOv8牙齿检测演示工具类
提供单图检测、可视化对比、IoU分析等功能
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
from ultralytics import YOLO
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def find_data_yaml(image_path: str) -> str:
    """
    根据图片路径自动查找对应的data.yaml文件
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        data.yaml文件路径
    """
    from pathlib import Path
    
    image_path = Path(image_path)
    
    # 向上查找包含data.yaml的目录
    for parent in image_path.parents:
        data_yaml_path = parent / "data.yaml"
        if data_yaml_path.exists():
            return str(data_yaml_path)
    
    raise FileNotFoundError(f"未找到data.yaml文件，请检查图片路径: {image_path}")


class DentalDetectionDemo:
    """牙齿检测演示类"""
    
    def __init__(self, model_path: str, data_yaml: Optional[str] = None):
        """
        初始化演示类
        
        Args:
            model_path: 模型文件路径
            data_yaml: 数据配置文件路径（可选，仅在需要类别信息或标签对比时使用）
        """
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.model = None
        self.class_names = []
        
        # 设置中文字体和颜色
        rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False
        
        self.GT_COLOR = 'lime'
        self.PRED_COLOR = 'red'
        self.GT_TEXT_COLOR = 'darkgreen' 
        self.PRED_TEXT_COLOR = 'darkred'
        
        self._load_model_and_config()
    
    def _load_model_and_config(self):
        """加载模型和配置"""
        # 检查模型文件存在性
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 加载模型
        print("正在加载模型...")
        self.model = YOLO(self.model_path)
        print("模型加载成功!")
        
        # 如果提供了数据配置文件，则读取类别名称
        if self.data_yaml:
            if not os.path.exists(self.data_yaml):
                raise FileNotFoundError(f"数据配置文件不存在: {self.data_yaml}")
            
            with open(self.data_yaml, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
                self.class_names = data_config.get('names', ['Class_0', 'Class_1', 'Class_2', 'Class_3'])
            
            print(f"类别名称: {self.class_names}")
        else:
            # 如果没有提供配置文件，使用默认类别名称
            self.class_names = ['Class_0', 'Class_1', 'Class_2', 'Class_3']
            print("未提供数据配置文件，使用默认类别名称")
    
    def get_available_images(self, test_dir: str, max_count: int = 20) -> List[Path]:
        """
        获取测试图片列表
        
        Args:
            test_dir: 测试图片目录
            max_count: 最大返回数量
            
        Returns:
            图片路径列表
        """
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"测试图片目录不存在: {test_dir}")
        
        images = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            images.extend(Path(test_dir).glob(f'*{ext}'))
            images.extend(Path(test_dir).glob(f'*{ext.upper()}'))
        
        return sorted(images)[:max_count]
    
    def load_ground_truth_labels(self, label_path: str) -> List[List[float]]:
        """
        加载真实标签
        
        Args:
            label_path: 标签文件路径
            
        Returns:
            标签列表，每个元素为[class_id, x_center, y_center, width, height]
        """
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append([float(x) for x in parts])
        return labels
    
    def denormalize_bbox(self, bbox: List[float], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        将YOLO格式的归一化边界框转换为像素坐标
        
        Args:
            bbox: [x_center, y_center, width, height] (归一化)
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            (x_min, y_min, x_max, y_max) 像素坐标
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
    
    def predict_image(self, image_path: str, conf_threshold: float = 0.1) -> List[Dict]:
        """
        对图像进行预测
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            
        Returns:
            预测结果列表
        """
        results = self.model.predict(image_path, verbose=False)
        predictions = []
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                if confidence >= conf_threshold:
                    predictions.append({
                        'bbox': [x_min, y_min, x_max, y_max],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'Class{class_id}'
                    })
        
        return predictions
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def match_predictions_with_ground_truth(self, gt_labels: List[List[float]], predictions: List[Dict], 
                                          image_shape: tuple, iou_threshold: float = 0.5) -> Dict:
        """
        匹配预测结果与真实标签
        
        Args:
            gt_labels: 真实标签列表 (YOLO格式)
            predictions: 预测结果列表
            image_shape: 图像形状 (height, width)
            iou_threshold: IoU阈值
            
        Returns:
            匹配分析结果
        """
        img_height, img_width = image_shape
        
        # 转换真实标签为像素坐标
        gt_boxes = []
        for label in gt_labels:
            class_id, x_center, y_center, width, height = label
            x_min, y_min, x_max, y_max = self.denormalize_bbox([x_center, y_center, width, height], img_width, img_height)
            gt_boxes.append([x_min, y_min, x_max, y_max])
        
        matched_gt = set()
        matched_pred = set()
        matches = []
        
        # 为每个真实标签找最佳匹配的预测
        for i, gt_box in enumerate(gt_boxes):
            best_iou = 0
            best_pred_idx = -1
            
            for j, pred in enumerate(predictions):
                if j in matched_pred:
                    continue
                    
                pred_box = pred['bbox']
                iou = self.calculate_iou(gt_box, pred_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = j
            
            if best_iou > iou_threshold:
                matched_gt.add(i)
                matched_pred.add(best_pred_idx)
                gt_class = self.class_names[int(gt_labels[i][0])]
                pred_class = predictions[best_pred_idx]['class_name']
                conf = predictions[best_pred_idx]['confidence']
                
                matches.append({
                    'gt_idx': i,
                    'pred_idx': best_pred_idx,
                    'gt_class': gt_class,
                    'pred_class': pred_class,
                    'confidence': conf,
                    'iou': best_iou
                })
        
        return {
            'matches': matches,
            'matched_gt_count': len(matched_gt),
            'matched_pred_count': len(matched_pred),
            'total_gt': len(gt_labels),
            'total_pred': len(predictions),
            'false_negatives': len(gt_labels) - len(matched_gt),
            'false_positives': len(predictions) - len(matched_pred)
        }
    
    def analyze_detection_results(self, gt_labels: List[List[float]], predictions: List[Dict], 
                                img_width: int, img_height: int, iou_threshold: float = 0.5) -> Dict:
        """
        分析检测结果
        
        Args:
            gt_labels: 真实标签
            predictions: 预测结果
            img_width: 图像宽度
            img_height: 图像高度
            iou_threshold: IoU阈值
            
        Returns:
            分析结果字典
        """
        # 转换真实标签为像素坐标
        gt_boxes = []
        for label in gt_labels:
            class_id, x_center, y_center, width, height = label
            x_min, y_min, x_max, y_max = self.denormalize_bbox(
                [x_center, y_center, width, height], img_width, img_height
            )
            gt_boxes.append((x_min, y_min, x_max, y_max))
        
        matched_gt = set()
        matched_pred = set()
        matches = []
        
        for i, gt_box in enumerate(gt_boxes):
            best_iou = 0
            best_pred_idx = -1
            
            for j, pred in enumerate(predictions):
                pred_box = pred['bbox']
                iou = self.calculate_iou(gt_box, pred_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = j
            
            if best_iou > iou_threshold:
                matched_gt.add(i)
                matched_pred.add(best_pred_idx)
                gt_class = self.class_names[int(gt_labels[i][0])]
                pred_class = predictions[best_pred_idx]['class_name']
                conf = predictions[best_pred_idx]['confidence']
                
                matches.append({
                    'gt_idx': i,
                    'pred_idx': best_pred_idx,
                    'gt_class': gt_class,
                    'pred_class': pred_class,
                    'confidence': conf,
                    'iou': best_iou
                })
        
        return {
            'matches': matches,
            'matched_gt_count': len(matched_gt),
            'matched_pred_count': len(matched_pred),
            'total_gt': len(gt_labels),
            'total_pred': len(predictions),
            'false_negatives': len(gt_labels) - len(matched_gt),
            'false_positives': len(predictions) - len(matched_pred)
        }
    
    def visualize_detection(self, image_path: str, save_path: Optional[str] = None, show_ground_truth: bool = True) -> Dict:
        """
        完整的检测可视化流程
        
        Args:
            image_path: 图像路径
            save_path: 保存路径（可选）
            show_ground_truth: 是否显示真实标签（需要data_yaml）
            
        Returns:
            检测分析结果
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        # 获取预测结果
        predictions = self.predict_image(image_path)
        
        result = {
            'image_shape': (img_height, img_width),
            'predictions': predictions,
            'ground_truth': [],
            'analysis': {}
        }
        
        # 如果需要显示真实标签且提供了配置文件
        if show_ground_truth and self.data_yaml:
            # 加载真实标签
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            label_dir = os.path.dirname(image_path).replace('images', 'labels')
            label_path = os.path.join(label_dir, f"{image_name}.txt")
            
            gt_labels = []
            if os.path.exists(label_path):
                gt_labels = self.load_ground_truth_labels(label_path)
                result['ground_truth'] = gt_labels
                
                # 匹配真实标签和预测结果
                analysis = self.match_predictions_with_ground_truth(gt_labels, predictions, image.shape[:2])
                result['analysis'] = analysis
        elif show_ground_truth and not self.data_yaml:
            print("警告: 未提供数据配置文件，无法显示真实标签")
        
        # 创建可视化
        plt.figure(figsize=(15, 10))
        plt.imshow(image)
        plt.axis('off')
        
        # 绘制预测结果
        for pred in predictions:
            x_min, y_min, x_max, y_max = pred['bbox']
            confidence = pred['confidence']
            class_name = pred['class_name']
            
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
            plt.gca().add_patch(rect)
            
            label = f"{class_name}: {confidence:.2f}"
            plt.text(x_min, y_min - 5, label, bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                    fontsize=10, color='white', weight='bold')
        
        # 如果有真实标签，绘制它们
        if show_ground_truth and self.data_yaml and result['ground_truth']:
            for gt in result['ground_truth']:
                class_id, x_center, y_center, width, height = gt
                x_min, y_min, x_max, y_max = self.denormalize_bbox(gt[1:], img_width, img_height)
                class_name = self.class_names[int(class_id)] if int(class_id) < len(self.class_names) else f'Class_{int(class_id)}'
                
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   linewidth=2, edgecolor='green', facecolor='none', alpha=0.8, linestyle='--')
                plt.gca().add_patch(rect)
                
                plt.text(x_max, y_max + 15, f"GT: {class_name}", 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7),
                        fontsize=10, color='white', weight='bold')
        
        plt.title(f"检测结果: {os.path.basename(image_path)}", fontsize=14, weight='bold')
        
        # 添加图例
        if show_ground_truth and self.data_yaml and result['ground_truth']:
            legend_elements = [
                plt.Line2D([0], [0], color='red', lw=2, label='预测结果'),
                plt.Line2D([0], [0], color='green', lw=2, linestyle='--', label='真实标签')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果保存到: {save_path}")
        
        plt.show()
        
        return result
        gt_labels = self.load_ground_truth_labels(label_path)
        
        # 模型预测
        predictions = self.predict_image(image_path)
        
        # 分析结果
        analysis = self.analyze_detection_results(gt_labels, predictions, img_width, img_height)
        
        # 可视化
        self._plot_comparison(image, gt_labels, predictions, img_width, img_height, analysis)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📁 结果已保存至: {save_path}")
        
        plt.show()
        
        return analysis
    
    def _plot_comparison(self, image: np.ndarray, gt_labels: List[List[float]], 
                        predictions: List[Dict], img_width: int, img_height: int, analysis: Dict):
        """绘制对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        TEXT_BG = dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='black')
        
        # 1. 原始图像
        axes[0].imshow(image)
        axes[0].set_title('原始图像', fontsize=16, fontweight='bold', pad=20)
        axes[0].axis('off')
        
        # 2. 真实标签
        axes[1].imshow(image)
        axes[1].set_title(f'真实标签 ({len(gt_labels)} 个)', fontsize=16, fontweight='bold', 
                         pad=20, color='darkgreen')
        axes[1].axis('off')
        
        for i, label in enumerate(gt_labels):
            class_id, x_center, y_center, width, height = label
            x_min, y_min, x_max, y_max = self.denormalize_bbox(
                [x_center, y_center, width, height], img_width, img_height
            )
            
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=3, edgecolor=self.GT_COLOR, facecolor='none', alpha=0.8
            )
            axes[1].add_patch(rect)
            
            class_name = self.class_names[int(class_id)] if int(class_id) < len(self.class_names) else f'Class{int(class_id)}'
            axes[1].text(x_min, y_min - 10, f'GT-{i+1}: {class_name}', 
                       fontsize=11, color=self.GT_TEXT_COLOR, fontweight='bold', bbox=TEXT_BG)
        
        # 3. 预测结果
        axes[2].imshow(image)
        axes[2].set_title(f'预测结果 ({len(predictions)} 个)', fontsize=16, fontweight='bold',
                         pad=20, color='darkred')
        axes[2].axis('off')
        
        for i, pred in enumerate(predictions):
            x_min, y_min, x_max, y_max = pred['bbox']
            
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=3, edgecolor=self.PRED_COLOR, facecolor='none', alpha=0.8
            )
            axes[2].add_patch(rect)
            
            axes[2].text(x_min, y_min - 10, 
                       f"Pred-{i+1}: {pred['class_name']} ({pred['confidence']:.2f})", 
                       fontsize=11, color=self.PRED_TEXT_COLOR, fontweight='bold', bbox=TEXT_BG)
        
        plt.tight_layout()
        
        # 打印分析结果
        self._print_analysis(analysis)
    
    def predict_only(self, image_path: str, save_path: Optional[str] = None, conf_threshold: float = 0.3) -> Dict:
        """
        仅进行预测的方法，不需要真实标签
        
        Args:
            image_path: 图像路径
            save_path: 保存路径（可选）
            conf_threshold: 置信度阈值
            
        Returns:
            预测结果
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        # 获取预测结果
        predictions = self.predict_image(image_path, conf_threshold)
        
        result = {
            'image_path': image_path,
            'image_shape': (img_height, img_width),
            'predictions': predictions,
            'prediction_count': len(predictions)
        }
        
        # 创建可视化
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis('off')
        
        # 绘制预测结果
        for pred in predictions:
            x_min, y_min, x_max, y_max = pred['bbox']
            confidence = pred['confidence']
            class_name = pred['class_name']
            
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
            plt.gca().add_patch(rect)
            
            label = f"{class_name}: {confidence:.2f}"
            plt.text(x_min, y_min - 5, label, bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                    fontsize=10, color='white', weight='bold')
        
        plt.title(f"检测结果: {os.path.basename(image_path)} (共找到 {len(predictions)} 个目标)", 
                 fontsize=14, weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果保存到: {save_path}")
        
        plt.show()
        
        return result
    
    def _print_analysis(self, analysis: Dict):
        """打印分析结果"""
        print("\n" + "="*70)
        print("检测结果分析:")
        print("="*70)
        print(f"真实标签: {analysis['total_gt']} 个")
        print(f"预测结果: {analysis['total_pred']} 个")
        print(f"成功匹配: {analysis['matched_gt_count']} 个")
        print(f"漏检 (FN): {analysis['false_negatives']} 个") 
        print(f"误检 (FP): {analysis['false_positives']} 个")
        
        if analysis['matches']:
            print(f"\n匹配详情:")
            for match in analysis['matches']:
                print(f"   GT-{match['gt_idx']+1} ({match['gt_class']}) <-> "
                      f"Pred-{match['pred_idx']+1} ({match['pred_class']}, "
                      f"{match['confidence']:.2f}) | IoU: {match['iou']:.3f}")
    
    def show_image_selector(self, test_dir: str):
        """显示图片选择器"""
        images = self.get_available_images(test_dir)
        if not images:
            print(f"错误: 在 {test_dir} 中未找到测试图片")
            return []
        
        print(f"可选择的测试图片 (共 {len(images)} 张):")
        for i, img_path in enumerate(images):
            print(f"   {i+1:2d}. {img_path.name}")
        
        return images
