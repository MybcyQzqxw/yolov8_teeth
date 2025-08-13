import argparse
import os
import sys
import torch
from ultralytics import YOLO, settings

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.yolov8.visualization import plot_loss_curve
from utils.yolov8.file_utils import create_output_dirs, validate_files, ensure_model_extension
from utils.yolov8.metrics import plot_enhanced_metrics, generate_metrics_report
from utils.yolov8.per_class_evaluator import evaluate_and_visualize_per_class

# 配置ultralytics将模型下载到models文件夹，数据集使用当前目录
settings.update({
    'weights_dir': 'models',
    'datasets_dir': 'dataset',
    'runs_dir': 'outputs/dentalai'  # 设置运行输出目录
})

def detect_device_with_user_prompt():
    """
    智能设备检测函数，自动检测GPU可用性并给出用户友好的提示
    """
    print("🔍 正在检测可用的训练设备...")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
        
        print(f"✅ 检测到 {device_count} 个可用 GPU")
        print(f"🎯 使用 GPU 训练: {gpu_name}")
        print(f"💾 GPU 显存: {gpu_memory}GB")
        return "0"
    else:
        print("⚠️  未检测到可用的 GPU，将使用 CPU 训练")
        print("💡 提示: CPU 训练速度较慢，建议:")
        print("   1. 安装支持 CUDA 的 PyTorch:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("   2. 确保 NVIDIA 显卡驱动已正确安装")
        print("   3. 检查 CUDA 版本兼容性")
        print("   4. 参考官方文档: https://pytorch.org/get-started/locally/")
        
        # 询问用户是否继续 CPU 训练
        while True:
            try:
                user_input = input("🤔 是否继续使用 CPU 训练? (y/n): ").lower().strip()
                if user_input in ['y', 'yes', '是', '']:
                    print("📝 继续使用 CPU 训练...")
                    return "cpu"
                elif user_input in ['n', 'no', '否']:
                    print("🚪 已取消训练，请配置 GPU 环境后重试")
                    return None
                else:
                    print("❓ 请输入 y(是) 或 n(否)")
            except (KeyboardInterrupt, EOFError):
                print("\n🚪 训练已取消")
                return None

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 牙齿检测模型训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python train.py                                      # 零配置训练(推荐新手)
  python train.py -e 100                               # 仅指定训练轮数
  python train.py -m yolov8n -e 50                     # 小模型快速训练
  python train.py -m yolov8s -e 100 -b 32              # 中等规模训练
  python train.py -m yolov8x -b -1 --device 0          # 大模型自动批量大小
        """)
    
    # 必需参数
    parser.add_argument('--model', '-m', type=str, default="yolov8m",
                        help="模型类型: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x (默认: yolov8m)")
    
    # 训练参数
    parser.add_argument('--epochs', '-e', type=int, default=30,
                        help="训练轮数 (默认: 30)")
    parser.add_argument('--batch', '-b', type=int, default=16,
                        help="批量大小: 正整数或-1(自动批量大小) (默认: 16)")
    parser.add_argument('--imgsz', type=int, default=640,
                        help="输入图像尺寸 (默认: 640)")
    parser.add_argument('--device', type=str, default="auto",
                        help="训练设备: auto, cpu, 0, 1, 2, 3... (默认: auto)")
    
    # 数据和输出
    parser.add_argument('--data_dir', '-d', type=str, default="./preprocessed_datasets/dentalai",
                        help="训练数据文件夹，包含 data.yaml (默认: ./preprocessed_datasets/dentalai)")
    parser.add_argument('--output_dir', '-o', type=str, default="./outputs/dentalai",
                        help="输出目录 (默认: ./outputs/dentalai)")
    
    # 训练选项
    parser.add_argument('--patience', type=int, default=30,
                        help="早停耐心值，多少轮无改善后停止 (默认: 30)")
    parser.add_argument('--save_period', type=int, default=10,
                        help="保存检查点的间隔轮数 (默认: 10)")
    
    # 输出控制
    parser.add_argument('--nolog', action='store_true',
                        help="禁用日志输出和可视化图表")
    parser.add_argument('--verbose', action='store_true',
                        help="显示详细训练信息")
    
    args = parser.parse_args()

    # 处理模型文件名
    model_file = ensure_model_extension(args.model)
    
    # 验证批量大小
    if args.batch <= 0 and args.batch != -1:
        raise ValueError("批量大小必须为正整数或-1(自动)")
    
    # 验证文件存在性
    data_yaml = os.path.join(args.data_dir, "data.yaml")
    validate_files(model_file, data_yaml)

    # 创建输出目录
    base_dir, weights_dir, logs_dir = create_output_dirs(
        args.model, args.epochs, args.output_dir, enable_logs=not args.nolog
    )

    # 智能设备检测和用户提示
    if args.device == "auto":
        device = detect_device_with_user_prompt()
        if device is None:  # 用户选择取消训练
            return
    else:
        device = args.device
        print(f"🎯 使用指定设备: {device}")

    print(f"🚀 开始训练 YOLOv8 牙齿检测模型")
    print(f"   📦 模型: {model_file}")
    print(f"   📊 训练轮数: {args.epochs}")
    print(f"   📏 批量大小: {args.batch}")
    print(f"   🖼️  图像尺寸: {args.imgsz}")
    print(f"   💻 训练设备: {device}")
    print(f"   📁 数据目录: {args.data_dir}")
    print(f"   💾 输出目录: {base_dir}")
    print(f"   📈 日志记录: {'关闭' if args.nolog else '开启'}")

    # 开始训练
    try:
        print("🔄 正在初始化模型...")
        
        # 设置环境变量确保模型下载到正确位置
        os.environ['YOLO_CONFIG_DIR'] = os.path.join(os.getcwd(), 'models')
        
        model = YOLO(model_file)
        print("✅ 模型初始化成功!")
        
        print("🚀 开始训练...")
        result = model.train(
            data=data_yaml,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,  # 使用智能检测的设备
            project=base_dir,
            name="weights",
            exist_ok=True,
            patience=args.patience,
            save_period=args.save_period,
            verbose=args.verbose,
            amp=False  # 禁用AMP以避免自动下载yolo11n.pt
        )
        
    except ConnectionError as e:
        print(f"❌ 网络连接错误: {e}")
        print("💡 解决方案:")
        print("   1. 检查网络连接")
        print("   2. 手动下载模型文件到项目目录:")
        print(f"      https://github.com/ultralytics/assets/releases/download/v8.2.0/{model_file}")
        print("   3. 或使用更小的模型: python train.py -m yolov8n")
        return
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        return

    # 默认生成训练可视化图表（除非显式指定 --nolog）
    if not args.nolog:
        # YOLOv8可能在嵌套的weights目录中保存results.csv
        results_csv = os.path.join(base_dir, "weights", "results.csv")
        alt_results_csv = os.path.join(base_dir, "weights", "weights", "results.csv")
        
        # 检查results.csv文件位置
        if os.path.exists(results_csv):
            csv_path_to_use = results_csv
        elif os.path.exists(alt_results_csv):
            csv_path_to_use = alt_results_csv
        else:
            csv_path_to_use = None
            
        if csv_path_to_use:
            # 读取类别名称
            import yaml
            try:
                with open(data_yaml, 'r', encoding='utf-8') as f:
                    data_config = yaml.safe_load(f)
                    class_names = data_config.get('names', ['Unknown'])
            except:
                class_names = ['Caries', 'Cavity', 'Crack', 'Tooth']  # 默认类别
            
            print("📊 开始生成完整的评估报告...")
            
            # 1. 生成传统的训练分析图表
            traditional_plot_path = os.path.join(logs_dir, "training_analysis.png")
            plot_loss_curve(csv_path_to_use, traditional_plot_path)
            
            # 2. 生成增强的指标可视化图表
            enhanced_plot_path = os.path.join(logs_dir, "enhanced_metrics_analysis.png")
            metrics = plot_enhanced_metrics(csv_path_to_use, enhanced_plot_path, class_names)
            
            # 3. 生成详细的指标报告
            report_path = os.path.join(logs_dir, "metrics_report.md")
            generate_metrics_report(csv_path_to_use, class_names, report_path)
            
            # 4. 进行每类别详细评估并保存到CSV
            # YOLOv8 创建嵌套的weights目录结构: project/name/weights/best.pt
            best_model_path = os.path.join(base_dir, "weights", "weights", "best.pt")
            # 备用路径，以防结构不同
            alt_best_model_path = os.path.join(base_dir, "weights", "best.pt")
            
            # 检查模型文件是否存在
            if os.path.exists(best_model_path):
                model_path_to_use = best_model_path
                print(f"🔍 开始每类别详细指标评估... (使用: {model_path_to_use})")
            elif os.path.exists(alt_best_model_path):
                model_path_to_use = alt_best_model_path
                print(f"🔍 开始每类别详细指标评估... (使用: {model_path_to_use})")
            else:
                model_path_to_use = None
                print("⚠️ 未找到best.pt模型文件，跳过每类别评估")
                print(f"   🔍 查找路径1: {best_model_path}")
                print(f"   🔍 查找路径2: {alt_best_model_path}")
            
            per_class_metrics = None
            if model_path_to_use:
                per_class_metrics = evaluate_and_visualize_per_class(
                    model_path_to_use, data_yaml, class_names, logs_dir
                )
                
                # 5. 生成完整的评估数据CSV文件
                if per_class_metrics:
                    evaluation_csv_path = os.path.join(logs_dir, "complete_evaluation_metrics.csv")
                    _save_complete_evaluation_csv(metrics, per_class_metrics, class_names, evaluation_csv_path)
                    print(f"📋 完整评估数据已保存至: {evaluation_csv_path}")
            else:
                print("⚠️ 未找到best.pt模型文件，跳过每类别评估")
            
            print(f"✅ 训练完成! 模型和完整评估结果保存至: {base_dir}")
            print(f"📊 评估结果文件:")
            print(f"   📈 训练曲线: {traditional_plot_path}")
            print(f"   📊 增强分析: {enhanced_plot_path}")
            print(f"   📋 整体报告: {report_path}")
            
            if per_class_metrics:
                print(f"   🏷️  每类别图表: {os.path.join(logs_dir, 'per_class_metrics.png')}")
                print(f"   📊 每类别报告: {os.path.join(logs_dir, 'per_class_report.md')}")
                print(f"   📋 完整评估CSV: {os.path.join(logs_dir, 'complete_evaluation_metrics.csv')}")
            
            # 显示关键指标摘要
            if metrics:
                print(f"🎯 关键指标摘要:")
                print(f"   - F1-Score: {metrics.get('f1_score', 0):.3f}")
                print(f"   - Precision: {metrics.get('precision', 0):.3f}")
                print(f"   - Recall: {metrics.get('recall', 0):.3f}")
                print(f"   - mAP@0.5: {metrics.get('map50', 0):.3f}")
                print(f"   - IoU质量: {metrics.get('avg_iou_at_0.5', 0):.3f}")
                
            # 显示每类别F1-Score摘要
            if per_class_metrics:
                print(f"🏆 每类别F1-Score:")
                for class_name, class_metrics in per_class_metrics.items():
                    print(f"   - {class_name}: {class_metrics.get('f1_score', 0):.3f}")
        else:
            print("⚠️ 未找到 results.csv，无法生成训练分析图表")
            print(f"   🔍 查找路径1: {results_csv}")
            print(f"   🔍 查找路径2: {alt_results_csv}")
    else:
        print(f"✅ 训练完成! 模型保存至: {base_dir}")


def _save_complete_evaluation_csv(overall_metrics, per_class_metrics, class_names, save_path):
    """
    保存完整的评估指标到CSV文件
    
    Args:
        overall_metrics (dict): 整体指标
        per_class_metrics (dict): 每类别指标
        class_names (list): 类别名称
        save_path (str): 保存路径
    """
    try:
        import pandas as pd
        
        # 准备数据
        data = []
        
        # 添加整体指标行
        overall_row = {
            'Type': 'Overall',
            'Class': 'All',
            'Precision': overall_metrics.get('precision', 0),
            'Recall': overall_metrics.get('recall', 0),
            'F1_Score': overall_metrics.get('f1_score', 0),
            'mAP_50': overall_metrics.get('map50', 0),
            'mAP_50_95': overall_metrics.get('map50_95', 0),
            'IoU_Quality_50': overall_metrics.get('avg_iou_at_0.5', 0),
            'IoU_Quality_50_95': overall_metrics.get('avg_iou_0.5_to_0.95', 0),
            'Epoch': int(overall_metrics.get('epoch', 0))
        }
        data.append(overall_row)
        
        # 添加每类别指标行
        if per_class_metrics:
            for class_name, class_metrics in per_class_metrics.items():
                class_row = {
                    'Type': 'Per_Class',
                    'Class': class_name,
                    'Precision': class_metrics.get('precision', 0),
                    'Recall': class_metrics.get('recall', 0),
                    'F1_Score': class_metrics.get('f1_score', 0),
                    'mAP_50': class_metrics.get('ap50', 0),
                    'mAP_50_95': class_metrics.get('ap50_95', 0),
                    'IoU_Quality_50': class_metrics.get('ap50', 0),  # AP50作为IoU@0.5质量指标
                    'IoU_Quality_50_95': class_metrics.get('ap50_95', 0),  # AP50-95作为综合IoU质量
                    'Epoch': int(overall_metrics.get('epoch', 0))
                }
                data.append(class_row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False, float_format='%.4f')
        
        print(f"[✓] 完整评估指标CSV已保存: {save_path}")
        
    except Exception as e:
        print(f"[!] 保存完整评估CSV失败: {e}")

if __name__ == '__main__':
    main()
