import argparse
import os
import sys
import torch
from ultralytics import YOLO, settings

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.visualization import plot_training_metrics
from utils.file_utils import create_output_dirs, validate_files, ensure_model_extension, reorganize_training_outputs
from utils.metrics import generate_metrics_report, enhanced_metrics_analysis
from utils.per_class_evaluator import evaluate_and_visualize_per_class

# 配置ultralytics将模型下载到models文件夹，数据集使用当前目录
# 注意：runs_dir将在解析参数后动态设置

def find_latest_checkpoint(output_dir):
    """
    在输出目录中查找最新的训练检查点
    
    Args:
        output_dir (str): 输出目录路径
        
    Returns:
        tuple: (checkpoint_path, training_dir) 或 (None, None)
    """
    if not os.path.exists(output_dir):
        return None, None
    
    # 查找所有训练目录（按时间戳命名）
    train_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith('train_'):
            train_dirs.append((item, item_path))
    
    if not train_dirs:
        return None, None
    
    # 按时间戳排序，取最新的
    train_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_dir = train_dirs[0][1]
    
    # 检查是否存在检查点文件
    possible_checkpoints = [
        os.path.join(latest_dir, "weights", "last.pt"),
        os.path.join(latest_dir, "temp_weights", "weights", "last.pt"),
        os.path.join(latest_dir, "temp_weights", "last.pt")
    ]
    
    for checkpoint in possible_checkpoints:
        if os.path.exists(checkpoint):
            return checkpoint, latest_dir
    
    return None, None


def setup_resume_training(args):
    """
    设置断点续训相关的配置
    
    Args:
        args: 命令行参数
        
    Returns:
        tuple: (resume_path, output_dir, is_resuming)
    """
    is_resuming = False
    resume_path = None
    output_dir = None
    
    if args.resume or args.resume_dir:
        # 处理指定续训目录的情况
        if args.resume_dir:
            if not os.path.exists(args.resume_dir):
                raise ValueError(f"指定的续训目录不存在: {args.resume_dir}")
            
            # 查找检查点
            possible_checkpoints = [
                os.path.join(args.resume_dir, "weights", "last.pt"),
                os.path.join(args.resume_dir, "temp_weights", "weights", "last.pt"),
                os.path.join(args.resume_dir, "temp_weights", "last.pt")
            ]
            
            for checkpoint in possible_checkpoints:
                if os.path.exists(checkpoint):
                    resume_path = checkpoint
                    output_dir = args.resume_dir
                    is_resuming = True
                    break
            
            if not resume_path:
                raise ValueError(f"在指定目录中未找到有效的检查点文件: {args.resume_dir}")
        
        # 处理resume参数的情况
        elif args.resume:
            if args.resume == "auto":
                # 自动查找最新检查点
                checkpoint, train_dir = find_latest_checkpoint(args.output_dir)
                if checkpoint:
                    resume_path = checkpoint
                    output_dir = train_dir
                    is_resuming = True
                else:
                    print("⚠️ 未找到可续训的检查点，将开始新的训练")
            elif os.path.isfile(args.resume):
                # 直接指定检查点文件
                resume_path = args.resume
                # 尝试从检查点路径推断输出目录
                if "temp_weights" in resume_path:
                    output_dir = os.path.dirname(os.path.dirname(resume_path))
                else:
                    output_dir = os.path.dirname(os.path.dirname(resume_path))
                is_resuming = True
            else:
                raise ValueError(f"指定的检查点文件不存在: {args.resume}")
    
    return resume_path, output_dir, is_resuming


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
    parser.add_argument('--dataset_name', type=str, default="dentalai",
                        help="数据集名称，用于自动设置数据和输出路径 (默认: dentalai)")
    parser.add_argument('--data_dir', '-d', type=str, default=None,
                        help="训练数据文件夹，包含 data.yaml (如不指定，将使用 ./preprocessed_datasets/{dataset_name})")
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                        help="输出目录 (如不指定，将使用 ./outputs/{dataset_name})")
    
    # 训练选项
    parser.add_argument('--patience', type=int, default=30,
                        help="早停耐心值，多少轮无改善后停止 (默认: 30)")
    parser.add_argument('--save_period', type=int, default=10,
                        help="保存检查点的间隔轮数 (默认: 10)")
    
    # 断点续训选项
    parser.add_argument('--resume', type=str, default=None,
                        help="断点续训: 'auto' 自动从最新检查点恢复，或指定检查点路径")
    parser.add_argument('--resume_dir', type=str, default=None,
                        help="指定续训的输出目录路径")
    
    # 输出控制
    parser.add_argument('--nolog', action='store_true',
                        help="禁用日志输出和可视化图表")
    parser.add_argument('--verbose', action='store_true',
                        help="显示详细训练信息")
    
    args = parser.parse_args()

    # 处理模型文件名
    model_file = ensure_model_extension(args.model)
    
    # 根据数据集名称自动设置路径
    if args.data_dir is None:
        args.data_dir = f"./preprocessed_datasets/{args.dataset_name}"
    if args.output_dir is None:
        args.output_dir = f"./outputs/{args.dataset_name}"
    
    # 动态配置ultralytics运行目录
    settings.update({
        'weights_dir': 'models',
        'datasets_dir': 'datasets',
        'runs_dir': args.output_dir
    })
    
    # 验证批量大小
    if args.batch <= 0 and args.batch != -1:
        raise ValueError("批量大小必须为正整数或-1(自动)")
    
    # 验证文件存在性
    data_yaml = os.path.join(args.data_dir, "data.yaml")
    validate_files(model_file, data_yaml)

    # 设置断点续训
    resume_path, resume_output_dir, is_resuming = setup_resume_training(args)
    
    if is_resuming:
        print(f"🔄 检测到续训模式")
        print(f"   📁 续训目录: {resume_output_dir}")
        print(f"   💾 检查点文件: {resume_path}")
        base_dir = resume_output_dir
        dirs = {
            'base': base_dir,
            'weights': os.path.join(base_dir, "weights"),
            'logs': os.path.join(base_dir, "logs"),
            'logs_records': os.path.join(base_dir, "logs", "records"),
            'analysis': os.path.join(base_dir, "analysis"), 
            'meta': os.path.join(base_dir, "meta")
        }
        # 确保目录存在
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    else:
        # 创建新的输出目录结构
        dirs = create_output_dirs(
            args.model, args.epochs, args.output_dir, enable_logs=not args.nolog
        )

    # 向后兼容，提取主要目录路径
    base_dir = dirs['base']
    weights_dir = dirs['weights']  # 这将是临时的YOLOv8输出目录

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
        
        if is_resuming:
            print(f"📂 从检查点续训: {resume_path}")
            # 续训时直接加载检查点文件
            model = YOLO(resume_path)
        else:
            # 正常训练时加载预训练模型
            model = YOLO(model_file)
        
        print("✅ 模型初始化成功!")
        
        print(f"🚀 开始训练...")
        if is_resuming:
            # 续训模式：从检查点开始新的训练（迁移学习方式）
            print(f"   🔄 使用检查点权重进行迁移训练")
            result = model.train(
                data=data_yaml,
                epochs=args.epochs,  # 明确指定总epoch数
                batch=args.batch,
                imgsz=args.imgsz,
                device=device,
                project=base_dir,
                name="temp_weights",
                exist_ok=True,
                patience=args.patience,
                save_period=args.save_period,
                verbose=args.verbose,
                amp=False
                # 不使用resume参数，而是基于检查点权重进行新的训练
            )
        else:
            # 正常训练模式
            result = model.train(
                data=data_yaml,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=device,  # 使用智能检测的设备
                project=base_dir,  # 使用base_dir作为临时训练目录
                name="temp_weights",  # 临时名称
                exist_ok=True,
                patience=args.patience,
                save_period=args.save_period,
                verbose=args.verbose,
                amp=False  # 禁用AMP以避免自动下载yolo11n.pt
            )
        
        # 训练完成后重新组织输出结构
        temp_weights_dir = os.path.join(base_dir, "temp_weights")
        
        if not args.nolog:
            # 读取类别名称
            import yaml
            try:
                with open(data_yaml, 'r', encoding='utf-8') as f:
                    data_config = yaml.safe_load(f)
                    class_names = data_config.get('names', ['Unknown'])
            except:
                class_names = ['Caries', 'Cavity', 'Crack', 'Tooth']  # 默认类别
            
            print("📊 开始生成完整的评估报告...")
            
            # 生成评估报告到临时logs目录
            temp_logs_dir = os.path.join(base_dir, "logs")
            os.makedirs(temp_logs_dir, exist_ok=True)
            
            # 检查results.csv文件位置
            results_csv = os.path.join(temp_weights_dir, "results.csv")
            
            if os.path.exists(results_csv):
                # 1. 生成完整的训练指标可视化图表
                metrics_plot_path = os.path.join(temp_logs_dir, "training_metrics.png")
                plot_training_metrics(results_csv, metrics_plot_path)
                
                # 2. 计算综合指标分析
                metrics = enhanced_metrics_analysis(results_csv, class_names)
                
                # 3. 生成详细的指标报告
                report_path = os.path.join(temp_logs_dir, "metrics_report.md")
                generate_metrics_report(results_csv, class_names, report_path)
                
                # 4. 进行每类别详细评估
                best_model_path = os.path.join(temp_weights_dir, "weights", "best.pt")
                
                if os.path.exists(best_model_path):
                    print(f"🔍 开始每类别详细指标评估... (使用: {best_model_path})")
                    per_class_metrics = evaluate_and_visualize_per_class(
                        best_model_path, data_yaml, class_names, temp_logs_dir
                    )
                    
                    # 5. 生成完整的评估数据CSV文件
                    if per_class_metrics:
                        evaluation_csv_path = os.path.join(temp_logs_dir, "complete_evaluation_metrics.csv")
                        _save_complete_evaluation_csv(metrics, per_class_metrics, class_names, evaluation_csv_path)
                        print(f"📋 完整评估数据已保存至: {evaluation_csv_path}")
                else:
                    per_class_metrics = None
                    print(f"⚠️ 未找到best.pt模型文件: {best_model_path}")
                
                # 6. 重新组织所有文件到新结构
                reorganize_training_outputs(temp_weights_dir, dirs, class_names)
                
                # 7. 清理临时目录
                if os.path.exists(temp_weights_dir):
                    import shutil
                    shutil.rmtree(temp_weights_dir)
                # 注意：不删除temp_logs_dir，因为它就是我们的目标logs目录
                
                print(f"✅ 训练完成! 模型和完整评估结果保存至: {base_dir}")
                print(f"📊 新的输出结构:")
                print(f"   📦 模型文件: {dirs['weights']}/")
                print(f"   📈 训练过程: {dirs['logs']}/")  
                print(f"   📊 结果分析: {dirs['analysis']}/")
                print(f"   ⚙️  训练配置: {dirs['meta']}/")
                
                # 显示关键指标摘要
                if 'metrics' in locals() and metrics:
                    print(f"🎯 关键指标摘要:")
                    print(f"   - F1-Score: {metrics.get('f1_score', 0):.3f}")
                    print(f"   - Precision: {metrics.get('precision', 0):.3f}")
                    print(f"   - Recall: {metrics.get('recall', 0):.3f}")
                    print(f"   - mAP@0.5: {metrics.get('map50', 0):.3f}")
                    print(f"   - IoU质量: {metrics.get('avg_iou_at_0.5', 0):.3f}")
                    
                # 显示每类别F1-Score摘要
                if 'per_class_metrics' in locals() and per_class_metrics:
                    print(f"🏆 每类别F1-Score:")
                    for class_name, class_metrics in per_class_metrics.items():
                        print(f"   - {class_name}: {class_metrics.get('f1_score', 0):.3f}")
            else:
                print(f"⚠️ 未找到训练结果文件: {results_csv}")
                # 至少重新组织基本文件
                reorganize_training_outputs(temp_weights_dir, dirs, class_names)
                if os.path.exists(temp_weights_dir):
                    import shutil
                    shutil.rmtree(temp_weights_dir)
        else:
            # 即使禁用日志，也要重新组织基本文件结构
            reorganize_training_outputs(temp_weights_dir, dirs, [])
            if os.path.exists(temp_weights_dir):
                import shutil
                shutil.rmtree(temp_weights_dir)
            print(f"✅ 训练完成! 模型保存至: {base_dir}")
        
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
