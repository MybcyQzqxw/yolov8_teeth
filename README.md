# YOLOv8 牙齿检测项目

基于 YOLOv8 的牙齿检测和识别项目，包含完整的数据处理、模型训练和结果可视化功能。

## 项目结构

``` text
yolov8_teeth/
├── README.md                    # 项目说明文档
├── LICENSE                      # 开源许可证
├── requirements.txt             # Python依赖包列表
├── demo.ipynb                   # 🎯 交互式检测演示Notebook
├── datasets/                    # 原始数据集目录
│   └── dentalai/               # Dentalai牙齿检测数据集
├── preprocessed_datasets/       # 🔄 预处理后的YOLO格式数据集
│   └── dentalai/               # 转换后的dentalai数据集
├── models/                      # 📦 预训练模型存储目录
├── outputs/                     # 🏆 训练输出结果目录
│   └── dentalai/               # dentalai数据集的训练结果
├── test_results/                # 📊 模型测试结果目录
├── scripts/                     # 🚀 核心脚本目录
│   ├── __init__.py             
│   ├── train.py                # 🎯 主训练脚本 (核心入口)
│   ├── test.py                 # 📈 模型测试评估脚本
│   └── data_preprocessing/     # 数据预处理脚本集合
│       ├── __init__.py
│       └── dentalai/          # Dentalai数据集处理脚本
│           ├── dataset_extract.py   # 数据集解压工具
│           └── dataset_convert.py   # 格式转换工具 (Supervisely → YOLO)
└── utils/                       # 🛠️ 工具模块库
    ├── __init__.py
    ├── demo_utils.py           # 🎨 Demo演示工具类
    ├── file_utils.py           # 📁 文件操作工具
    ├── visualization.py        # 📊 数据可视化工具
    ├── metrics.py              # 📋 指标计算和分析
    └── per_class_evaluator.py  # 🔍 每类别性能评估器
```

### 📋 目录功能说明 *（必读）*

#### 🎯 核心功能文件

- **`scripts/train.py`**: 主训练脚本，支持断点续训和完整的模型训练流程
- **`scripts/test.py`**: 模型测试脚本，提供全面的性能评估和可视化
- **`demo.ipynb`**: 交互式Jupyter演示，支持单张图片检测和对比分析

#### 📂 数据目录结构

- **`datasets/`**: 存放原始下载的数据集文件（如tar压缩包）
- **`preprocessed_datasets/`**: 存放转换为YOLO格式的训练数据，包含images/labels目录和data.yaml配置
- **`outputs/`**: 训练结果输出，按时间戳文件夹组织，包含模型权重、日志、分析图表等
- **`test_results/`**: 测试评估结果，包含性能指标、对比图表和样本可视化

#### 🎨 数据处理脚本

- **`scripts/data_preprocessing/dentalai/`**: 专门处理Dentalai数据集的工具
  - `dataset_extract.py`: 自动解压tar格式的数据集文件
  - `dataset_convert.py`: 将Supervisely JSON格式转换为YOLO txt格式

## 快速开始

### 1. 环境准备

```bash
# 克隆项目后进入项目根目录
cd path/to/your/project/

# 创建 conda 环境
conda create --name yolov8_teeth python=3.9 -y

# 激活 conda 环境
conda activate yolov8_teeth

# 配置显卡驱动、CUDA 之后，安装合适的 PyTorch 版本
# 例如：CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# 例如：CPU Only
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 cpuonly -c pytorch

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 数据集准备

（1）dentalai 数据集（照片）

下载地址：<https://datasetninja.com/dentalai>

将 `tar` 格式的压缩包下载到 `datasets/dentalai` 目录下。

```bash
# 解压数据集
python scripts/data_preprocessing/dentalai/dataset_extract.py

# 转换为 YOLO 格式
python scripts/data_preprocessing/dentalai/dataset_convert.py
```

（2）dentalx 数据集（X光片）

下载地址：<https://zenodo.org/records/7812323#.ZDQE1uxBwUG>

将压缩包：

- `training_data.zip`
- `validation_data.zip`

下载到 `datasets/dentalx` 目录下。

```bash
# 解压数据集
python scripts/data_preprocessing/dentalx/dataset_extract.py

# 转换为 YOLO 格式
python scripts/data_preprocessing/dentalx/dataset_convert.py
```

**注意**: dentalx转换脚本专门处理疾病检测任务，只使用 `quadrant-enumeration-disease` 变体，包含4个疾病类别：Impacted、Caries、Periapical Lesion、Deep Caries。数据集共705张图像，按7:2:1比例划分为493张训练集、141张验证集、71张测试集。

（3）oralxrays9 数据集（X光片）

下载地址：<https://drive.google.com/drive/folders/1_y7ERcFicnOYY2DMR6Qe1W4KGdCsoQ1n?usp=drive_link>

将压缩包：

- `annotations.zip`
- `train2017.zip`
- `val2017.zip`

下载到 `datasets/oralxrays9` 目录下。

```bash
# 解压数据集
python scripts/data_preprocessing/oralxrays9/dataset_extract.py

# 转换为 YOLO 格式
python scripts/data_preprocessing/oralxrays9/dataset_convert.py
```

### 3. 训练模型

通过 `--dataset_name` 命令行参数指定数据集。

``` bash
# 使用数据集名称【推荐方式】
python scripts/train.py --dataset_name dentalai --epochs 30
python scripts/train.py --dataset_name dentalx --model yolov8m
python scripts/train.py --dataset_name oralxrays9 --epochs 50 --batch 16

# 传统指定路径方式（仍然支持）
python scripts/train.py --data_dir ./preprocessed_datasets/dentalai --output_dir ./outputs/dentalai

# 完整参数示例

# （1）dentalai 数据集（照片）
python scripts/train.py --dataset_name dentalai --model yolov8m --epochs 1 --batch -1

# （2）dentalx 数据集（X光片）
python scripts/train.py --dataset_name dentalx --model yolov8m --epochs 1 --batch -1

# （3）oralxrays9 数据集（X光片）
python scripts/train.py --dataset_name oralxrays9 --model yolov8m --epochs 1 --batch -1

# 查看帮助信息
python scripts/train.py --help
```

## 训练参数详解

### 默认参数

使用 `python scripts/train.py` 命令时的默认配置：

- 模型: yolov8m
- 训练轮数: 30
- 批量大小: 16
- 图像尺寸: 640x640
- 训练设备: 自动选择
- 数据目录: ./preprocessed_datasets/dentalai
- 输出目录: ./outputs/dentalai
- 日志记录: 开启

### 自定义参数

#### 模型参数

| 参数      | 简写 | 类型 | 默认值    | 说明     | 示例                                                  |
| --------- | ---- | ---- | --------- | -------- | ----------------------------------------------------- |
| `--model` | `-m` | str  | "yolov8m" | 模型类型 | `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x` |

#### 训练控制参数

| 参数       | 简写 | 类型 | 默认值 | 说明         | 示例                         |
| ---------- | ---- | ---- | ------ | ------------ | ---------------------------- |
| `--epochs` | `-e` | int  | 30     | 训练轮数     | `-e 100`                     |
| `--batch`  | `-b` | int  | 16     | 批量大小     | `-b 32`, `-b -1` (自动)      |
| `--imgsz`  | -    | int  | 640    | 输入图像尺寸 | `--imgsz 1024`               |
| `--device` | -    | str  | "auto" | 训练设备     | `--device 0`, `--device cpu` |

#### 数据和输出参数

| 参数           | 简写 | 类型 | 默认值                         | 说明       | 示例              |
| -------------- | ---- | ---- | ------------------------------ | ---------- | ----------------- |
| `--data_dir`   | `-d` | str  | "./preprocessed_datasets/dentalai" | 数据集目录 | `-d ./my_dataset` |
| `--output_dir` | `-o` | str  | "./outputs/dentalai"               | 输出目录   | `-o ./results`    |

#### 高级训练参数

| 参数            | 类型 | 默认值 | 说明           | 示例              |
| --------------- | ---- | ------ | -------------- | ----------------- |
| `--patience`    | int  | 30     | 早停耐心值     | `--patience 50`   |
| `--save_period` | int  | 10     | 保存检查点间隔 | `--save_period 5` |

#### 断点续训参数

| 参数           | 类型 | 默认值 | 说明                           | 示例                              |
| -------------- | ---- | ------ | ------------------------------ | --------------------------------- |
| `--resume`     | str  | None   | 断点续训检查点                 | `--resume auto`, `--resume path/to/last.pt` |
| `--resume_dir` | str  | None   | 指定续训的输出目录路径         | `--resume_dir ./outputs/train_xxx` |

#### 输出控制参数

| 参数        | 类型 | 默认值 | 说明             |
| ----------- | ---- | ------ | ---------------- |
| `--nolog`   | flag | False  | 禁用日志和可视化 |
| `--verbose` | flag | False  | 显示详细训练信息 |

## 输出结果

训练完成后会在 `outputs/dentalai/` （可通过训练脚本命令行参数 `--output_dir` 指定）目录生成以时间戳命名的训练文件夹：

```text
outputs/dentalai/
└── train_yolov8n_2ep_2025_08_14_09_14_07/  # 🕒 时间戳文件夹
    ├── weights/                            # 📦 模型权重文件
    │   ├── best.pt                        # 验证集上最佳模型权重
    │   ├── last.pt                        # 最后一轮训练权重
    │   └── epoch*.pt                      # 各轮次保存的权重文件
    ├── logs/                              # 📊 训练日志和图表
    │   ├── training_metrics.csv           # 原始训练指标数据
    │   ├── training_metrics.png           # 训练指标可视化图表
    │   └── records/                       # 训练过程记录图片
    │       ├── train_batch*.jpg           # 训练批次样本图
    │       ├── val_batch*_labels.jpg      # 验证集标签图
    │       └── val_batch*_pred.jpg        # 验证集预测图
    ├── analysis/                          # 📈 详细分析图表
    │   ├── BoxF1_curve.png               # F1分数曲线图
    │   ├── BoxPR_curve.png               # 精确率-召回率曲线
    │   ├── BoxP_curve.png                # 精确率曲线图
    │   ├── BoxR_curve.png                # 召回率曲线图
    │   ├── confusion_matrix.png          # 混淆矩阵图
    │   ├── confusion_matrix_normalized.png # 标准化混淆矩阵
    │   ├── per_class_metrics.png         # 每类别指标对比图
    │   └── README.md                     # 分析报告说明
    └── meta/                             # 🔧 元数据和配置
        ├── args.yaml                     # 训练参数配置
        ├── labels.jpg                    # 数据集标签分布图
        └── labels_correlogram.jpg        # 标签相关性图
```

### 🚨 重要说明：时间戳文件夹命名规则

**文件夹命名格式**: `train_{model}_{epochs}ep_{YYYY_MM_DD_HH_MM_SS}`

- `{model}`: 使用的模型类型 (yolov8n/yolov8s/yolov8m/yolov8l/yolov8x)
- `{epochs}`: 训练轮数
- `{YYYY_MM_DD_HH_MM_SS}`: 训练开始的时间戳

**示例文件夹名称**:

- `train_yolov8m_30ep_2025_08_14_14_30_25` - 使用yolov8m模型，30轮训练，2025年8月14日14:30:25开始
- `train_yolov8n_100ep_2025_08_14_09_15_32` - 使用yolov8n模型，100轮训练，2025年8月14日09:15:32开始

### 📁 输出文件详细说明

#### 1. weights/ - 模型权重目录

- **`best.pt`**: 🏆 在验证集上表现最佳的模型权重（推荐用于推理）
- **`last.pt`**: 🔄 最后一轮训练的模型权重（用于断点续训）  
- **`epoch*.pt`**: 💾 根据`--save_period`参数定期保存的权重文件

#### 2. logs/ - 训练日志目录

- **`training_metrics.csv`**: 📋 包含所有训练轮次的详细指标数据
- **`training_metrics.png`**: 📊 训练过程可视化图表（损失曲线、精度指标等）
- **`records/`**: 📸 训练过程的样本图片记录

#### 3. analysis/ - 分析图表目录

- **各类曲线图**: 详细的性能分析图表
- **混淆矩阵**: 模型分类效果的直观展示
- **每类别指标**: 各检测类别的独立性能分析

#### 4. meta/ - 元数据目录

- **`args.yaml`**: 🔧 完整的训练参数配置（用于复现实验）
- **标签分析图**: 数据集的标签分布和相关性分析

### 🔥 断点续训支持

时间戳文件夹支持完整的断点续训功能：

```bash
# 自动从最新训练继续
python scripts/train.py --resume auto

# 指定具体的训练文件夹继续
python scripts/train.py --resume_dir ./outputs/dentalai/train_yolov8m_30ep_2025_08_14_14_30_25
```

## 模型测试

训练完成后，使用 `scripts/test.py` 对模型进行全面评估和可视化分析。

```bash
# 完整参数示例

# （1）dentalai 数据集
python scripts/test.py --model ./outputs/dentalai/train_yolov8m_1ep_2025_08_14_12_35_22/weights/best.pt -d preprocessed_datasets/dentalai/data.yaml

# （2）dentalx 数据集
python scripts/test.py --model ./outputs/dentalx/train_yolov8m_1ep_2025_08_16_12_19_12/weights/best.pt -d preprocessed_datasets/dentalx/data.yaml

# （3）OralXrays-9 数据集
python scripts/test.py --model ./outputs/OralXrays-9/train_yolov8m_1ep_2025_08_16_12_19_12/weights/best.pt -d preprocessed_datasets/OralXrays-9/data.yaml
```

### 📋 测试参数

| 参数              | 简写 | 类型  | 默认值                           | 说明                 |
| ----------------- | ---- | ----- | -------------------------------- | -------------------- |
| `--model`         | `-m` | str   | None (自动查找)                  | 模型文件路径         |
| `--data`          | `-d` | str   | "./preprocessed_datasets/dentalai/data.yaml" | 数据配置文件         |
| `--output_dir`    | `-o` | str   | "./test_results"                 | 测试结果输出目录     |
| `--samples`       | `-s` | int   | 10                               | 可视化对比样本数量   |
| `--conf_threshold`| `-c` | float | 0.3                              | 预测置信度阈值       |

### 📊 测试输出结果

测试完成后，会在指定输出目录生成以下文件：

```text
test_results/
└── test_yolov8m_2025_08_14_15_30_45/     # 时间戳文件夹
    ├── logs/                             # 📊 测试日志和指标
    │   ├── training_metrics.csv          # 原始训练指标复制
    │   ├── training_metrics.png          # 训练过程图表
    │   ├── enhanced_metrics_analysis.png # 增强指标分析图
    │   ├── metrics_report.md             # 详细指标报告  
    │   ├── per_class_metrics.png         # 每类别指标图
    │   └── per_class_report.md           # 每类别详细报告
    ├── analysis/                         # 📈 测试分析图表
    │   ├── BoxF1_curve.png              # F1分数曲线
    │   ├── BoxPR_curve.png              # 精确率-召回率曲线  
    │   ├── confusion_matrix.png         # 混淆矩阵
    │   └── per_class_metrics.png        # 每类别性能对比
    └── samples/                          # 🖼️ 样本对比可视化
        ├── sample_001_comparison.jpg     # 真实vs预测对比图
        ├── sample_002_comparison.jpg
        └── ...
```

## Demo

`demo.ipynb` 提供两种检测模式：

1. **📊 测试集图片检测** - 带真实标签对比分析
2. **🔍 任意图片检测** - 仅显示模型预测结果

## 常见问题

### Q: 批量大小如何设置？

- 根据显存大小调整: 8GB显存建议16-32
- 使用 `-b -1` 让系统自动选择最大可用批量大小
- 批量大小越大，训练越稳定，但需要更多显存

### Q: 训练设备如何选择？

- `--device auto`: 自动选择最佳设备
- `--device 0`: 使用第一块GPU
- `--device cpu`: 使用CPU（速度较慢）

## 依赖要求

- **Python >= 3.8** (支持 3.8, 推荐 3.9+)
- Ultralytics >= 8.0.0
- 其他依赖见 `requirements.txt`

## 许可证

本项目遵循 MIT 许可证。
