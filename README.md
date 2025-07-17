# YOLOv8 牙齿检测项目

基于 YOLOv8 的牙齿检测和识别项目，包含完整的数据处理、模型训练和结果可视化功能。

## 项目结构

```
yolov8_teeth/
├── README.md               # 项目说明文档
├── requirements.txt        # 项目依赖
├── train.py                # 主训练脚本 (唯一训练入口)
├── models/                 # 预训练模型存储目录
├── dentalai_dataset/       # 原始数据集目录
├── yolo_dataset/           # 处理后的YOLO格式数据集
├── outputs/                # 训练输出目录
├── scripts/                # 数据处理脚本
│   ├── dataset_extract.py  # 数据集解压工具
│   └── dataset_convert.py  # 数据集格式转换工具
└── utils/                  # 工具模块
    ├── __init__.py
    ├── config.py           # 配置管理
    ├── file_utils.py       # 文件操作工具
    └── visualization.py    # 可视化工具
```

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

使用 Dentalai 数据集，下载地址：
https://datasetninja.com/dentalai

将 `tar` 格式的压缩包下载到项目根目录下的 `dentalai_dataset` 文件夹中。

```bash
# 解压数据集
python scripts/dataset_extract.py

# 转换为 YOLO 格式
python scripts/dataset_convert.py
```

### 3. 训练模型

```bash
# 🚀 立即训练！（智能设备检测，自动选择 GPU 或 CPU）
python train.py
```

### 4. 进阶训练
```
# 指定轮数
python train.py --epochs 50
python train.py -e 100

# 指定模型
python train.py --model yolov8n --epochs 50
python train.py -m yolov8s -e 100

# 完整参数示例
python train.py -m yolov8l -e 200 -b 32 --imgsz 1024 --device 0 --patience 50

# 查看帮助信息
python train.py --help
```

## 训练参数详解

### 📋 默认参数总览
使用 `python train.py` 命令时的默认配置：
- 模型: yolov8m (平衡精度和速度)
- 训练轮数: 30
- 批量大小: 16
- 图像尺寸: 640x640
- 训练设备: 自动选择
- 数据目录: ./yolo_dataset
- 输出目录: ./outputs
- 日志记录: 开启

### 模型参数
| 参数      | 简写 | 类型 | 默认值    | 说明     | 示例                                                  |
| --------- | ---- | ---- | --------- | -------- | ----------------------------------------------------- |
| `--model` | `-m` | str  | "yolov8m" | 模型类型 | `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x` |

### 训练控制参数
| 参数       | 简写 | 类型 | 默认值 | 说明         | 示例                         |
| ---------- | ---- | ---- | ------ | ------------ | ---------------------------- |
| `--epochs` | `-e` | int  | 30     | 训练轮数     | `-e 100`                     |
| `--batch`  | `-b` | int  | 16     | 批量大小     | `-b 32`, `-b -1` (自动)      |
| `--imgsz`  | -    | int  | 640    | 输入图像尺寸 | `--imgsz 1024`               |
| `--device` | -    | str  | "auto" | 训练设备     | `--device 0`, `--device cpu` |

### 数据和输出参数
| 参数           | 简写 | 类型 | 默认值           | 说明       | 示例              |
| -------------- | ---- | ---- | ---------------- | ---------- | ----------------- |
| `--data_dir`   | `-d` | str  | "./yolo_dataset" | 数据集目录 | `-d ./my_dataset` |
| `--output_dir` | `-o` | str  | "./outputs"      | 输出目录   | `-o ./results`    |

### 高级训练参数
| 参数            | 类型 | 默认值 | 说明           | 示例              |
| --------------- | ---- | ------ | -------------- | ----------------- |
| `--patience`    | int  | 30     | 早停耐心值     | `--patience 50`   |
| `--save_period` | int  | 10     | 保存检查点间隔 | `--save_period 5` |

### 输出控制参数
| 参数        | 类型 | 默认值 | 说明             |
| ----------- | ---- | ------ | ---------------- |
| `--nolog`   | flag | False  | 禁用日志和可视化 |
| `--verbose` | flag | False  | 显示详细训练信息 |

## 输出结果

训练完成后会在 `outputs/` 目录生成：

```
outputs/
└── train_yolov8n_50ep_2024_07_17_14_30_25/
    ├── weights/
    │   ├── best.pt           # 最佳模型权重
    │   ├── last.pt           # 最后一轮模型权重
    │   └── results.csv       # 训练结果数据
    └── logs/
        └── training_analysis.png  # 训练分析图表
```

### 训练分析图表包含：
- 📈 **损失曲线**: Box Loss, Object Loss, Class Loss
- 🎯 **精度指标**: Precision, Recall 曲线
- 📊 **mAP指标**: mAP@0.5, mAP@0.5:0.95 可视化
- 📉 **学习率**: 学习率调度可视化

## 常见问题

### Q: 如何选择合适的模型？
- **yolov8n**: 速度最快，精度较低，适合实时检测
- **yolov8s**: 速度和精度平衡，推荐入门使用
- **yolov8m**: 中等精度，适合一般应用
- **yolov8l**: 高精度，需要较多显存
- **yolov8x**: 最高精度，需要大量显存和时间

### Q: 批量大小如何设置？
- 根据显存大小调整: 8GB显存建议16-32
- 使用 `-b -1` 让系统自动选择最大可用批量大小
- 批量大小越大，训练越稳定，但需要更多显存

### Q: 训练设备如何选择？
- `--device auto`: 自动选择最佳设备
- `--device 0`: 使用第一块GPU
- `--device cpu`: 使用CPU（速度较慢）

## 主要功能

### 数据处理
- 支持 `tar` 格式数据集自动解压
- Supervisely 格式到 YOLO 格式的转换
- 自动创建 train/val/test 数据分割

### 模型训练
- 支持所有 YOLOv8 模型变体
- 自动创建时间戳输出目录
- 灵活的 batch 大小配置
- 可选的日志和可视化输出

### 结果可视化
- 损失曲线图（Box Loss、Object Loss、Class Loss）
- 精度和召回率曲线
- mAP 指标可视化
- 学习率调度可视化

## 开发说明

项目采用模块化设计：

- `utils/visualization.py`: 可视化功能
- `utils/file_utils.py`: 文件操作工具
- `utils/config.py`: 配置管理
- `scripts/`: 数据处理脚本

## 依赖要求

- **Python >= 3.8** (支持 3.8, 推荐 3.9+)
- Ultralytics >= 8.0.0
- 其他依赖见 `requirements.txt`

## 许可证

本项目遵循 MIT 许可证。
