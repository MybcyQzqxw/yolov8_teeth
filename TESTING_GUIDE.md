# 模型测试使用说明

项目现已增加两个测试脚本，用于评估和演示训练好的YOLOv8牙齿检测模型：

## 1. test.py - 完整测试脚本

### 功能特点
- ✅ 在测试集上进行完整模型评估
- ✅ 输出详细的性能指标（精确率、召回率、F1-Score、mAP等）
- ✅ 随机抽取10张图片进行预测vs真实标签的可视化对比
- ✅ 自动生成测试报告和结果文件
- ✅ 自动查找最新训练模型或手动指定模型

### 使用方法

```bash
# 基本用法（自动查找最新模型）
python scripts/test.py

# 指定模型文件
python scripts/test.py -m "./outputs/train_xxx/weights/best.pt"

# 指定输出目录和可视化样本数
python scripts/test.py -o "./my_test_results" -s 5

# 完整参数示例
python scripts/test.py -m "best_model.pt" -d "./custom_dataset" -o "./results" -s 8 -c 0.5
```

### 参数说明
- `--model, -m`: 模型文件路径（默认：自动查找最新训练模型）
- `--data_dir, -d`: 数据集目录（默认：./preprocessed_datasets/dentalai）
- `--output_dir, -o`: 输出目录（默认：./test_results）
- `--samples, -s`: 可视化样本数量（默认：10）
- `--conf_threshold, -c`: 预测置信度阈值（默认：0.3）

### 输出内容
测试完成后会生成：
- `test_results.json`: 详细数值结果
- `test_report.md`: 完整测试报告
- `test_predictions_comparison.png`: 10张图片的预测对比可视化

## 2. demo.ipynb - 单图演示Notebook

### 功能特点
- ✅ 交互式Jupyter Notebook环境
- ✅ 单张图片的详细分析
- ✅ 真实标签vs预测结果的对比可视化
- ✅ 可灵活修改图片路径进行测试
- ✅ 详细的检测统计信息

### 使用方法

1. **启动Jupyter Notebook**
   ```bash
   jupyter notebook demo.ipynb
   ```

2. **修改配置**
   - 在notebook中修改 `MODEL_PATH` 指定模型文件
   - 修改 `IMAGE_PATH` 指定要测试的图片

3. **逐步执行**
   - 按顺序执行每个代码块
   - 查看可视化结果和统计信息

### 可视化内容
- 原始图像展示
- 真实标签可视化（绿色框）
- 模型预测结果（红色框）
- 叠加对比图
- 详细检测统计

## 快速开始

### 1. 运行完整测试
```bash
# 使用最新训练的模型在测试集上评估
python scripts/test.py

# 查看结果
ls test_results/
```

### 2. 单图演示
```bash
# 启动notebook
jupyter notebook demo.ipynb

# 或在VSCode中直接打开demo.ipynb
```

## 注意事项

⚠️ **模型要求**: 需要先使用 `scripts/train.py` 训练模型，或手动指定模型文件路径  
⚠️ **数据集**: 测试脚本会自动使用data.yaml中配置的测试集  
⚠️ **环境**: 确保已激活正确的conda环境并安装所有依赖  
⚠️ **Jupyter**: demo.ipynb需要Jupyter环境，可通过 `pip install jupyter` 安装

## 示例输出

### test.py输出示例
```
🔍 正在查找最新训练的模型...
🎯 使用模型: ./outputs/train_yolov8n_2ep_xxx/weights/best.pt
📊 测试集评估结果:
   - 精确率 (Precision): 0.6127
   - 召回率 (Recall): 0.3724
   - F1-Score: 0.4632
   - mAP@0.5: 0.3451
✅ 测试完成! 结果保存至: ./test_results
```

### demo.ipynb功能
- 📸 原图展示
- 🟢 真实标签可视化
- 🔴 预测结果可视化  
- 📊 详细统计信息
- 🎨 交互式调试
