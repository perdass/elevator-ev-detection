# 电梯内电动车检测系统

基于 YOLOv8 的电梯内电动车/自行车检测系统，用于智慧社区安全管理。

## 🎯 功能特点

- 电动车/自行车实时检测
- 支持图片、视频、摄像头输入
- 提供 PyQt5 可视化界面
- 支持模型导出（ONNX/TensorRT）

## 📊 数据集

使用 [popxoq/elevator-vehicle-dataset](https://github.com/popxoq/elevator-vehicle-dataset) 数据集：

- **规模**: 28,000 张图像
- **类别**: 自行车、电动车
- **格式**: YOLO 格式

### 下载数据集

百度网盘下载：
- 链接: https://pan.baidu.com/s/1VJ-HAOjlYpfnz9UYA5R3vA?pwd=4mnw
- 提取码: `4mnw`

下载后解压到 `datasets/` 目录。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install ultralytics opencv-python pyqt5
```

### 2. 准备数据集

```bash
# 下载数据集后解压到 datasets/
datasets/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 3. 训练模型

```bash
python train.py
```

### 4. 推理检测

```bash
# 图片检测
python detect.py --source image.jpg

# 视频检测
python detect.py --source video.mp4

# 摄像头检测
python detect.py --source 0
```

### 5. 启动可视化界面

```bash
python gui.py
```

## 📁 项目结构

```
elevator-ev-detection/
├── README.md
├── requirements.txt
├── data.yaml          # 数据集配置
├── train.py           # 训练脚本
├── detect.py          # 推理脚本
├── gui.py             # PyQt5 界面
├── export.py          # 模型导出
├── datasets/          # 数据集目录
└── runs/              # 训练结果
```

## 📝 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 100 | 训练轮数 |
| batch | 16 | 批次大小 |
| imgsz | 640 | 输入图像尺寸 |
| model | yolov8n.pt | 预训练模型 |

## 🔧 自定义训练

```bash
python train.py --epochs 200 --batch 32 --model yolov8s.pt
```

## 📄 License

MIT License
