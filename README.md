# 工业阀门角度检测系统

基于深度学习的工业阀门角度检测系统，支持 0°~80° 角度回归预测，目标 MAE ≤ 1°。

## 项目简介

本项目针对工业场景中阀门开度角度的自动检测需求，采用深度学习回归方法，通过对阀门图像的分析实现精确的角度预测。系统支持多种视角（俯视、侧视、多视角融合），提供从模型训练、评估到部署的完整流程，并附带 RESTful API 和 Web 前端 Demo。

**核心指标**：在测试集上平均绝对误差（MAE）≤ 1°。

## 功能特性

- **多模型支持**：内置 ResNet50/101、EfficientNet-B4/B5、ConvNeXt-Base/Large、Swin-Base/Large 等多种骨干网络，推荐使用 ConvNeXt-Base
- **数据增强**：基于 Albumentations 的丰富增强策略（亮度对比度、色调饱和度、高斯模糊/噪声、随机裁剪、遮挡、透视变换、镜头畸变等），不使用旋转和翻转以避免角度语义改变
- **视角感知训练**：支持 side 视角样本加权（`side_weight`）、过采样、专用增强流水线，针对侧视角预测精度优化
- **难样本挖掘损失**：Focal-style 自适应损失，根据预测误差动态调整样本权重，误差越大的样本权重越高
- **两阶段训练**：先用 all_view 训练基础模型，再用 side_view 数据低学习率微调，提升 side 视角精度
- **渐进式解冻**：支持 warmup 阶段冻结骨干网络，之后解冻微调，加速收敛
- **自适应 batch_size**：根据 GPU 显存和模型类型自动计算最优批大小
- **图像优化**：提供颜色增强、边缘检测、区域提取、光照校正（CLAHE + Gamma）等阀门图像专用优化技术
- **远距离优化**：智能裁剪自动定位并放大阀门区域，多尺度推理融合原图和裁剪图预测，提升远距离拍摄精度
- **视频抽帧推理**：支持视频文件抽帧预测，可选择按 fps 或帧间隔抽帧，输出标注视频和角度变化曲线
- **API 接口**：基于 FastAPI 的 RESTful API，支持单张/批量/视频预测、健康检查、模型信息查询，自带 Swagger/ReDoc 文档
- **前端 Demo**：基于 Streamlit 的交互式 Web 界面，支持单张/批量/视频预测和结果可视化

## 环境搭建

### 前置条件

- Python 3.10
- CUDA 12.1（GPU 推理/训练）
- Conda（推荐）

### 安装步骤

1. **创建 Conda 环境**

```bash
conda create -n valve_detector python=3.10 -y
conda activate valve_detector
```

2. **安装 PyTorch（CUDA 12.1）**

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. **安装项目依赖**

```bash
pip install -r requirements.txt
```

**云服务器注意事项**：

如果部署在无图形界面的云服务器上，需要替换 `opencv-python` 为 `opencv-python-headless`：

```bash
pip uninstall opencv-python -y
pip install opencv-python-headless
```

如果无法访问 HuggingFace，设置国内镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 快速开始

### 训练

```bash
# 使用 ConvNeXt-Base 训练（推荐）
python train.py --model convnext_base --data_dir ./dataset --view all_view

# 使用 ResNet50 训练
python train.py --model resnet50 --epochs 100 --batch_size 16

# 启用视角加权 + 难样本挖掘（提升 side 视角精度）
python train.py --model convnext_base --view all_view --side_weight 5.0 --focal_gamma 2.0

# 两阶段训练：先 all_view 训练，再用 side_view 微调
python train.py --model convnext_base --view all_view --stage2 --stage2_epochs 30 --stage2_lr 1e-5

# 渐进式解冻：前 5 个 epoch 冻结骨干，之后解冻微调
python train.py --model convnext_base --warmup_epochs 5

# 从检查点恢复训练
python train.py --model convnext_base --resume ./weights/valve-epoch=XX-val_mae=XX.ckpt
```

### 评估

```bash
# 在测试集上评估模型
python evaluate.py --model_path ./weights/last.ckpt --data_dir ./dataset

# 指定视角评估
python evaluate.py --model_path ./weights/best.ckpt --view top_view
```

### 预测

```bash
# 单张图片预测
python predict.py --model_path ./weights/last.ckpt --input ./test.jpg

# 批量预测
python predict.py --model_path ./weights/last.ckpt --input ./test_images/ --output ./results/

# 启用智能裁剪（远距离拍摄优化）
python predict.py --model_path ./weights/last.ckpt --input ./test.jpg --smart_crop

# 启用多尺度推理（精度最高）
python predict.py --model_path ./weights/last.ckpt --input ./test.jpg --multi_scale

# 使用 ONNX 模型推理
python predict.py --model_path ./weights/model.onnx --input ./test.jpg --onnx

# 启用图像优化
python predict.py --model_path ./weights/last.ckpt --input ./test.jpg --optimize

# 视频抽帧预测（按每秒 2 帧抽帧）
python predict.py --model_path ./weights/last.ckpt --input ./test.mp4 --fps 2

# 视频抽帧预测（每 30 帧抽 1 帧，输出标注视频）
python predict.py --model_path ./weights/last.ckpt --input ./test.mp4 --frame_interval 30 --save_video
```

### API 服务

```bash
# 启动 API 服务
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 开发模式（自动重载）
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 启动 API 服务（使用 SSL）
$env:SSL_KEYFILE="certs/key.pem"
$env:SSL_CERTFILE="certs/cert.pem"
python -m api.main

# 启动 API 服务（使用 SSL，指定证书文件）
uvicorn api.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
```

启动后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 前端 Demo

```bash
streamlit run frontend/app.py
```

## 项目结构

```
Valve_Detector/
├── api/                        # API 服务
│   ├── __init__.py
│   └── main.py                 # FastAPI 应用（/predict、/predict/batch、/predict/video、/health、/info）
├── config/                     # 配置文件
│   ├── data_config.yaml        # 数据配置（路径、增强、预处理）
│   ├── model_config.yaml       # 模型配置（架构、ONNX导出）
│   └── train_config.yaml       # 训练配置（优化器、调度器、早停）
├── data/                       # 数据处理模块
│   ├── __init__.py
│   ├── augmentation.py         # 数据增强（Albumentations）
│   ├── dataset.py              # 数据集与数据模块
│   └── preprocess.py           # 图像预处理
├── dataset/                    # 数据集目录
│   ├── all_view/               # 多视角图像
│   ├── side_view/              # 侧视图像
│   └── top_view/               # 俯视图像
├── docs/                       # 文档
├── frontend/                   # 前端 Demo
│   └── app.py                  # Streamlit 应用
├── models/                     # 模型模块
│   ├── __init__.py             # 模型注册表与工厂函数、ONNX导出
│   ├── convnext.py             # ConvNeXt 回归模型
│   ├── efficientnet.py         # EfficientNet 回归模型
│   ├── resnet.py               # ResNet 回归模型
│   └── swin.py                 # Swin Transformer 回归模型
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── image_optimization.py   # 图像优化（颜色增强、边缘检测、光照校正）
│   ├── image_utils.py          # 图像读写与绘制工具
│   ├── logger.py               # 日志工具
│   └── metrics.py              # 评估指标计算
├── weights/                    # 模型权重保存目录
├── logs/                       # 日志与 TensorBoard 目录
├── train.py                    # 训练入口脚本
├── evaluate.py                 # 评估入口脚本
├── predict.py                  # 预测入口脚本
├── requirements.txt            # Python 依赖
└── README.md                   # 项目文档
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.3+、PyTorch Lightning |
| 模型库 | timm、torchvision |
| 数据增强 | Albumentations |
| API 框架 | FastAPI、Uvicorn |
| 前端框架 | Streamlit |
| 模型导出 | ONNX、ONNX Runtime |
| 可视化 | Matplotlib、TensorBoard |
| 运行环境 | CUDA 12.1、Python 3.10 |
