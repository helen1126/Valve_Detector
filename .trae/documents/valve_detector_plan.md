# 工业阀门角度检测系统 - 实施计划

## 摘要

从零开发一个完整的工业阀门角度检测系统，基于深度学习实现 0°~80° 角度回归预测，目标 MAE ≤ 1°。项目使用 PyTorch + PyTorch Lightning 框架，支持多种模型架构，提供 FastAPI 接口和 Streamlit 前端。

## 当前状态分析

### 已有资源
- **数据集**：`dataset/` 目录下包含三个子目录
  - `all_view/`：约 430+ 张图片，混合视角，角度范围 0°~80°
  - `top_view/`：约 216 张图片，顶部视角
  - `side_view/`：约 100+ 张图片，侧面视角
  - 命名格式：`编号_角度.后缀`（如 `0001_6.7.jpg`、`0941_0.0.png`）
  - 支持格式：`.jpg` 和 `.png`
- **配置文件**：`.gitignore`、`requirements.txt`、`environment.yml` 已就绪
- **无任何代码**：需要从零实现所有模块

### 数据集特点
- 角度值包含小数（如 6.7, 11.5, 17.6 等），也有整数（如 7, 14, 30 等）
- 同一角度有多张图片（不同编号），适合数据增强
- 角度分布不均匀：低角度（0°~10°）样本较多，中间角度密集，高角度（>60°）样本较少
- 视角差异明显：top_view 和 side_view 呈现不同的视觉特征

## 实施步骤

### 第 1 步：项目基础设施

**创建文件：**

1. **`config/data_config.yaml`** - 数据配置
   - 数据集路径、图像尺寸（默认 384×384）、数据划分比例（8:1:1）
   - 数据增强参数开关与配置
   - 视角选择（all_view/top_view/side_view）

2. **`config/model_config.yaml`** - 模型配置
   - 模型架构选择（resnet50/efficientnet_b4/convnext_base/swin_base）
   - 预训练权重路径
   - 输出层配置（单神经元，0~80°回归）
   - ONNX 导出配置

3. **`config/train_config.yaml`** - 训练配置
   - 损失函数：组合损失（MAE + MSE，权重 0.7:0.3）
   - 优化器：AdamW（lr=1e-4, weight_decay=1e-4）
   - 学习率调度：CosineAnnealingLR（T_max=50, eta_min=1e-6）
   - 早停：patience=10，监控 val_mae
   - 批大小：32（可调）、最大 epoch：200
   - 梯度裁剪：max_norm=1.0
   - 混合精度训练（AMP）

4. **`utils/__init__.py`** - 工具模块初始化

5. **`utils/logger.py`** - 日志工具
   - 基于 Loguru 的统一日志管理
   - 控制台 + 文件双输出
   - 日志轮转与保留策略

6. **`utils/metrics.py`** - 评估指标
   - MAE、MSE、RMSE、R² 计算
   - 角度误差分布统计
   - 误差区间统计（<1°, <2°, <5° 的比例）

7. **`utils/image_utils.py`** - 图像工具
   - 图像读取/保存（支持中文路径）
   - 颜色空间转换
   - 图像可视化工具

8. **目录占位文件**：`weights/.gitkeep`、`logs/.gitkeep`

### 第 2 步：数据预处理与增强模块

1. **`data/__init__.py`** - 数据模块初始化

2. **`data/dataset.py`** - 数据集类
   - `ValveDataset`：继承 `torch.utils.data.Dataset`
   - 从文件名解析角度标签：正则 `r'(\d+)_(\d+\.?\d*)\.'`
   - 支持按视角选择数据目录
   - 8:1:1 随机划分训练/验证/测试集（固定随机种子确保可复现）
   - 返回图像张量和归一化角度值（0~1，原始角度/80）
   - `ValveDataModule`：继承 `pytorch_lightning.LightningDataModule`
   - 封装数据加载、预处理、增强、DataLoader 创建

3. **`data/preprocess.py`** - 图像预处理
   - 尺寸统一（可配置，默认 384×384）
   - 亮度/对比度/饱和度调整
   - 高斯去噪
   - 颜色空间转换（RGB→HSV，提取绿色/红色通道）
   - 光照校正（CLAHE 自适应直方图均衡化）
   - 预处理流水线：可配置组合多种预处理操作

4. **`data/augmentation.py`** - 数据增强
   - 基于 Albumentations 实现训练集增强：
     - `RandomRotate90`（不使用，因为旋转会影响角度标签）
     - `HorizontalFlip`（不使用，翻转会影响角度标签）
     - `RandomBrightnessContrast`（亮度对比度变化）
     - `HueSaturationValue`（色调饱和度变化）
     - `GaussianBlur`（高斯模糊）
     - `GaussNoise`（高斯噪声）
     - `RandomResizedCrop`（随机裁剪缩放）
     - `CoarseDropout`（随机遮挡）
     - `ColorJitter`（颜色抖动）
     - `Sharpen`（锐化）
   - 验证集/测试集仅做 Resize + Normalize
   - **重要**：不使用旋转和翻转增强，因为会改变阀门角度的视觉表现

### 第 3 步：模型设计模块

1. **`models/__init__.py`** - 模型模块初始化，提供模型工厂函数

2. **`models/resnet.py`** - ResNet 模型
   - 基于 torchvision 的 ResNet50/101
   - 修改最后全连接层为单神经元输出
   - 加载 ImageNet 预训练权重
   - 可选冻结前几层

3. **`models/efficientnet.py`** - EfficientNet 模型
   - 基于 timm 的 EfficientNet-B4/B5
   - 修改分类头为回归输出
   - 预训练权重加载

4. **`models/convnext.py`** - ConvNeXt 模型
   - 基于 timm 的 ConvNeXt-Base
   - 修改分类头为回归输出
   - 预训练权重加载
   - **推荐模型**：ConvNeXt 在工业视觉任务中表现优异

5. **`models/swin.py`** - Swin Transformer 模型（新增）
   - 基于 timm 的 Swin-Base-Patch4-Window12
   - 修改分类头为回归输出
   - 预训练权重加载

### 第 4 步：训练模块

1. **`train.py`** - 训练脚本
   - `ValveRegressionModel`：继承 `pytorch_lightning.LightningModule`
   - 组合损失函数：`loss = 0.7 * MAE + 0.3 * MSE`
   - 训练步骤：前向传播 → 计算损失 → 反向传播 → 日志记录
   - 验证步骤：计算 MAE/MSE，记录到 TensorBoard
   - 学习率调度配置
   - 模型检查点：保存 val_mae 最低的模型
   - 早停回调
   - 命令行参数：
     - `--model`：选择模型架构
     - `--data_dir`：数据集目录
     - `--view`：视角选择
     - `--epochs`：最大训练轮数
     - `--batch_size`：批大小
     - `--lr`：学习率
     - `--resume`：从检查点恢复训练
   - 训练完成后自动在测试集上评估

### 第 5 步：图像优化模块

1. **`utils/image_optimization.py`** - 图像优化工具
   - 颜色增强：HSV 空间增强绿色/红色通道对比度
   - 边缘检测：Canny/Sobel 边缘提取，叠加到原图
   - 区域提取：基于颜色特征定位角度盘区域并裁剪
   - 光照校正：CLAHE + Gamma 校正
   - 优化流水线：可配置组合多种优化操作
   - 优化前后对比可视化

### 第 6 步：评估模块

1. **`evaluate.py`** - 模型评估脚本
   - 加载训练好的模型权重
   - 在测试集上计算 MAE、RMSE、R²
   - 生成预测结果与真实值对比散点图
   - 生成误差分布直方图
   - 输出误差最大的前 N 个样本
   - 计算不同角度区间的误差统计
   - 保存评估报告到 `logs/` 目录
   - 命令行参数：`--model_path`、`--data_dir`、`--view`

### 第 7 步：预测模块

1. **`predict.py`** - 预测脚本
   - `ValvePredictor` 类：
     - 加载模型权重（支持热加载）
     - 单张图片预测：返回角度值（保留1位小数）
     - 批量图片预测：支持文件夹输入，输出 CSV 结果
     - 预测结果可视化：在图像上标注预测角度
     - ONNX 模型推理支持
   - 命令行参数：`--model_path`、`--input`（图片路径/文件夹）、`--output`

### 第 8 步：API 接口模块

1. **`api/__init__.py`** - API 模块初始化

2. **`api/main.py`** - FastAPI 应用
   - `/predict`：POST，单张图片预测
     - 输入：图片文件（multipart/form-data）
     - 输出：预测角度、处理时间、带标注的图片（base64）
   - `/predict/batch`：POST，批量图片预测
     - 输入：多个图片文件
     - 输出：每个图片的预测结果列表
   - `/health`：GET，健康检查
   - `/info`：GET，模型信息和系统状态
   - CORS 中间件配置
   - 全局异常处理
   - 模型热加载：监听权重文件变化，自动重新加载
   - Swagger UI 自动文档

### 第 9 步：前端 Demo

1. **`frontend/app.py`** - Streamlit 应用
   - 侧边栏：模型选择、参数配置
   - 主界面：
     - 单张图片上传与预测
     - 原始图片与处理后图片对比展示
     - 预测角度和处理时间显示
     - 批量上传与结果下载（CSV）
   - 界面简洁直观，中文界面

### 第 10 步：文档

1. **`README.md`** - 项目主文档（中文）
2. **`docs/训练指南.md`** - 训练指南
3. **`docs/部署指南.md`** - 部署指南
4. **`docs/API文档.md`** - API 接口文档
5. **`docs/常见问题.md`** - 常见问题

## 关键设计决策

### 1. 角度归一化策略
- 训练时将角度归一化到 [0, 1]（angle / 80.0）
- 模型输出使用 Sigmoid 激活确保在 [0, 1] 范围
- 推理时反归一化：predicted_angle = output * 80.0
- 这样有利于模型训练的稳定性和收敛速度

### 2. 数据增强策略
- **不使用旋转和翻转**：阀门角度与视觉方向强相关，旋转/翻转会改变角度的视觉表现
- 重点使用颜色增强和噪声注入：工业现场光照变化大，颜色增强可提高鲁棒性
- 随机裁剪和遮挡：模拟部分遮挡场景

### 3. 模型选择策略
- **推荐 ConvNeXt-Base**：在工业视觉任务中表现优异，精度和速度平衡好
- 提供 ResNet50、EfficientNet-B4、Swin-Base 作为备选
- 所有模型使用 ImageNet 预训练权重
- 输入尺寸 384×384（精度优先，使用较大分辨率）

### 4. 损失函数设计
- 组合损失：`L = 0.7 * L1 + 0.3 * L2`
- L1 (MAE) 占主导：对异常值鲁棒，直接优化目标指标
- L2 (MSE) 辅助：惩罚大误差，加速收敛
- 权重比例可配置

### 5. 视角处理
- 默认使用 `all_view` 训练统一模型
- 支持按视角分别训练，可能获得更高精度
- 数据集类通过配置选择视角目录

### 6. 图像尺寸
- 默认 384×384（精度优先）
- 可配置为 224×224（速度优先）或 512×512（极致精度）

## 文件清单

```
valve_detector/
├── .gitignore                    # [已有]
├── .vscode/settings.json         # [已有]
├── environment.yml               # [已有]
├── requirements.txt              # [已有]
├── README.md                     # [新建] 项目主文档
├── config/
│   ├── data_config.yaml          # [新建] 数据配置
│   ├── model_config.yaml         # [新建] 模型配置
│   └── train_config.yaml         # [新建] 训练配置
├── data/
│   ├── __init__.py               # [新建]
│   ├── dataset.py                # [新建] 数据集类
│   ├── preprocess.py             # [新建] 图像预处理
│   └── augmentation.py           # [新建] 数据增强
├── models/
│   ├── __init__.py               # [新建] 模型工厂
│   ├── resnet.py                 # [新建] ResNet
│   ├── efficientnet.py           # [新建] EfficientNet
│   ├── convnext.py               # [新建] ConvNeXt
│   └── swin.py                   # [新建] Swin Transformer
├── utils/
│   ├── __init__.py               # [新建]
│   ├── logger.py                 # [新建] 日志工具
│   ├── metrics.py                # [新建] 评估指标
│   ├── image_utils.py            # [新建] 图像工具
│   └── image_optimization.py     # [新建] 图像优化
├── train.py                      # [新建] 训练脚本
├── evaluate.py                   # [新建] 评估脚本
├── predict.py                    # [新建] 预测脚本
├── api/
│   ├── __init__.py               # [新建]
│   └── main.py                   # [新建] FastAPI 应用
├── frontend/
│   └── app.py                    # [新建] Streamlit 应用
├── weights/
│   └── .gitkeep                  # [新建] 占位文件
├── logs/
│   └── .gitkeep                  # [新建] 占位文件
├── docs/
│   ├── 训练指南.md               # [新建]
│   ├── 部署指南.md               # [新建]
│   ├── API文档.md                # [新建]
│   └── 常见问题.md               # [新建]
└── dataset/                      # [已有] 数据集目录
```

## 实施顺序

按依赖关系排序，确保每步可独立验证：

1. **基础设施**：配置文件 + 工具模块（logger, metrics, image_utils）
2. **数据模块**：dataset.py → preprocess.py → augmentation.py
3. **模型模块**：models/__init__.py（工厂） → 各模型实现
4. **训练模块**：train.py（依赖数据+模型+工具）
5. **图像优化**：image_optimization.py（依赖 image_utils）
6. **评估模块**：evaluate.py（依赖模型+数据+工具）
7. **预测模块**：predict.py（依赖模型+图像优化）
8. **API 模块**：api/main.py（依赖预测模块）
9. **前端 Demo**：frontend/app.py（依赖 API 或直接调用预测模块）
10. **文档**：README.md + docs/ 下所有文档

## 验证步骤

1. **数据模块验证**：运行数据集加载，检查标签解析正确性，可视化样本
2. **模型验证**：前向传播测试，确认输出形状和范围
3. **训练验证**：小数据集上过拟合测试（应能快速降到极低误差）
4. **评估验证**：测试集上 MAE < 1° 的目标验证
5. **API 验证**：使用 curl/Postman 测试各接口
6. **前端验证**：上传图片测试预测功能
7. **端到端验证**：从训练到部署的完整流程测试

## 假设与约束

- 硬件：NVIDIA GPU，支持 CUDA 12.1
- 数据集：现有约 430+ 张 all_view 图片足够训练（精度优先可考虑更多数据增强）
- 角度范围固定为 0°~80°，不会出现范围外的角度
- 所有阀门为同一型号，无需考虑型号差异
- 文件名格式严格遵循 `编号_角度.后缀` 的命名规范
