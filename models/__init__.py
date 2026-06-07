"""模型模块

提供阀门角度检测的多种模型架构，包括模型工厂函数。
"""

import torch
import torch.nn as nn
from typing import Optional

from models.resnet import ResNetRegressor
from models.efficientnet import EfficientNetRegressor
from models.convnext import ConvNeXtRegressor
from models.swin import SwinRegressor

# 模型注册表
MODEL_REGISTRY = {
    "resnet50": ResNetRegressor,
    "resnet101": ResNetRegressor,
    "efficientnet_b4": EfficientNetRegressor,
    "efficientnet_b5": EfficientNetRegressor,
    "convnext_base": ConvNeXtRegressor,
    "convnext_large": ConvNeXtRegressor,
    "swin_base": SwinRegressor,
    "swin_large": SwinRegressor,
}


def build_model(
    model_name: str = "convnext_base",
    pretrained: bool = True,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
) -> nn.Module:
    """模型工厂函数：根据模型名称创建对应的模型实例

    Args:
        model_name: 模型名称（resnet50/resnet101/efficientnet_b4/efficientnet_b5/
                    convnext_base/convnext_large/swin_base/swin_large）
        pretrained: 是否使用预训练权重
        dropout: Dropout 概率
        freeze_backbone: 是否冻结骨干网络

    Returns:
        模型实例

    Raises:
        ValueError: 不支持的模型名称
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"不支持的模型: {model_name}，可选模型: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_name]
    model = model_class(
        model_name=model_name,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )

    return model


def get_model_info(model: nn.Module) -> dict:
    """获取模型信息

    Args:
        model: 模型实例

    Returns:
        包含模型参数量等信息的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_name": model.__class__.__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
    }


def export_onnx(
    model: nn.Module,
    save_path: str,
    input_size: tuple = (1, 3, 384, 384),
    opset_version: int = 17,
    dynamic_axes: Optional[dict] = None,
) -> None:
    """导出模型为 ONNX 格式

    Args:
        model: 模型实例
        save_path: 保存路径
        input_size: 输入尺寸（batch, channels, height, width）
        opset_version: ONNX 算子集版本
        dynamic_axes: 动态轴配置
    """
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size).to(device)

    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
