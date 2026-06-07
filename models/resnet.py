"""ResNet 回归模型

基于 torchvision 的 ResNet 架构，修改输出层为单神经元回归输出。
"""

import torch
import torch.nn as nn
import torchvision.models as torchvision_models


class ResNetRegressor(nn.Module):
    """ResNet 角度回归模型

    修改 ResNet 的全连接层为单神经元输出，使用 Sigmoid 激活
    确保输出在 [0, 1] 范围内（归一化角度）。

    Args:
        model_name: 模型名称（resnet50 或 resnet101）
        pretrained: 是否使用 ImageNet 预训练权重
        dropout: Dropout 概率
        freeze_backbone: 是否冻结骨干网络
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.model_name = model_name

        # 加载预训练骨干网络
        weights = "IMAGENET1K_V1" if pretrained else None
        if model_name == "resnet50":
            backbone = torchvision_models.resnet50(weights=weights)
        elif model_name == "resnet101":
            backbone = torchvision_models.resnet101(weights=weights)
        else:
            raise ValueError(f"不支持的 ResNet 变体: {model_name}")

        # 冻结骨干网络
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # 提取特征维度
        in_features = backbone.fc.in_features

        # 移除原始全连接层
        backbone.fc = nn.Identity()

        self.backbone = backbone

        # 回归头
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
            nn.Sigmoid(),  # 输出归一化到 [0, 1]
        )

        # 初始化回归头权重
        self._init_weights()

    def _init_weights(self):
        """初始化回归头权重"""
        for module in self.regressor.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入图像张量 (B, 3, H, W)

        Returns:
            归一化角度预测值 (B, 1)，范围 [0, 1]
        """
        features = self.backbone(x)
        output = self.regressor(features)
        return output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征向量

        Args:
            x: 输入图像张量

        Returns:
            特征向量
        """
        return self.backbone(x)
