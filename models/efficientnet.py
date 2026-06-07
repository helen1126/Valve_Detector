"""EfficientNet 回归模型

基于 timm 的 EfficientNet 架构，修改分类头为单神经元回归输出。
"""

import torch
import torch.nn as nn
import timm


class EfficientNetRegressor(nn.Module):
    """EfficientNet 角度回归模型

    修改 EfficientNet 的分类头为单神经元输出，使用 Sigmoid 激活
    确保输出在 [0, 1] 范围内（归一化角度）。

    Args:
        model_name: 模型名称（efficientnet_b4 或 efficientnet_b5）
        pretrained: 是否使用 ImageNet 预训练权重
        dropout: Dropout 概率
        freeze_backbone: 是否冻结骨干网络
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b4",
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.model_name = model_name

        # 加载预训练骨干网络
        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 移除分类头
        )

        # 冻结骨干网络
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # 获取特征维度
        in_features = backbone.num_features

        self.backbone = backbone

        # 回归头
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
            nn.Sigmoid(),
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
