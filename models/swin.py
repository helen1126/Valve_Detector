"""Swin Transformer 回归模型

基于 timm 的 Swin Transformer 架构，修改分类头为单神经元回归输出。
"""

import torch
import torch.nn as nn
import timm


class SwinRegressor(nn.Module):
    """Swin Transformer 角度回归模型

    修改 Swin Transformer 的分类头为单神经元输出，使用 Sigmoid 激活
    确保输出在 [0, 1] 范围内（归一化角度）。

    Args:
        model_name: 模型名称（swin_base 或 swin_large）
        pretrained: 是否使用 ImageNet 预训练权重
        dropout: Dropout 概率
        freeze_backbone: 是否冻结骨干网络
    """

    def __init__(
        self,
        model_name: str = "swin_base",
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.model_name = model_name

        # timm 中的 Swin 模型名称映射
        timm_name_map = {
            "swin_base": "swin_base_patch4_window12_384.fb_in22k_ft_in1k",
            "swin_large": "swin_large_patch4_window12_384.fb_in22k_ft_in1k",
        }

        timm_name = timm_name_map.get(model_name, model_name)

        # 加载预训练骨干网络
        backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,  # 移除分类头
            img_size=384,
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
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
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
                nn.init.trunc_normal_(module.weight, std=0.02)
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
