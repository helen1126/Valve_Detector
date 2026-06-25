"""训练脚本

阀门角度检测模型的训练入口，使用 PyTorch Lightning 简化训练流程。
支持多种模型架构、组合损失函数、学习率调度和早停机制。

使用示例：
    # 使用 ConvNeXt-Base 训练（推荐）
    python train.py --model convnext_base --data_dir ./dataset --view all_view

    # 使用 ResNet50 训练
    python train.py --model resnet50 --epochs 100 --batch_size 16

    # 从检查点恢复训练
    python train.py --model convnext_base --resume ./weights/valve-epoch=XX-val_mae=XX.ckpt
"""

import argparse
import os
from pathlib import Path

import yaml
import torch
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger

from models import build_model
from data.dataset import ValveDataModule
from data.augmentation import get_transforms_from_config
from utils.logger import setup_logger, get_logger


class ValveRegressionModel(pl.LightningModule):
    """阀门角度回归模型

    基于 PyTorch Lightning 的训练模块，封装模型前向传播、
    损失计算、优化器配置和训练/验证步骤。

    Args:
        model_name: 模型架构名称
        pretrained: 是否使用预训练权重
        lr: 学习率
        weight_decay: 权重衰减
        mae_weight: MAE 损失权重
        mse_weight: MSE 损失权重
        angle_min: 最小角度
        angle_max: 最大角度
        dropout: Dropout 概率
        freeze_backbone: 是否冻结骨干网络
        T_max: 余弦退火周期
        eta_min: 最小学习率
    """

    def __init__(
        self,
        model_name: str = "convnext_base",
        pretrained: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        mae_weight: float = 0.7,
        mse_weight: float = 0.3,
        angle_min: float = 0.0,
        angle_max: float = 80.0,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        T_max: int = 50,
        eta_min: float = 1e-6,
        warmup_epochs: int = 0,
        side_weight: float = 5.0,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 角度范围
        self.angle_min = angle_min
        self.angle_max = angle_max

        # 损失权重
        self.mae_weight = mae_weight
        self.mse_weight = mse_weight
        self.side_weight = side_weight
        self.focal_gamma = focal_gamma

        # 渐进式解冻：warmup_epochs > 0 时，前 N 个 epoch 冻结骨干
        self.warmup_epochs = warmup_epochs
        self._warmup_done = False  # 标记 warmup 是否已完成

        # 损失函数（reduction='none' 以支持逐样本加权）
        self.l1_loss = torch.nn.L1Loss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        # 构建模型
        # warmup 模式下初始冻结骨干，否则按 freeze_backbone 参数决定
        should_freeze = freeze_backbone or (warmup_epochs > 0)
        self.model = build_model(
            model_name=model_name,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=should_freeze,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入图像张量

        Returns:
            归一化角度预测值
        """
        return self.model(x)

    def _unfreeze_backbone(self) -> None:
        """解冻骨干网络

        warmup 阶段结束后调用，解冻骨干网络并降低学习率，
        使微调过程更加稳定。
        """
        if self._warmup_done:
            return

        # 解冻骨干网络
        for param in self.model.backbone.parameters():
            param.requires_grad = True

        # 降低学习率为原来的 1/10，避免解冻后梯度过大
        optimizer = self.optimizers()
        for pg in optimizer.param_groups:
            pg["lr"] = self.hparams.lr * 0.1

        self._warmup_done = True
        self.print(f"[Epoch {self.current_epoch}] 骨干网络已解冻，学习率调整为 {self.hparams.lr * 0.1:.2e}")

    def _build_sample_weights(self, views: list) -> torch.Tensor:
        """根据视角标签构建样本权重

        side 视角样本给予更高权重，弥补其样本量少、学习难度大的问题。

        Args:
            views: 视角标签列表（字符串）

        Returns:
            样本权重张量 (B,)
        """
        weights = []
        for v in views:
            if v == "side":
                weights.append(self.side_weight)
            else:
                weights.append(1.0)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _compute_view_mae(
        self,
        pred_angles: torch.Tensor,
        true_angles: torch.Tensor,
        views: list,
    ) -> dict:
        """按视角分别计算 MAE

        Args:
            pred_angles: 预测角度（度数）(B,)
            true_angles: 真实角度（度数）(B,)
            views: 视角标签列表

        Returns:
            包含各视角 MAE 的字典
        """
        abs_errors = torch.abs(pred_angles - true_angles)
        side_mask = torch.tensor(
            [v == "side" for v in views], dtype=torch.bool, device=self.device
        )
        top_mask = ~side_mask

        result = {}
        if side_mask.any():
            result["side_mae"] = abs_errors[side_mask].mean()
        if top_mask.any():
            result["top_mae"] = abs_errors[top_mask].mean()
        return result

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算组合损失（支持逐样本加权 + 难样本挖掘）

        当 focal_gamma > 0 时，根据每个样本的预测误差自适应调整权重：
        focal_weight = (abs_error ^ gamma) / mean(abs_error ^ gamma)
        误差越大的样本权重越高，自动聚焦难样本。

        Args:
            predictions: 预测值（归一化）(B, 1)
            targets: 目标值（归一化）(B, 1)
            sample_weights: 逐样本权重 (B,)，None 时等权

        Returns:
            组合损失值
        """
        mae = self.l1_loss(predictions, targets).squeeze(1)  # (B,)
        mse = self.mse_loss(predictions, targets).squeeze(1)  # (B,)
        per_sample_loss = self.mae_weight * mae + self.mse_weight * mse

        # 难样本挖掘：根据误差自适应加权
        if self.focal_gamma > 0 and self.training:
            abs_error = mae.detach()  # 停止梯度，只用于权重计算
            # 归一化误差到 [0, 1] 范围（归一化角度空间最大误差为 1）
            focal_weight = (abs_error ** self.focal_gamma)
            focal_mean = focal_weight.mean()
            if focal_mean > 0:
                focal_weight = focal_weight / focal_mean
            per_sample_loss = per_sample_loss * focal_weight

        if sample_weights is not None:
            per_sample_loss = per_sample_loss * sample_weights

        return per_sample_loss.mean()

    def _denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """反归一化角度值

        Args:
            values: 归一化角度值

        Returns:
            原始角度值
        """
        return values * (self.angle_max - self.angle_min) + self.angle_min

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """训练步骤

        Args:
            batch: 数据批次
            batch_idx: 批次索引

        Returns:
            损失值
        """
        # 渐进式解冻：warmup 结束后解冻骨干网络
        if (
            self.warmup_epochs > 0
            and not self._warmup_done
            and self.current_epoch >= self.warmup_epochs
        ):
            self._unfreeze_backbone()

        images = batch["image"]
        angles = batch["angle"].unsqueeze(1)  # (B,) -> (B, 1)
        views = batch["view"]  # 视角标签列表

        # 前向传播
        predictions = self(images)

        # 构建视角感知的样本权重
        sample_weights = self._build_sample_weights(views)

        # 计算加权损失
        loss = self._compute_loss(predictions, angles, sample_weights)

        # 计算角度空间的 MAE（用于监控）
        pred_angles = self._denormalize(predictions).squeeze(1)
        true_angles = self._denormalize(angles).squeeze(1)
        mae_degrees = torch.mean(torch.abs(pred_angles - true_angles))

        # 记录日志
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mae", mae_degrees, prog_bar=True)

        # 分视角 MAE
        view_mae = self._compute_view_mae(pred_angles, true_angles, views)
        if "side_mae" in view_mae:
            self.log("train_side_mae", view_mae["side_mae"])
        if "top_mae" in view_mae:
            self.log("train_top_mae", view_mae["top_mae"])

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """验证步骤

        验证时使用等权损失，记录真实 MAE 和分视角 MAE。

        Args:
            batch: 数据批次
            batch_idx: 批次索引
        """
        images = batch["image"]
        angles = batch["angle"].unsqueeze(1)
        views = batch["view"]

        predictions = self(images)

        # 验证时使用等权损失，记录真实 MAE
        loss = self._compute_loss(predictions, angles)

        pred_angles = self._denormalize(predictions).squeeze(1)
        true_angles = self._denormalize(angles).squeeze(1)
        mae_degrees = torch.mean(torch.abs(pred_angles - true_angles))
        mse_degrees = torch.mean((pred_angles - true_angles) ** 2)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", mae_degrees, prog_bar=True)
        self.log("val_mse", mse_degrees)

        # 分视角 MAE
        view_mae = self._compute_view_mae(pred_angles, true_angles, views)
        if "side_mae" in view_mae:
            self.log("val_side_mae", view_mae["side_mae"])
        if "top_mae" in view_mae:
            self.log("val_top_mae", view_mae["top_mae"])

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """测试步骤

        Args:
            batch: 数据批次
            batch_idx: 批次索引
        """
        images = batch["image"]
        angles = batch["angle"].unsqueeze(1)

        predictions = self(images)

        pred_angles = self._denormalize(predictions)
        true_angles = self._denormalize(angles)
        mae_degrees = torch.mean(torch.abs(pred_angles - true_angles))

        self.log("test_mae", mae_degrees)

    def configure_optimizers(self) -> dict:
        """配置优化器和学习率调度器

        Returns:
            优化器和调度器配置
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.T_max,
            eta_min=self.hparams.eta_min,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 各模型在 384×384 输入下每个样本的近似显存占用（MB）
# 用于自适应 batch_size 计算
MODEL_VRAM_PER_SAMPLE = {
    "resnet50": 180,
    "resnet101": 260,
    "efficientnet_b4": 280,
    "efficientnet_b5": 420,
    "convnext_base": 520,
    "convnext_large": 800,
    "swin_base": 600,
    "swin_large": 900,
}


def auto_batch_size(model_name: str, image_size: int = 384, vram_reserve_gb: float = 1.5) -> int:
    """根据 GPU 显存自适应计算 batch_size

    根据模型类型和可用显存，自动计算合适的批大小。
    优先保证训练不会 OOM，同时尽量利用显存。

    Args:
        model_name: 模型架构名称
        image_size: 图像尺寸
        vram_reserve_gb: 预留显存（GB），用于系统和其他开销

    Returns:
        推荐的 batch_size
    """
    if not torch.cuda.is_available():
        return 16  # CPU 模式使用较小批大小

    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    available_vram_mb = total_vram_mb - vram_reserve_gb * 1024

    # 获取单样本显存估算
    vram_per_sample = MODEL_VRAM_PER_SAMPLE.get(model_name, 400)

    # 根据图像尺寸缩放（面积比）
    size_ratio = (image_size / 384) ** 2
    vram_per_sample = vram_per_sample * size_ratio

    # 计算最大 batch_size
    batch_size = int(available_vram_mb / vram_per_sample)

    # 限制在合理范围
    batch_size = max(2, min(batch_size, 128))
    # 向下对齐到 8 的倍数（至少为 2）
    batch_size = max(2, (batch_size // 8) * 8)

    return batch_size


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="阀门角度检测模型训练")

    # 数据参数
    parser.add_argument(
        "--data_dir", type=str, default="./dataset",
        help="数据集根目录"
    )
    parser.add_argument(
        "--view", type=str, default="all_view",
        choices=["all_view", "top_view", "side_view"],
        help="视角选择"
    )

    # 模型参数
    parser.add_argument(
        "--model", type=str, default="convnext_base",
        choices=["resnet50", "resnet101", "efficientnet_b4", "efficientnet_b5",
                 "convnext_base", "convnext_large", "swin_base", "swin_large"],
        help="模型架构"
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=200, help="最大训练轮数")
    parser.add_argument("--batch_size", type=int, default=0, help="批大小（0=自动根据显存适配）")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--image_size", type=int, default=384, help="图像尺寸")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="渐进式解冻warmup轮数（0=不使用，默认端到端训练）")

    # 损失函数参数
    parser.add_argument("--mae_weight", type=float, default=0.7, help="MAE 损失权重")
    parser.add_argument("--mse_weight", type=float, default=0.3, help="MSE 损失权重")
    parser.add_argument("--side_weight", type=float, default=5.0, help="side 视角样本损失权重（1.0=等权，>1 提升 side 重要性）")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="难样本挖掘的 focal gamma（0=关闭，>0 误差越大的样本权重越高）")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument(
        "--config_dir", type=str, default="./config",
        help="配置文件目录"
    )

    # 两阶段训练参数
    parser.add_argument("--stage2", action="store_true", help="启用两阶段训练：第一阶段用 all_view，第二阶段用 side_view 微调")
    parser.add_argument("--stage2_epochs", type=int, default=30, help="第二阶段训练轮数")
    parser.add_argument("--stage2_lr", type=float, default=1e-5, help="第二阶段学习率（默认 1e-5，比第一阶段低 10x）")

    return parser.parse_args()


def main():
    """训练主函数"""
    args = parse_args()

    # 初始化日志
    setup_logger(log_dir="./logs")
    logger = get_logger()

    # 设置随机种子
    pl.seed_everything(args.seed)

    # 加载配置
    data_config = load_config(os.path.join(args.config_dir, "data_config.yaml"))
    model_config = load_config(os.path.join(args.config_dir, "model_config.yaml"))
    train_config = load_config(os.path.join(args.config_dir, "train_config.yaml"))

    # 命令行参数覆盖配置文件
    image_size = args.image_size

    # 自适应 batch_size
    if args.batch_size == 0:
        batch_size = auto_batch_size(args.model, image_size)
        logger.info(f"自适应 batch_size: {batch_size}（模型: {args.model}, 图像尺寸: {image_size}）")
    else:
        batch_size = args.batch_size

    # 创建数据增强
    train_transforms, train_transforms_side, val_transforms = get_transforms_from_config(
        data_config, image_size=image_size
    )

    # 创建数据模块
    data_module = ValveDataModule(
        data_dir=args.data_dir,
        view=args.view,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=args.num_workers,
        train_ratio=data_config.get("train_ratio", 0.8),
        val_ratio=data_config.get("val_ratio", 0.1),
        test_ratio=data_config.get("test_ratio", 0.1),
        seed=args.seed,
        train_transform=train_transforms,
        train_transform_side=train_transforms_side,
        val_transform=val_transforms,
        angle_min=data_config.get("angle_min", 0.0),
        angle_max=data_config.get("angle_max", 80.0),
    )

    # 创建模型
    model = ValveRegressionModel(
        model_name=args.model,
        pretrained=model_config.get("pretrained", True),
        lr=args.lr,
        weight_decay=args.weight_decay,
        mae_weight=args.mae_weight,
        mse_weight=args.mse_weight,
        angle_min=data_config.get("angle_min", 0.0),
        angle_max=data_config.get("angle_max", 80.0),
        dropout=model_config.get("dropout", 0.2),
        freeze_backbone=model_config.get("freeze_backbone", False),
        T_max=train_config.get("scheduler", {}).get("T_max", 50),
        eta_min=train_config.get("scheduler", {}).get("eta_min", 1e-6),
        warmup_epochs=args.warmup_epochs,
        side_weight=args.side_weight,
        focal_gamma=args.focal_gamma,
    )

    logger.info(f"模型: {args.model}")
    logger.info(f"视角: {args.view}")
    logger.info(f"图像尺寸: {image_size}")
    logger.info(f"批大小: {batch_size}")
    if args.warmup_epochs > 0:
        logger.info(f"渐进式解冻: 前 {args.warmup_epochs} 个 epoch 冻结骨干，之后解冻微调")
    else:
        logger.info("训练模式: 端到端训练（不冻结骨干网络）")
    logger.info(f"学习率: {args.lr}")
    if args.side_weight != 1.0:
        logger.info(f"视角加权: side 视角损失权重 = {args.side_weight}")
    if args.focal_gamma > 0:
        logger.info(f"难样本挖掘: focal_gamma = {args.focal_gamma}")

    # 回调函数
    callbacks = []

    # 早停
    early_stop_config = train_config.get("early_stopping", {})
    if early_stop_config.get("enabled", True):
        callbacks.append(
            EarlyStopping(
                monitor=early_stop_config.get("monitor", "val_mae"),
                patience=early_stop_config.get("patience", 10),
                mode=early_stop_config.get("mode", "min"),
                min_delta=early_stop_config.get("min_delta", 0.01),
            )
        )

    # 模型检查点
    ckpt_config = train_config.get("checkpoint", {})
    callbacks.append(
        ModelCheckpoint(
            dirpath=model_config.get("save_dir", "./weights"),
            monitor=ckpt_config.get("monitor", "val_mae"),
            mode=ckpt_config.get("mode", "min"),
            save_top_k=ckpt_config.get("save_top_k", 3),
            filename=ckpt_config.get("filename", "valve-{epoch:03d}-{val_mae:.4f}"),
            save_last=ckpt_config.get("save_last", True),
        )
    )

    # 学习率监控
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # TensorBoard 日志
    tb_logger = TensorBoardLogger(
        save_dir=train_config.get("tensorboard_dir", "./logs/tensorboard"),
        name=args.model,
    )

    # 训练器
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=tb_logger,
        accelerator="auto",
        devices="auto",
        precision=train_config.get("precision", "16-mixed"),
        gradient_clip_val=train_config.get("gradient_clip", {}).get("max_norm", 1.0)
            if train_config.get("gradient_clip", {}).get("enabled", True) else 0.0,
        log_every_n_steps=train_config.get("log_every_n_steps", 10),
        val_check_interval=train_config.get("val_check_interval", 1.0),
        deterministic=False,
    )

    # 开始训练
    logger.info("开始训练...")
    trainer.fit(model, data_module, ckpt_path=args.resume)

    # 训练完成后在测试集上评估
    logger.info("训练完成，在测试集上评估...")
    trainer.test(model, data_module)

    # 两阶段训练：第二阶段用 side_view 微调
    if args.stage2:
        logger.info("=" * 60)
        logger.info("第二阶段训练：使用 side_view 数据微调")
        logger.info(f"学习率: {args.stage2_lr}")
        logger.info(f"训练轮数: {args.stage2_epochs}")
        logger.info("=" * 60)

        # 找到第一阶段的最佳 checkpoint
        best_ckpt = None
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint):
                best_ckpt = cb.best_model_path
                break

        if best_ckpt is None:
            best_ckpt = os.path.join(
                model_config.get("save_dir", "./weights"), "last.ckpt"
            )

        logger.info(f"加载第一阶段最佳模型: {best_ckpt}")

        # 创建 side_view 数据模块
        side_data_module = ValveDataModule(
            data_dir=args.data_dir,
            view="side_view",
            image_size=image_size,
            batch_size=batch_size,
            num_workers=args.num_workers,
            train_ratio=data_config.get("train_ratio", 0.8),
            val_ratio=data_config.get("val_ratio", 0.1),
            test_ratio=data_config.get("test_ratio", 0.1),
            seed=args.seed,
            train_transform=train_transforms,
            train_transform_side=train_transforms_side,
            val_transform=val_transforms,
            angle_min=data_config.get("angle_min", 0.0),
            angle_max=data_config.get("angle_max", 80.0),
        )

        # 加载第一阶段模型，覆盖学习率
        stage2_model = ValveRegressionModel.load_from_checkpoint(
            best_ckpt,
            lr=args.stage2_lr,
        )

        # 第二阶段的回调
        stage2_callbacks = []
        stage2_callbacks.append(
            EarlyStopping(
                monitor="val_mae",
                patience=15,
                mode="min",
                min_delta=0.01,
            )
        )
        stage2_callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(
                    model_config.get("save_dir", "./weights"), "stage2"
                ),
                monitor="val_mae",
                mode="min",
                save_top_k=3,
                filename="valve-stage2-{epoch:03d}-{val_mae:.4f}",
                save_last=True,
            )
        )
        stage2_callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        # 第二阶段日志
        stage2_tb_logger = TensorBoardLogger(
            save_dir=train_config.get("tensorboard_dir", "./logs/tensorboard"),
            name=f"{args.model}_stage2",
        )

        # 第二阶段训练器
        stage2_trainer = pl.Trainer(
            max_epochs=args.stage2_epochs,
            callbacks=stage2_callbacks,
            logger=stage2_tb_logger,
            accelerator="auto",
            devices="auto",
            precision=train_config.get("precision", "16-mixed"),
            gradient_clip_val=train_config.get("gradient_clip", {}).get("max_norm", 1.0)
                if train_config.get("gradient_clip", {}).get("enabled", True) else 0.0,
            log_every_n_steps=train_config.get("log_every_n_steps", 10),
            deterministic=False,
        )

        stage2_trainer.fit(stage2_model, side_data_module)
        stage2_trainer.test(stage2_model, side_data_module)

        logger.info("第二阶段训练完成！")

    logger.info("全部完成！")


if __name__ == "__main__":
    main()
