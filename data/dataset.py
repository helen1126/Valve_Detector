"""数据集模块

提供阀门角度检测数据集的加载、划分和 DataLoader 创建功能。
"""

import os
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from pytorch_lightning import LightningDataModule

from utils.image_utils import read_image


class ValveDataset(Dataset):
    """阀门角度检测数据集

    从指定目录加载图片，从文件名中解析角度标签。
    文件名格式：编号_角度.后缀名（例如：001_45.5.jpg）

    Args:
        data_dir: 数据集根目录
        view: 视角选择（all_view/top_view/side_view）
        image_size: 图像尺寸
        transform: 数据增强/预处理变换
        angle_min: 最小角度
        angle_max: 最大角度
        extensions: 支持的图片格式
    """

    # 文件名解析正则：编号_角度.后缀
    FILENAME_PATTERN = re.compile(r"(\d+)_(\d+\.?\d*)\.")

    def __init__(
        self,
        data_dir: str = "./dataset",
        view: str = "all_view",
        image_size: int = 384,
        transform=None,
        transform_side=None,
        angle_min: float = 0.0,
        angle_max: float = 80.0,
        extensions: Optional[List[str]] = None,
        oversample_side: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.view = view
        self.image_size = image_size
        self.transform = transform
        self.transform_side = transform_side  # side 视角专用增强（更强畸变）
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.extensions = extensions or [".jpg", ".jpeg", ".png", ".bmp"]
        self.oversample_side = oversample_side

        # 加载数据
        self.samples = self._load_samples()
        if len(self.samples) == 0:
            raise ValueError(
                f"未找到有效数据，请检查数据目录: {self.data_dir / self.view}"
            )

    def _infer_view(self, file_path: Path) -> str:
        """从文件路径推断视角标签

        Args:
            file_path: 图片文件路径

        Returns:
            视角标签（"top" / "side" / "unknown"）
        """
        path_str = str(file_path).replace("\\", "/").lower()
        if "side_view" in path_str:
            return "side"
        elif "top_view" in path_str:
            return "top"
        return "unknown"

    def _load_samples(self) -> List[Dict]:
        """加载所有样本，解析文件名获取角度标签和视角标签

        Returns:
            样本列表，每项包含 image_path、angle、view 和 filename
        """
        view_dir = self.data_dir / self.view
        if not view_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {view_dir}")

        samples = []
        for file_path in sorted(view_dir.iterdir()):
            if file_path.suffix.lower() not in self.extensions:
                continue

            match = self.FILENAME_PATTERN.search(file_path.name)
            if match is None:
                continue

            angle = float(match.group(2))

            # 验证角度范围
            if angle < self.angle_min or angle > self.angle_max:
                continue

            samples.append({
                "image_path": str(file_path),
                "angle": angle,
                "view": self._infer_view(file_path),
                "filename": file_path.name,
            })

        # side 样本过采样：重复 side 样本使数量接近 top 样本
        if self.oversample_side and self.view == "all_view":
            side_samples = [s for s in samples if s["view"] == "side"]
            top_samples = [s for s in samples if s["view"] == "top"]
            if side_samples and top_samples:
                n_top = len(top_samples)
                n_side = len(side_samples)
                # 计算需要重复的次数（向上取整）
                repeat = max(1, (n_top + n_side - 1) // n_side) - 1
                if repeat > 0:
                    samples = samples + side_samples * repeat

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, float, str]]:
        """获取单个样本

        Args:
            idx: 样本索引

        Returns:
            包含图像张量、归一化角度、原始角度和文件名的字典
        """
        sample = self.samples[idx]

        # 读取图像
        image = read_image(sample["image_path"])

        # 调整尺寸
        image = self._resize_image(image, self.image_size)

        # 根据视角选择增强流水线：side 使用更强的畸变增强
        transform = self.transform
        if sample["view"] == "side" and self.transform_side is not None:
            transform = self.transform_side

        # 应用数据增强/预处理
        if transform is not None:
            transformed = transform(image=image)
            image = transformed["image"]

        # 转换为张量 (H, W, C) -> (C, H, W)，归一化到 [0, 1]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # 归一化角度到 [0, 1]
        normalized_angle = (sample["angle"] - self.angle_min) / (
            self.angle_max - self.angle_min
        )

        return {
            "image": image,
            "angle": torch.tensor(normalized_angle, dtype=torch.float32),
            "raw_angle": sample["angle"],
            "view": sample["view"],
            "filename": sample["filename"],
        }

    @staticmethod
    def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
        """调整图像尺寸（保持宽高比，填充到正方形）

        Args:
            image: 输入图像
            size: 目标尺寸

        Returns:
            调整后的图像
        """
        h, w = image.shape[:2]

        # 计算缩放比例
        scale = size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 填充到正方形
        pad_h = size - new_h
        pad_w = size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        return padded

    @staticmethod
    def split_dataset(
        dataset: "ValveDataset",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[Subset, Subset, Subset]:
        """将数据集划分为训练集、验证集和测试集

        Args:
            dataset: 完整数据集
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子

        Returns:
            (训练集, 验证集, 测试集) 子集
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "数据划分比例之和必须为 1.0"

        n = len(dataset)
        indices = list(range(n))

        # 固定随机种子确保可复现
        rng = random.Random(seed)
        rng.shuffle(indices)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        return (
            Subset(dataset, train_indices),
            Subset(dataset, val_indices),
            Subset(dataset, test_indices),
        )

    def get_angle_distribution(self) -> Dict[str, int]:
        """获取角度分布统计

        Returns:
            各角度区间的样本数量
        """
        distribution = {}
        for sample in self.samples:
            # 每 10 度一个区间
            interval_key = f"{int(sample['angle'] // 10) * 10}-{int(sample['angle'] // 10 + 1) * 10}"
            distribution[interval_key] = distribution.get(interval_key, 0) + 1

        return distribution


class ValveDataModule(LightningDataModule):
    """阀门角度检测数据模块

    封装数据加载、预处理、增强和 DataLoader 创建。

    Args:
        data_dir: 数据集根目录
        view: 视角选择
        image_size: 图像尺寸
        batch_size: 批大小
        num_workers: 数据加载线程数
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        train_transform: 训练集数据增强
        val_transform: 验证集/测试集预处理
        angle_min: 最小角度
        angle_max: 最大角度
    """

    def __init__(
        self,
        data_dir: str = "./dataset",
        view: str = "all_view",
        image_size: int = 384,
        batch_size: int = 32,
        num_workers: int = 4,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        train_transform=None,
        train_transform_side=None,
        val_transform=None,
        angle_min: float = 0.0,
        angle_max: float = 80.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.view = view
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.train_transform = train_transform
        self.train_transform_side = train_transform_side
        self.val_transform = val_transform
        self.angle_min = angle_min
        self.angle_max = angle_max

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """初始化数据集

        Args:
            stage: 训练阶段（fit/validate/test）
        """
        # 创建完整数据集（不带变换）
        full_dataset = ValveDataset(
            data_dir=self.data_dir,
            view=self.view,
            image_size=self.image_size,
            angle_min=self.angle_min,
            angle_max=self.angle_max,
        )

        # 划分数据集
        train_subset, val_subset, test_subset = ValveDataset.split_dataset(
            full_dataset,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed,
        )

        # 为训练集设置数据增强
        if self.train_transform is not None:
            train_subset.dataset = ValveDataset(
                data_dir=self.data_dir,
                view=self.view,
                image_size=self.image_size,
                transform=self.train_transform,
                transform_side=self.train_transform_side,
                angle_min=self.angle_min,
                angle_max=self.angle_max,
            )

        # 为验证集和测试集设置预处理
        if self.val_transform is not None:
            val_subset.dataset = ValveDataset(
                data_dir=self.data_dir,
                view=self.view,
                image_size=self.image_size,
                transform=self.val_transform,
                angle_min=self.angle_min,
                angle_max=self.angle_max,
            )
            test_subset.dataset = ValveDataset(
                data_dir=self.data_dir,
                view=self.view,
                image_size=self.image_size,
                transform=self.val_transform,
                angle_min=self.angle_min,
                angle_max=self.angle_max,
            )

        self.train_dataset = train_subset
        self.val_dataset = val_subset
        self.test_dataset = test_subset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

