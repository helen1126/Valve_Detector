"""数据增强模块

基于 Albumentations 实现阀门角度检测任务的数据增强策略。
注意：不使用旋转和翻转增强，因为会改变阀门角度的视觉表现。
"""

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from typing import Optional

import numpy as np


class ValveSmartCrop(ImageOnlyTransform):
    """训练时智能裁剪：以一定概率裁剪并放大阀门区域

    让模型在训练时也能见到裁剪放大后的图像，
    减少训练-推理的分布不一致问题。
    """

    def __init__(
        self,
        min_area_ratio: float = 0.15,
        padding_ratio: float = 0.25,
        min_padding: int = 20,
        always_apply: bool = False,
        p: float = 0.3,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.min_area_ratio = min_area_ratio
        self.padding_ratio = padding_ratio
        self.min_padding = min_padding
        self._optimizer = None  # 延迟初始化，避免 import 循环

    def _get_optimizer(self):
        """延迟初始化 ImageOptimizer"""
        if self._optimizer is None:
            from utils.image_optimization import ImageOptimizer
            self._optimizer = ImageOptimizer()
        return self._optimizer

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """执行智能裁剪"""
        optimizer = self._get_optimizer()
        cropped, was_cropped = optimizer.smart_crop(
            img,
            min_area_ratio=self.min_area_ratio,
            padding_ratio=self.padding_ratio,
            min_padding=self.min_padding,
        )
        return cropped

    def get_transform_init_args_names(self):
        return ("min_area_ratio", "padding_ratio", "min_padding")


def get_train_transforms(
    image_size: int = 384,
    config: Optional[dict] = None,
    is_side: bool = False,
) -> A.Compose:
    """获取训练集数据增强流水线

    Args:
        image_size: 目标图像尺寸
        config: 数据增强配置字典（来自 data_config.yaml）
        is_side: 是否为 side 视角样本（side 使用更强的畸变增强）

    Returns:
        Albumentations 增强流水线
    """
    transforms_list = []

    # 智能裁剪：以一定概率裁剪并放大阀门区域，模拟远距离拍摄
    if config is not None and config.get("smart_crop", {}).get("enabled", True):
        crop_config = config.get("smart_crop", {})
        transforms_list.append(
            ValveSmartCrop(
                min_area_ratio=crop_config.get("min_area_ratio", 0.15),
                padding_ratio=crop_config.get("padding_ratio", 0.25),
                min_padding=crop_config.get("min_padding", 20),
                p=crop_config.get("p", 0.3),
            )
        )

    # 随机裁剪缩放
    if config is None or config.get("random_resized_crop", {}).get("enabled", True):
        crop_config = (config or {}).get("random_resized_crop", {})
        transforms_list.append(
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=tuple(crop_config.get("scale", [0.8, 1.0])),
                ratio=tuple(crop_config.get("ratio", [0.9, 1.1])),
                p=crop_config.get("p", 0.5),
            )
        )
    else:
        transforms_list.append(
            A.Resize(height=image_size, width=image_size)
        )

    # 透视变换：模拟 side 视角的透视变形
    if config is None or config.get("perspective", {}).get("enabled", True):
        persp_config = (config or {}).get("perspective", {})
        persp_p = persp_config.get("p", 0.3)
        if is_side:
            persp_p = min(persp_p * 2, 0.8)
        transforms_list.append(
            A.Perspective(
                scale=tuple(persp_config.get("scale", [0.08, 0.15])),
                keep_size=persp_config.get("keep_size", True),
                p=persp_p,
            )
        )

    # 镜头畸变：模拟 side 视角的桶形/枕形畸变
    if config is None or config.get("optical_distortion", {}).get("enabled", True):
        distort_config = (config or {}).get("optical_distortion", {})
        distort_p = distort_config.get("p", 0.4)
        if is_side:
            distort_p = min(distort_p * 2, 0.8)
        transforms_list.append(
            A.OpticalDistortion(
                distort_limit=tuple(distort_config.get("distort_limit", [-0.3, 0.3])),
                shift_limit=tuple(distort_config.get("shift_limit", [-0.1, 0.1])),
                interpolation=distort_config.get("interpolation", 1),
                border_mode=distort_config.get("border_mode", 0),
                p=distort_p,
            )
        )

    # 随机缩放：模拟不同拍摄距离
    if config is None or config.get("random_scale", {}).get("enabled", True):
        scale_config = (config or {}).get("random_scale", {})
        transforms_list.append(
            A.RandomScale(
                scale_limit=scale_config.get("scale_limit", 0.5),
                interpolation=scale_config.get("interpolation", 1),
                p=scale_config.get("p", 0.3),
            )
        )

    # 随机亮度对比度
    if config is None or config.get("random_brightness_contrast", {}).get("enabled", True):
        bc_config = (config or {}).get("random_brightness_contrast", {})
        transforms_list.append(
            A.RandomBrightnessContrast(
                brightness_limit=bc_config.get("brightness_limit", 0.2),
                contrast_limit=bc_config.get("contrast_limit", 0.2),
                p=bc_config.get("p", 0.5),
            )
        )

    # 色调饱和度变化
    if config is None or config.get("hue_saturation_value", {}).get("enabled", True):
        hsv_config = (config or {}).get("hue_saturation_value", {})
        transforms_list.append(
            A.HueSaturationValue(
                hue_shift_limit=hsv_config.get("hue_shift_limit", 20),
                sat_shift_limit=hsv_config.get("sat_shift_limit", 30),
                val_shift_limit=hsv_config.get("val_shift_limit", 20),
                p=hsv_config.get("p", 0.5),
            )
        )

    # 高斯模糊
    if config is None or config.get("gaussian_blur", {}).get("enabled", True):
        blur_config = (config or {}).get("gaussian_blur", {})
        transforms_list.append(
            A.GaussianBlur(
                blur_limit=blur_config.get("blur_limit", 7),
                p=blur_config.get("p", 0.3),
            )
        )

    # 高斯噪声
    if config is None or config.get("gauss_noise", {}).get("enabled", True):
        noise_config = (config or {}).get("gauss_noise", {})
        transforms_list.append(
            A.GaussNoise(
                var_limit=tuple(noise_config.get("var_limit", [10.0, 50.0])),
                p=noise_config.get("p", 0.3),
            )
        )

    # 随机遮挡
    if config is None or config.get("coarse_dropout", {}).get("enabled", True):
        dropout_config = (config or {}).get("coarse_dropout", {})
        transforms_list.append(
            A.CoarseDropout(
                max_holes=dropout_config.get("max_holes", 8),
                max_height=dropout_config.get("max_height", 32),
                max_width=dropout_config.get("max_width", 32),
                min_holes=dropout_config.get("min_holes", 1),
                min_height=dropout_config.get("min_height", 8),
                min_width=dropout_config.get("min_width", 8),
                p=dropout_config.get("p", 0.3),
            )
        )

    # 锐化
    if config is None or config.get("sharpen", {}).get("enabled", True):
        sharpen_config = (config or {}).get("sharpen", {})
        transforms_list.append(
            A.Sharpen(
                alpha=tuple(sharpen_config.get("alpha", [0.2, 0.5])),
                lightness=tuple(sharpen_config.get("lightness", [0.5, 1.0])),
                p=sharpen_config.get("p", 0.3),
            )
        )

    # RGB 偏移
    if config is None or config.get("rgb_shift", {}).get("enabled", True):
        rgb_config = (config or {}).get("rgb_shift", {})
        transforms_list.append(
            A.RGBShift(
                r_shift_limit=rgb_config.get("r_shift_limit", 15),
                g_shift_limit=rgb_config.get("g_shift_limit", 15),
                b_shift_limit=rgb_config.get("b_shift_limit", 15),
                p=rgb_config.get("p", 0.3),
            )
        )

    # 通道混洗
    if config is not None and config.get("channel_shuffle", {}).get("enabled", False):
        shuffle_config = config.get("channel_shuffle", {})
        transforms_list.append(
            A.ChannelShuffle(p=shuffle_config.get("p", 0.1))
        )

    # 确保尺寸统一
    transforms_list.append(A.Resize(height=image_size, width=image_size))

    return A.Compose(transforms_list)


def get_val_transforms(image_size: int = 384) -> A.Compose:
    """获取验证集/测试集预处理流水线

    验证集和测试集仅做 Resize 和 Normalize，不做数据增强。

    Args:
        image_size: 目标图像尺寸

    Returns:
        Albumentations 预处理流水线
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
    ])


def get_transforms_from_config(
    config: dict,
    image_size: int = 384,
) -> tuple:
    """从配置字典创建训练和验证的变换流水线

    Args:
        config: 数据配置字典
        image_size: 目标图像尺寸

    Returns:
        (train_transforms, train_transforms_side, val_transforms) 元组
    """
    aug_config = config.get("augmentation", {})
    if aug_config.get("enabled", True):
        train_transforms = get_train_transforms(image_size, aug_config, is_side=False)
        train_transforms_side = get_train_transforms(image_size, aug_config, is_side=True)
    else:
        train_transforms = get_val_transforms(image_size)
        train_transforms_side = get_val_transforms(image_size)

    val_transforms = get_val_transforms(image_size)

    return train_transforms, train_transforms_side, val_transforms
