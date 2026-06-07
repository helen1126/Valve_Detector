"""数据增强模块

基于 Albumentations 实现阀门角度检测任务的数据增强策略。
注意：不使用旋转和翻转增强，因为会改变阀门角度的视觉表现。
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional


def get_train_transforms(
    image_size: int = 384,
    config: Optional[dict] = None,
) -> A.Compose:
    """获取训练集数据增强流水线

    Args:
        image_size: 目标图像尺寸
        config: 数据增强配置字典（来自 data_config.yaml）

    Returns:
        Albumentations 增强流水线
    """
    transforms_list = []

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
        (train_transforms, val_transforms) 元组
    """
    aug_config = config.get("augmentation", {})
    if aug_config.get("enabled", True):
        train_transforms = get_train_transforms(image_size, aug_config)
    else:
        train_transforms = get_val_transforms(image_size)

    val_transforms = get_val_transforms(image_size)

    return train_transforms, val_transforms
