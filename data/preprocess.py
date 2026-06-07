"""图像预处理模块

提供阀门图像的预处理功能，包括尺寸统一、亮度/对比度调整、
去噪、颜色空间转换和光照校正等。
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from utils.image_utils import bgr_to_hsv, hsv_to_bgr


class PreprocessPipeline:
    """图像预处理流水线

    可配置组合多种预处理操作，按顺序执行。

    Args:
        image_size: 目标图像尺寸
        brightness: 亮度调整系数（1.0 为原始亮度）
        contrast: 对比度调整系数（1.0 为原始对比度）
        saturation: 饱和度调整系数（1.0 为原始饱和度）
        denoise: 是否启用去噪
        denoise_strength: 去噪强度
        clahe: 是否启用 CLAHE 光照校正
        clahe_clip_limit: CLAHE 裁剪限制
        clahe_grid_size: CLAHE 网格大小
    """

    def __init__(
        self,
        image_size: int = 384,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        denoise: bool = False,
        denoise_strength: int = 10,
        clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
    ):
        self.image_size = image_size
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.denoise = denoise
        self.denoise_strength = denoise_strength
        self.clahe = clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """执行预处理流水线

        Args:
            image: BGR 格式输入图像

        Returns:
            预处理后的图像
        """
        # 1. 调整尺寸
        image = self._resize(image, self.image_size)

        # 2. 亮度/对比度调整
        if self.brightness != 1.0 or self.contrast != 1.0:
            image = self._adjust_brightness_contrast(
                image, self.brightness, self.contrast
            )

        # 3. 饱和度调整
        if self.saturation != 1.0:
            image = self._adjust_saturation(image, self.saturation)

        # 4. 去噪
        if self.denoise:
            image = self._denoise(image, self.denoise_strength)

        # 5. 光照校正
        if self.clahe:
            image = self._apply_clahe(
                image, self.clahe_clip_limit, self.clahe_grid_size
            )

        return image

    @staticmethod
    def _resize(image: np.ndarray, size: int) -> np.ndarray:
        """调整图像尺寸（保持宽高比，填充到正方形）

        Args:
            image: 输入图像
            size: 目标尺寸

        Returns:
            调整后的图像
        """
        h, w = image.shape[:2]
        scale = size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

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
    def _adjust_brightness_contrast(
        image: np.ndarray, brightness: float, contrast: float
    ) -> np.ndarray:
        """调整亮度和对比度

        Args:
            image: 输入图像
            brightness: 亮度系数
            contrast: 对比度系数

        Returns:
            调整后的图像
        """
        # brightness: 乘法调整
        # contrast: 以 128 为中心的对比度调整
        result = np.clip(
            image.astype(np.float32) * brightness * contrast
            + 128 * (1 - contrast),
            0, 255
        ).astype(np.uint8)
        return result

    @staticmethod
    def _adjust_saturation(image: np.ndarray, saturation: float) -> np.ndarray:
        """调整饱和度

        Args:
            image: 输入图像（BGR 格式）
            saturation: 饱和度系数

        Returns:
            调整后的图像
        """
        hsv = bgr_to_hsv(image)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * saturation, 0, 255)
        return hsv_to_bgr(hsv)

    @staticmethod
    def _denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
        """高斯去噪

        Args:
            image: 输入图像
            strength: 去噪强度

        Returns:
            去噪后的图像
        """
        return cv2.fastNlMeansDenoisingColored(
            image, None, strength, strength, 7, 21
        )

    @staticmethod
    def _apply_clahe(
        image: np.ndarray,
        clip_limit: float = 2.0,
        grid_size: int = 8,
    ) -> np.ndarray:
        """应用 CLAHE 自适应直方图均衡化

        Args:
            image: 输入图像（BGR 格式）
            clip_limit: 裁剪限制
            grid_size: 网格大小

        Returns:
            处理后的图像
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(grid_size, grid_size)
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_green_red_contrast(
    image: np.ndarray,
    green_boost: float = 1.5,
    red_boost: float = 1.5,
) -> np.ndarray:
    """增强绿色和红色区域的对比度

    在 HSV 空间中增强绿色和红色通道的饱和度和明度，
    突出阀门角度盘的颜色特征。

    Args:
        image: BGR 格式输入图像
        green_boost: 绿色增强系数
        red_boost: 红色增强系数

    Returns:
        增强后的图像
    """
    hsv = bgr_to_hsv(image).astype(np.float32)

    # 绿色区域（H: 35-85）
    green_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85)
    hsv[green_mask, 1] = np.clip(hsv[green_mask, 1] * green_boost, 0, 255)

    # 红色区域（H: 0-10 或 170-180）
    red_mask = (
        (hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)
    ) & (hsv[:, :, 1] >= 50)
    hsv[red_mask, 1] = np.clip(hsv[red_mask, 1] * red_boost, 0, 255)

    return hsv_to_bgr(hsv.astype(np.uint8))


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Gamma 校正

    Args:
        image: 输入图像
        gamma: Gamma 值（<1 提亮，>1 变暗）

    Returns:
        校正后的图像
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
    ]).astype(np.uint8)
    return cv2.LUT(image, table)


def extract_dial_region(
    image: np.ndarray,
    min_area_ratio: float = 0.05,
) -> np.ndarray:
    """基于颜色特征提取角度盘区域

    通过检测绿色和红色区域来定位角度盘，裁剪并返回包含角度盘的区域。

    Args:
        image: BGR 格式输入图像
        min_area_ratio: 最小面积比例（相对于图像总面积）

    Returns:
        裁剪后的角度盘区域图像，如果未检测到则返回原图
    """
    from utils.image_utils import extract_green_mask, extract_red_mask

    # 提取绿色和红色掩码
    green_mask = extract_green_mask(image)
    red_mask = extract_red_mask(image)
    combined_mask = cv2.bitwise_or(green_mask, red_mask)

    # 形态学操作，填充空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    # 找到最大轮廓
    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)

    # 面积太小则返回原图
    if area < image.shape[0] * image.shape[1] * min_area_ratio:
        return image

    # 获取边界框
    x, y, w, h = cv2.boundingRect(max_contour)

    # 扩展边界框（增加 10% 边距）
    margin = int(max(w, h) * 0.1)
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)

    return image[y:y+h, x:x+w]
