"""图像优化模块

提供专门针对阀门角度盘的图像优化技术，包括颜色增强、
边缘检测、区域提取和光照校正等。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from utils.image_utils import (
    read_image,
    save_image,
    bgr_to_rgb,
    bgr_to_hsv,
    hsv_to_bgr,
    extract_green_mask,
    extract_red_mask,
)

# 设置 matplotlib 中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class ImageOptimizer:
    """阀门图像优化器

    提供多种图像优化操作，可配置组合使用。

    Args:
        green_boost: 绿色通道增强系数
        red_boost: 红色通道增强系数
        edge_weight: 边缘叠加权重
        clahe_clip_limit: CLAHE 裁剪限制
        gamma: Gamma 校正值
    """

    def __init__(
        self,
        green_boost: float = 1.5,
        red_boost: float = 1.5,
        edge_weight: float = 0.3,
        clahe_clip_limit: float = 2.0,
        gamma: float = 1.0,
    ):
        self.green_boost = green_boost
        self.red_boost = red_boost
        self.edge_weight = edge_weight
        self.clahe_clip_limit = clahe_clip_limit
        self.gamma = gamma

    def enhance_colors(self, image: np.ndarray) -> np.ndarray:
        """颜色增强：突出绿色和红色区域

        在 HSV 空间中增强绿色和红色区域的饱和度，
        使角度盘的颜色特征更加明显。

        Args:
            image: BGR 格式输入图像

        Returns:
            颜色增强后的图像
        """
        hsv = bgr_to_hsv(image).astype(np.float32)

        # 增强绿色区域饱和度
        green_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 50)
        hsv[green_mask, 1] = np.clip(
            hsv[green_mask, 1] * self.green_boost, 0, 255
        )

        # 增强红色区域饱和度
        red_mask = (
            ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 50)
        )
        hsv[red_mask, 1] = np.clip(
            hsv[red_mask, 1] * self.red_boost, 0, 255
        )

        return hsv_to_bgr(hsv.astype(np.uint8))

    def enhance_edges(
        self, image: np.ndarray, method: str = "sobel"
    ) -> np.ndarray:
        """边缘检测增强

        提取图像边缘信息并叠加到原图上，增强角度盘的边缘特征。

        Args:
            image: BGR 格式输入图像
            method: 边缘检测方法（sobel/canny）

        Returns:
            边缘增强后的图像
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if method == "sobel":
            # Sobel 边缘检测
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            edges = np.clip(edges, 0, 255).astype(np.uint8)
        elif method == "canny":
            # Canny 边缘检测
            edges = cv2.Canny(gray, 50, 150)
        else:
            raise ValueError(f"不支持的边缘检测方法: {method}")

        # 叠加边缘到原图
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(
            image, 1.0, edges_color, self.edge_weight, 0
        )

        return result

    def extract_dial_region(
        self, image: np.ndarray, min_area_ratio: float = 0.05
    ) -> np.ndarray:
        """区域提取：自动定位并裁剪角度盘区域

        基于绿色和红色颜色特征定位角度盘，裁剪并返回包含角度盘的区域。

        Args:
            image: BGR 格式输入图像
            min_area_ratio: 最小面积比例

        Returns:
            裁剪后的角度盘区域图像
        """
        # 提取颜色掩码
        green_mask = extract_green_mask(image)
        red_mask = extract_red_mask(image)
        combined_mask = cv2.bitwise_or(green_mask, red_mask)

        # 形态学操作
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

        if area < image.shape[0] * image.shape[1] * min_area_ratio:
            return image

        # 获取边界框并扩展
        x, y, w, h = cv2.boundingRect(max_contour)
        margin = int(max(w, h) * 0.1)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        return image[y:y+h, x:x+w]

    def correct_lighting(self, image: np.ndarray) -> np.ndarray:
        """光照校正：消除不均匀光照的影响

        使用 CLAHE 自适应直方图均衡化和 Gamma 校正。

        Args:
            image: BGR 格式输入图像

        Returns:
            光照校正后的图像
        """
        # CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8)
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Gamma 校正
        if self.gamma != 1.0:
            inv_gamma = 1.0 / self.gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
            ]).astype(np.uint8)
            result = cv2.LUT(result, table)

        return result

    def optimize(self, image: np.ndarray) -> np.ndarray:
        """执行完整优化流水线

        按顺序执行：光照校正 → 颜色增强 → 边缘增强

        Args:
            image: BGR 格式输入图像

        Returns:
            优化后的图像
        """
        result = self.correct_lighting(image)
        result = self.enhance_colors(result)
        result = self.enhance_edges(result)
        return result

    def visualize_optimization(
        self,
        image: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """可视化优化效果

        展示原始图像和各优化步骤的结果对比。

        Args:
            image: BGR 格式输入图像
            save_path: 保存路径（可选）
        """
        corrected = self.correct_lighting(image)
        color_enhanced = self.enhance_colors(corrected)
        edge_enhanced = self.enhance_edges(color_enhanced)

        images = [image, corrected, color_enhanced, edge_enhanced]
        titles = ["原始图像", "光照校正", "颜色增强", "边缘增强"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for ax, img, title in zip(axes.flat, images, titles):
            ax.imshow(bgr_to_rgb(img))
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

        plt.close()
