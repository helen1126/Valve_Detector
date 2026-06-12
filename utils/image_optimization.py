"""图像优化模块

提供专门针对阀门角度盘的图像优化技术，包括颜色增强、
边缘检测、区域提取、显著性检测和智能裁剪等。
远距离拍摄时阀门在画面中占比小，通过智能裁剪可大幅提升预测精度。
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
    支持远距离拍摄场景的智能裁剪，自动定位阀门区域并放大预测。

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

    def _detect_by_color(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """基于颜色特征检测阀门区域

        利用绿色和红色掩码定位阀门，返回边界框。

        Args:
            image: BGR 格式输入图像

        Returns:
            (x, y, w, h) 边界框，未检测到返回 None
        """
        green_mask = extract_green_mask(image)
        red_mask = extract_red_mask(image)
        combined_mask = cv2.bitwise_or(green_mask, red_mask)

        # 形态学操作：先闭运算填充空洞，再开运算去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        max_contour = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(max_contour)

    def _detect_by_saliency(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """基于显著性检测定位阀门区域

        使用频谱残差显著性检测，适用于颜色特征不明显的场景。
        优先使用 opencv-contrib 的 saliency 模块，不可用时回退到
        基于 Laplacian 的简易显著性检测。

        Args:
            image: BGR 格式输入图像

        Returns:
            (x, y, w, h) 边界框，未检测到返回 None
        """
        saliency_map = None

        # 尝试使用 opencv-contrib 的频谱残差显著性检测
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency.computeSaliency(image)
            if not success or saliency_map is None:
                saliency_map = None
        except AttributeError:
            saliency_map = None

        # 回退：基于 Laplacian 的简易显著性检测
        if saliency_map is None:
            saliency_map = self._simple_saliency(image)

        if saliency_map is None:
            return None

        # 二值化显著性图
        saliency_map = (saliency_map * 255).astype(np.uint8) if saliency_map.max() <= 1.0 else saliency_map.astype(np.uint8)
        _, binary = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        max_contour = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(max_contour)

    @staticmethod
    def _simple_saliency(image: np.ndarray) -> Optional[np.ndarray]:
        """简易显著性检测（不依赖 opencv-contrib）

        基于 Laplacian + 高斯模糊的显著性估计：
        对图像做高斯模糊后与原图做差，再通过 Laplacian 增强高频细节。

        Args:
            image: BGR 格式输入图像

        Returns:
            显著性图 (0~1 浮点)，失败返回 None
        """
        try:
            # 缩小图像加速计算
            h, w = image.shape[:2]
            scale = 1.0
            if max(h, w) > 512:
                scale = 512.0 / max(h, w)
                small = cv2.resize(image, None, fx=scale, fy=scale)
            else:
                small = image

            # 高斯模糊与原图做差，提取高频信息
            blurred = cv2.GaussianBlur(small, (21, 21), 3)
            diff = cv2.absdiff(small, blurred)

            # 转灰度并归一化
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if gray.max() == 0:
                return None

            # Laplacian 增强边缘
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            saliency = np.abs(lap)

            # 归一化到 [0, 1]
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

            # 恢复原始尺寸
            if scale != 1.0:
                saliency = cv2.resize(saliency, (w, h))

            return saliency
        except Exception:
            return None

    def _detect_by_edge_density(self, image: np.ndarray, grid_size: int = 8) -> Optional[Tuple[int, int, int, int]]:
        """基于边缘密度检测阀门区域

        将图像划分为网格，计算每个网格的边缘密度，
        找到边缘密度最高的区域作为阀门候选区域。

        Args:
            image: BGR 格式输入图像
            grid_size: 网格划分大小

        Returns:
            (x, y, w, h) 边界框，未检测到返回 None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        h, w = edges.shape
        cell_h, cell_w = h // grid_size, w // grid_size

        if cell_h == 0 or cell_w == 0:
            return None

        # 计算每个网格的边缘密度
        density_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        for i in range(grid_size):
            for j in range(grid_size):
                cell = edges[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
                density_map[i, j] = np.mean(cell) / 255.0

        # 找到密度最高的 2x2 区域
        best_score = 0
        best_region = None
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                score = np.sum(density_map[i:i+2, j:j+2])
                if score > best_score:
                    best_score = score
                    best_region = (j * cell_w, i * cell_h, 2 * cell_w, 2 * cell_h)

        if best_region is None or best_score < 0.3:
            return None

        return best_region

    def detect_valve_region(
        self,
        image: np.ndarray,
        min_area_ratio: float = 0.02,
    ) -> Optional[Tuple[int, int, int, int]]:
        """多策略检测阀门区域

        按优先级依次尝试：颜色特征 → 显著性检测 → 边缘密度检测。
        返回置信度最高的阀门区域边界框。

        Args:
            image: BGR 格式输入图像
            min_area_ratio: 最小面积占图像面积的比例

        Returns:
            (x, y, w, h) 边界框，未检测到返回 None
        """
        img_area = image.shape[0] * image.shape[1]
        min_area = img_area * min_area_ratio

        # 策略1：颜色特征检测（最可靠）
        bbox = self._detect_by_color(image)
        if bbox is not None and bbox[2] * bbox[3] >= min_area:
            return bbox

        # 策略2：显著性检测
        bbox = self._detect_by_saliency(image)
        if bbox is not None and bbox[2] * bbox[3] >= min_area:
            return bbox

        # 策略3：边缘密度检测
        bbox = self._detect_by_edge_density(image)
        if bbox is not None and bbox[2] * bbox[3] >= min_area:
            return bbox

        return None

    def smart_crop(
        self,
        image: np.ndarray,
        target_size: int = 384,
        min_area_ratio: float = 0.15,
        padding_ratio: float = 0.25,
        min_padding: int = 20,
    ) -> Tuple[np.ndarray, bool]:
        """智能裁剪：远距离拍摄时自动定位并放大阀门区域

        当阀门在画面中占比较小时，自动裁剪阀门区域并放大到目标尺寸，
        使模型能获取更多阀门细节信息。

        Args:
            image: BGR 格式输入图像
            target_size: 目标图像尺寸
            min_area_ratio: 触发裁剪的最小面积比例阈值，
                            阀门占比低于此值时执行裁剪（默认0.15，即15%）
            padding_ratio: 裁剪区域向外扩展的比例
            min_padding: 最小扩展像素数

        Returns:
            (cropped_image, was_cropped) 元组：
            - cropped_image: 裁剪（或原始）图像
            - was_cropped: 是否执行了裁剪
        """
        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        # 检测阀门区域
        bbox = self.detect_valve_region(image, min_area_ratio=0.02)

        if bbox is None:
            # 未检测到阀门区域，返回原图
            return image, False

        x, y, w, h = bbox
        valve_area = w * h
        valve_ratio = valve_area / img_area

        # 阀门占比足够大，不需要裁剪
        if valve_ratio >= min_area_ratio:
            return image, False

        # 计算扩展后的裁剪区域
        padding = max(int(max(w, h) * padding_ratio), min_padding)

        crop_x1 = max(0, x - padding)
        crop_y1 = max(0, y - padding)
        crop_x2 = min(img_w, x + w + padding)
        crop_y2 = min(img_h, y + h + padding)

        # 保持正方形裁剪（模型输入为正方形）
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        # 以阀门中心为基准扩展为正方形
        center_x = (crop_x1 + crop_x2) // 2
        center_y = (crop_y1 + crop_y2) // 2
        crop_size = max(crop_w, crop_h)

        # 确保正方形区域在图像范围内
        half_size = crop_size // 2
        new_x1 = max(0, center_x - half_size)
        new_y1 = max(0, center_y - half_size)
        new_x2 = min(img_w, new_x1 + crop_size)
        new_y2 = min(img_h, new_y1 + crop_size)

        # 如果超出边界则调整起点
        if new_x2 - new_x1 < crop_size:
            new_x1 = max(0, new_x2 - crop_size)
        if new_y2 - new_y1 < crop_size:
            new_y1 = max(0, new_y2 - crop_size)

        cropped = image[new_y1:new_y2, new_x1:new_x2]

        return cropped, True

    def extract_dial_region(
        self, image: np.ndarray, min_area_ratio: float = 0.05
    ) -> np.ndarray:
        """区域提取：自动定位并裁剪角度盘区域

        基于多策略检测定位角度盘，裁剪并返回包含角度盘的区域。

        Args:
            image: BGR 格式输入图像
            min_area_ratio: 最小面积比例

        Returns:
            裁剪后的角度盘区域图像
        """
        bbox = self.detect_valve_region(image, min_area_ratio=min_area_ratio)

        if bbox is None:
            return image

        x, y, w, h = bbox
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
