"""图像工具模块

提供图像读取、保存、颜色空间转换和可视化等工具函数。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Union

# 设置 matplotlib 中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def read_image(image_path: Union[str, Path]) -> np.ndarray:
    """读取图像（支持中文路径）

    Args:
        image_path: 图像文件路径

    Returns:
        BGR 格式的图像数组

    Raises:
        FileNotFoundError: 图像文件不存在
        ValueError: 图像读取失败
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 使用 numpy 读取以支持中文路径
    image = cv2.imdecode(
        np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(f"图像读取失败: {image_path}")

    return image


def save_image(
    image: np.ndarray,
    save_path: Union[str, Path],
) -> None:
    """保存图像（支持中文路径）

    Args:
        image: BGR 格式的图像数组
        save_path: 保存路径

    Raises:
        ValueError: 图像保存失败
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用 imencode 以支持中文路径
    ext = save_path.suffix
    success, encoded = cv2.imencode(ext, image)

    if not success:
        raise ValueError(f"图像编码失败: {save_path}")

    encoded.tofile(str(save_path))


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR 转 RGB

    Args:
        image: BGR 格式图像

    Returns:
        RGB 格式图像
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """RGB 转 BGR

    Args:
        image: RGB 格式图像

    Returns:
        BGR 格式图像
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    """BGR 转 HSV

    Args:
        image: BGR 格式图像

    Returns:
        HSV 格式图像
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def hsv_to_bgr(image: np.ndarray) -> np.ndarray:
    """HSV 转 BGR

    Args:
        image: HSV 格式图像

    Returns:
        BGR 格式图像
    """
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


def extract_green_mask(
    image: np.ndarray,
    h_range: Tuple[int, int] = (35, 85),
    s_range: Tuple[int, int] = (50, 255),
    v_range: Tuple[int, int] = (50, 255),
) -> np.ndarray:
    """提取图像中的绿色区域掩码

    Args:
        image: BGR 格式图像
        h_range: HSV 中 H 通道的范围
        s_range: HSV 中 S 通道的范围
        v_range: HSV 中 V 通道的范围

    Returns:
        二值掩码图像
    """
    hsv = bgr_to_hsv(image)
    lower = np.array([h_range[0], s_range[0], v_range[0]])
    upper = np.array([h_range[1], s_range[1], v_range[1]])
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def extract_red_mask(
    image: np.ndarray,
    h_range1: Tuple[int, int] = (0, 10),
    h_range2: Tuple[int, int] = (170, 180),
    s_range: Tuple[int, int] = (50, 255),
    v_range: Tuple[int, int] = (50, 255),
) -> np.ndarray:
    """提取图像中的红色区域掩码

    红色在 HSV 空间中跨越 0° 和 180° 两个区域，需要分别提取后合并。

    Args:
        image: BGR 格式图像
        h_range1: HSV 中 H 通道低值范围
        h_range2: HSV 中 H 通道高值范围
        s_range: HSV 中 S 通道的范围
        v_range: HSV 中 V 通道的范围

    Returns:
        二值掩码图像
    """
    hsv = bgr_to_hsv(image)

    # 红色低值区域
    lower1 = np.array([h_range1[0], s_range[0], v_range[0]])
    upper1 = np.array([h_range1[1], s_range[1], v_range[1]])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # 红色高值区域
    lower2 = np.array([h_range2[0], s_range[0], v_range[0]])
    upper2 = np.array([h_range2[1], s_range[1], v_range[1]])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    return cv2.bitwise_or(mask1, mask2)


def draw_angle_on_image(
    image: np.ndarray,
    angle: float,
    position: Optional[Tuple[int, int]] = None,
    font_scale: float = 1.5,
    thickness: int = 3,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """在图像上绘制角度标注

    Args:
        image: BGR 格式图像
        angle: 角度值
        position: 标注位置（左上角坐标），默认为图像左上角
        font_scale: 字体大小
        thickness: 字体粗细
        color: 字体颜色（BGR 格式）

    Returns:
        标注后的图像
    """
    result = image.copy()
    text = f"{angle:.1f}"

    if position is None:
        position = (20, 50)

    # 添加背景矩形
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.rectangle(
        result,
        (position[0] - 5, position[1] - text_h - 5),
        (position[0] + text_w + 5, position[1] + baseline + 5),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        result,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return result


def visualize_samples(
    images: list,
    titles: Optional[list] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """可视化多个图像样本

    Args:
        images: 图像列表（BGR 格式）
        titles: 标题列表
        figsize: 图像大小
        save_path: 保存路径（可选）
    """
    n = len(images)
    if titles is None:
        titles = [f"样本 {i+1}" for i in range(n)]

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        rgb_img = bgr_to_rgb(img)
        ax.imshow(rgb_img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

    plt.close()


def visualize_augmentation(
    original: np.ndarray,
    augmented: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """可视化数据增强效果

    Args:
        original: 原始图像（BGR 格式）
        augmented: 增强后的图像（BGR 格式）
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(bgr_to_rgb(original))
    axes[0].set_title("原始图像")
    axes[0].axis("off")

    axes[1].imshow(bgr_to_rgb(augmented))
    axes[1].set_title("增强后图像")
    axes[1].axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

    plt.close()
