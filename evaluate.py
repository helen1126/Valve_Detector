"""模型评估脚本

在测试集上评估训练好的模型，计算各项指标并生成可视化报告。

使用示例：
    python evaluate.py --model_path ./weights/last.ckpt --data_dir ./dataset
    python evaluate.py --model_path ./weights/best.ckpt --view top_view
"""

import argparse
import os
from pathlib import Path

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_model
from data.dataset import ValveDataset
from data.augmentation import get_val_transforms
from utils.metrics import compute_all_metrics
from utils.logger import setup_logger, get_logger
from utils.image_utils import draw_angle_on_image, save_image

# 设置 matplotlib 中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    angle_min: float = 0.0,
    angle_max: float = 80.0,
    device: torch.device = None,
) -> dict:
    """在数据集上评估模型

    Args:
        model: 模型实例
        dataloader: 数据加载器
        angle_min: 最小角度
        angle_max: 最大角度
        device: 计算设备

    Returns:
        评估结果字典
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []
    all_filenames = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            images = batch["image"].to(device)
            angles = batch["angle"]
            raw_angles = batch["raw_angle"]
            filenames = batch["filename"]

            # 前向传播
            outputs = model(images)

            # 反归一化
            predictions = outputs.squeeze().cpu().numpy() * (angle_max - angle_min) + angle_min
            targets = raw_angles.numpy()

            # 确保是一维数组
            if predictions.ndim == 0:
                predictions = np.array([predictions])
            if targets.ndim == 0:
                targets = np.array([targets])

            all_predictions.extend(predictions.tolist())
            all_targets.extend(targets.tolist())
            all_filenames.extend(list(filenames))

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # 计算所有指标
    metrics = compute_all_metrics(
        predictions=all_predictions,
        targets=all_targets,
        filenames=all_filenames,
        top_k=10,
    )

    # 添加原始数据
    metrics["predictions"] = all_predictions
    metrics["targets"] = all_targets
    metrics["filenames"] = all_filenames

    return metrics


def plot_scatter(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str,
    angle_max: float = 80.0,
) -> None:
    """绘制预测值与真实值对比散点图

    Args:
        predictions: 预测值数组
        targets: 真实值数组
        save_path: 保存路径
        angle_max: 最大角度
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(targets, predictions, alpha=0.5, s=20, c="steelblue")

    # 绘制理想线
    ax.plot([0, angle_max], [0, angle_max], "r--", linewidth=2, label="理想预测线")

    # 绘制 ±1° 误差带
    ax.fill_between(
        [0, angle_max],
        [0 - 1, angle_max - 1],
        [0 + 1, angle_max + 1],
        alpha=0.2,
        color="green",
        label="±1° 误差带",
    )

    ax.set_xlabel("真实角度 (°)", fontsize=12)
    ax.set_ylabel("预测角度 (°)", fontsize=12)
    ax.set_title("预测值 vs 真实值", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(-2, angle_max + 2)
    ax.set_ylim(-2, angle_max + 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str,
) -> None:
    """绘制误差分布直方图

    Args:
        predictions: 预测值数组
        targets: 真实值数组
        save_path: 保存路径
    """
    errors = predictions - targets

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 误差分布直方图
    axes[0].hist(errors, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].axvline(x=0, color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("预测误差 (°)", fontsize=12)
    axes[0].set_ylabel("样本数量", fontsize=12)
    axes[0].set_title("误差分布", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # 绝对误差分布
    abs_errors = np.abs(errors)
    axes[1].hist(abs_errors, bins=50, edgecolor="black", alpha=0.7, color="coral")
    axes[1].axvline(x=1.0, color="green", linestyle="--", linewidth=2, label="1° 阈值")
    axes[1].set_xlabel("绝对误差 (°)", fontsize=12)
    axes[1].set_ylabel("样本数量", fontsize=12)
    axes[1].set_title("绝对误差分布", fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_interval_errors(
    metrics: dict,
    save_path: str,
) -> None:
    """绘制不同角度区间的误差柱状图

    Args:
        metrics: 评估指标字典
        save_path: 保存路径
    """
    interval_metrics = metrics.get("interval_metrics", {})
    if not interval_metrics:
        return

    intervals = list(interval_metrics.keys())
    maes = [interval_metrics[k]["mae"] for k in intervals]
    counts = [interval_metrics[k]["count"] for k in intervals]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = range(len(intervals))
    bars = ax1.bar(x, maes, color="steelblue", alpha=0.7, label="MAE")
    ax1.set_xlabel("角度区间 (°)", fontsize=12)
    ax1.set_ylabel("MAE (°)", fontsize=12, color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(intervals, rotation=45)
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # 在柱状图上添加数值标签
    for bar, mae in zip(bars, maes):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mae:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 右侧 Y 轴：样本数量
    ax2 = ax1.twinx()
    ax2.plot(x, counts, "ro-", linewidth=2, markersize=6, label="样本数")
    ax2.set_ylabel("样本数量", fontsize=12, color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.set_title("各角度区间误差分析", fontsize=14)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_report(metrics: dict, save_dir: str) -> str:
    """生成评估报告

    Args:
        metrics: 评估指标字典
        save_dir: 报告保存目录

    Returns:
        报告文件路径
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    report_path = save_dir / "evaluation_report.txt"

    lines = []
    lines.append("=" * 60)
    lines.append("阀门角度检测模型评估报告")
    lines.append("=" * 60)
    lines.append("")

    # 基础指标
    lines.append("【基础指标】")
    lines.append(f"  MAE (平均绝对误差): {metrics['mae']:.4f}°")
    lines.append(f"  MSE (均方误差):     {metrics['mse']:.4f}")
    lines.append(f"  RMSE (均方根误差):  {metrics['rmse']:.4f}°")
    lines.append(f"  R² (决定系数):      {metrics['r2']:.4f}")
    lines.append(f"  最大误差:           {metrics['max_error']:.4f}°")
    lines.append(f"  平均误差:           {metrics['mean_error']:.4f}°")
    lines.append(f"  误差标准差:         {metrics['std_error']:.4f}°")
    lines.append("")

    # 误差分布
    lines.append("【误差分布】")
    for key, value in metrics["error_distribution"].items():
        lines.append(f"  误差 ≤ {key.split('_')[-1]}°: {value * 100:.2f}%")
    lines.append("")

    # 各角度区间指标
    lines.append("【各角度区间指标】")
    for interval, values in metrics["interval_metrics"].items():
        lines.append(
            f"  {interval}°: 样本数={values['count']}, "
            f"MAE={values['mae']:.4f}°, "
            f"RMSE={values['rmse']:.4f}°, "
            f"最大误差={values['max_error']:.4f}°"
        )
    lines.append("")

    # 误差最大的样本
    if "worst_samples" in metrics:
        lines.append("【误差最大的 10 个样本】")
        lines.append(f"  {'文件名':<25} {'真实角度':>8} {'预测角度':>8} {'绝对误差':>8}")
        lines.append("  " + "-" * 55)
        for filename, true_angle, pred_angle, error in metrics["worst_samples"]:
            lines.append(
                f"  {filename:<25} {true_angle:>8.1f} {pred_angle:>8.1f} {error:>8.2f}"
            )
    lines.append("")
    lines.append("=" * 60)

    report_text = "\n".join(lines)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return str(report_path)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="阀门角度检测模型评估")

    parser.add_argument(
        "--model_path", type=str, required=True,
        help="模型权重路径（.ckpt 或 .pth）"
    )
    parser.add_argument(
        "--model_name", type=str, default="convnext_base",
        help="模型架构名称"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./dataset",
        help="数据集根目录"
    )
    parser.add_argument(
        "--view", type=str, default="all_view",
        choices=["all_view", "top_view", "side_view"],
        help="视角选择"
    )
    parser.add_argument("--image_size", type=int, default=384, help="图像尺寸")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument(
        "--output_dir", type=str, default="./logs/evaluation",
        help="评估结果输出目录"
    )
    parser.add_argument(
        "--config_dir", type=str, default="./config",
        help="配置文件目录"
    )

    return parser.parse_args()


def main():
    """评估主函数"""
    args = parse_args()

    # 初始化日志
    setup_logger(log_dir="./logs")
    logger = get_logger()

    # 加载配置
    data_config = load_config(os.path.join(args.config_dir, "data_config.yaml"))

    # 创建测试数据集
    val_transforms = get_val_transforms(args.image_size)
    dataset = ValveDataset(
        data_dir=args.data_dir,
        view=args.view,
        image_size=args.image_size,
        transform=val_transforms,
        angle_min=data_config.get("angle_min", 0.0),
        angle_max=data_config.get("angle_max", 80.0),
    )

    # 使用全部数据作为测试集（实际使用时应该只使用测试集划分）
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name=args.model_name, pretrained=False)

    # 加载权重
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    # 处理不同的权重格式
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # 移除 "model." 前缀（PyTorch Lightning 保存的权重可能带有前缀）
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_state_dict[key[6:]] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)

    logger.info(f"模型权重已加载: {args.model_path}")

    # 评估
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        angle_min=data_config.get("angle_min", 0.0),
        angle_max=data_config.get("angle_max", 80.0),
        device=device,
    )

    # 输出关键指标
    logger.info(f"MAE: {metrics['mae']:.4f}°")
    logger.info(f"RMSE: {metrics['rmse']:.4f}°")
    logger.info(f"R²: {metrics['r2']:.4f}")

    # 生成可视化
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = metrics["predictions"]
    targets = metrics["targets"]

    plot_scatter(predictions, targets, str(output_dir / "scatter.png"))
    plot_error_distribution(predictions, targets, str(output_dir / "error_distribution.png"))
    plot_interval_errors(metrics, str(output_dir / "interval_errors.png"))

    # 生成报告
    report_path = generate_report(metrics, str(output_dir))
    logger.info(f"评估报告已保存: {report_path}")

    # 判断是否达标
    if metrics["mae"] <= 1.0:
        logger.info("评估结果：MAE ≤ 1°，精度达标！")
    else:
        logger.warning(f"评估结果：MAE = {metrics['mae']:.4f}°，未达到 MAE ≤ 1° 的目标")


if __name__ == "__main__":
    main()
