"""评估指标模块

提供阀门角度检测任务的各种评估指标计算功能。
"""

import numpy as np
from typing import Dict, List, Tuple


def calc_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算平均绝对误差（MAE）

    Args:
        predictions: 预测值数组
        targets: 真实值数组

    Returns:
        MAE 值
    """
    return float(np.mean(np.abs(predictions - targets)))


def calc_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算均方误差（MSE）

    Args:
        predictions: 预测值数组
        targets: 真实值数组

    Returns:
        MSE 值
    """
    return float(np.mean((predictions - targets) ** 2))


def calc_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算均方根误差（RMSE）

    Args:
        predictions: 预测值数组
        targets: 真实值数组

    Returns:
        RMSE 值
    """
    return float(np.sqrt(calc_mse(predictions, targets)))


def calc_r2(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算决定系数（R²）

    Args:
        predictions: 预测值数组
        targets: 真实值数组

    Returns:
        R² 值
    """
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def calc_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    thresholds: List[float] = None,
) -> Dict[str, float]:
    """计算误差分布统计

    Args:
        predictions: 预测值数组
        targets: 真实值数组
        thresholds: 误差阈值列表，默认为 [1.0, 2.0, 5.0]

    Returns:
        包含各误差区间比例的字典
    """
    if thresholds is None:
        thresholds = [1.0, 2.0, 5.0]

    errors = np.abs(predictions - targets)
    result = {}

    for threshold in thresholds:
        ratio = float(np.mean(errors <= threshold))
        key = f"error_le_{threshold:.1f}"
        result[key] = ratio

    return result


def calc_interval_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    interval: float = 10.0,
    angle_min: float = 0.0,
    angle_max: float = 80.0,
) -> Dict[str, Dict[str, float]]:
    """计算不同角度区间的误差指标

    Args:
        predictions: 预测值数组
        targets: 真实值数组
        interval: 角度区间大小
        angle_min: 最小角度
        angle_max: 最大角度

    Returns:
        各角度区间的误差指标字典
    """
    result = {}
    start = angle_min

    while start < angle_max:
        end = min(start + interval, angle_max)
        mask = (targets >= start) & (targets < end)

        if mask.sum() == 0:
            start = end
            continue

        interval_preds = predictions[mask]
        interval_targets = targets[mask]

        key = f"{start:.0f}-{end:.0f}"
        result[key] = {
            "count": int(mask.sum()),
            "mae": calc_mae(interval_preds, interval_targets),
            "rmse": calc_rmse(interval_preds, interval_targets),
            "max_error": float(np.max(np.abs(interval_preds - interval_targets))),
        }

        start = end

    return result


def find_worst_samples(
    predictions: np.ndarray,
    targets: np.ndarray,
    filenames: List[str],
    top_k: int = 10,
) -> List[Tuple[str, float, float, float]]:
    """找出误差最大的样本

    Args:
        predictions: 预测值数组
        targets: 真实值数组
        filenames: 文件名列表
        top_k: 返回的最大误差样本数

    Returns:
        误差最大的样本列表，每项为 (文件名, 真实角度, 预测角度, 绝对误差)
    """
    errors = np.abs(predictions - targets)
    worst_indices = np.argsort(errors)[::-1][:top_k]

    results = []
    for idx in worst_indices:
        results.append((
            filenames[idx],
            float(targets[idx]),
            float(predictions[idx]),
            float(errors[idx]),
        ))

    return results


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    filenames: List[str] = None,
    top_k: int = 10,
) -> Dict:
    """计算所有评估指标

    Args:
        predictions: 预测值数组
        targets: 真实值数组
        filenames: 文件名列表（用于找出误差最大的样本）
        top_k: 误差最大样本数

    Returns:
        包含所有指标的字典
    """
    result = {
        "mae": calc_mae(predictions, targets),
        "mse": calc_mse(predictions, targets),
        "rmse": calc_rmse(predictions, targets),
        "r2": calc_r2(predictions, targets),
        "max_error": float(np.max(np.abs(predictions - targets))),
        "mean_error": float(np.mean(predictions - targets)),
        "std_error": float(np.std(predictions - targets)),
        "error_distribution": calc_error_distribution(predictions, targets),
        "interval_metrics": calc_interval_metrics(predictions, targets),
    }

    if filenames is not None:
        result["worst_samples"] = find_worst_samples(
            predictions, targets, filenames, top_k
        )

    return result
