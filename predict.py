"""预测脚本

提供单张/批量图片预测功能，支持模型热加载和 ONNX 推理。

使用示例：
    # 单张图片预测
    python predict.py --model_path ./weights/best.ckpt --input ./test.jpg

    # 批量预测
    python predict.py --model_path ./weights/best.ckpt --input ./test_images/ --output ./results/

    # 使用 ONNX 模型推理
    python predict.py --model_path ./weights/model.onnx --input ./test.jpg --onnx
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from models import build_model
from data.augmentation import get_val_transforms
from utils.image_utils import read_image, save_image, draw_angle_on_image
from utils.image_optimization import ImageOptimizer
from utils.logger import setup_logger, get_logger


class ValvePredictor:
    """阀门角度预测器

    封装模型加载、图像预处理和角度预测功能，
    支持单张/批量预测和模型热加载。

    Args:
        model_path: 模型权重路径
        model_name: 模型架构名称
        image_size: 图像尺寸
        angle_min: 最小角度
        angle_max: 最大角度
        device: 计算设备
        use_onnx: 是否使用 ONNX 模型推理
        use_optimization: 是否使用图像优化
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = "convnext_base",
        image_size: int = 384,
        angle_min: float = 0.0,
        angle_max: float = 80.0,
        device: Optional[str] = None,
        use_onnx: bool = False,
        use_optimization: bool = False,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.image_size = image_size
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.use_onnx = use_onnx
        self.use_optimization = use_optimization

        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 图像优化器
        self.optimizer = ImageOptimizer() if use_optimization else None

        # 预处理变换
        self.transform = get_val_transforms(image_size)

        # 加载模型
        self.model = None
        self.onnx_session = None
        self._load_model()

        # 记录模型加载时间（用于热加载检测）
        self._model_mtime = Path(model_path).stat().st_mtime if Path(model_path).exists() else 0

    def _load_model(self) -> None:
        """加载模型权重"""
        if self.use_onnx:
            self._load_onnx_model()
        else:
            self._load_pytorch_model()

    def _load_pytorch_model(self) -> None:
        """加载 PyTorch 模型"""
        model = build_model(model_name=self.model_name, pretrained=False)

        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )

        # 处理不同的权重格式
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    new_state_dict[key[6:]] = value
                else:
                    new_state_dict[key] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()
        self.model = model

    def _load_onnx_model(self) -> None:
        """加载 ONNX 模型"""
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.onnx_session = ort.InferenceSession(self.model_path, providers=providers)

    def check_and_reload(self) -> bool:
        """检查模型文件是否更新，支持热加载

        Returns:
            是否重新加载了模型
        """
        if not Path(self.model_path).exists():
            return False

        current_mtime = Path(self.model_path).stat().st_mtime
        if current_mtime > self._model_mtime:
            self._model_mtime = current_mtime
            self._load_model()
            return True

        return False

    @torch.no_grad()
    def predict_single(
        self,
        image: np.ndarray,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """单张图片预测

        Args:
            image: BGR 格式图像

        Returns:
            预测结果字典，包含预测角度和处理时间
        """
        start_time = time.time()

        # 图像优化
        if self.optimizer is not None:
            image = self.optimizer.optimize(image)

        # 预处理
        transformed = self.transform(image=image)
        input_tensor = torch.from_numpy(
            transformed["image"].transpose(2, 0, 1)
        ).float().unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(self.device)

        # 推理
        if self.use_onnx:
            output = self.onnx_session.run(
                None, {"input": input_tensor.cpu().numpy()}
            )[0]
            prediction = float(output[0][0])
        else:
            output = self.model(input_tensor)
            prediction = float(output.squeeze().cpu().numpy())

        # 反归一化
        angle = prediction * (self.angle_max - self.angle_min) + self.angle_min

        # 限制角度范围
        angle = max(self.angle_min, min(self.angle_max, angle))

        elapsed_time = time.time() - start_time

        return {
            "angle": round(angle, 1),
            "confidence": None,
            "time": round(elapsed_time, 4),
            "image": image,
        }

    def predict_image_path(self, image_path: str) -> Dict:
        """从文件路径预测单张图片

        Args:
            image_path: 图片文件路径

        Returns:
            预测结果字典
        """
        image = read_image(image_path)
        result = self.predict_single(image)
        result["image_path"] = image_path
        return result

    def predict_batch(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None,
    ) -> List[Dict]:
        """批量图片预测

        Args:
            image_paths: 图片文件路径列表
            output_dir: 结果输出目录（可选）

        Returns:
            预测结果列表
        """
        results = []

        for image_path in image_paths:
            try:
                result = self.predict_image_path(image_path)
                results.append(result)

                # 保存标注后的图像
                if output_dir is not None:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)

                    annotated = draw_angle_on_image(
                        result["image"], result["angle"]
                    )
                    save_name = f"pred_{Path(image_path).stem}_{result['angle']:.1f}.jpg"
                    save_image(annotated, str(output_path / save_name))

            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "angle": None,
                    "error": str(e),
                })

        return results

    def predict_folder(
        self,
        folder_path: str,
        output_dir: Optional[str] = None,
        extensions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """预测文件夹中的所有图片

        Args:
            folder_path: 文件夹路径
            output_dir: 结果输出目录
            extensions: 支持的图片格式

        Returns:
            预测结果 DataFrame
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        folder = Path(folder_path)
        image_paths = [
            str(p) for p in sorted(folder.iterdir())
            if p.suffix.lower() in extensions
        ]

        if not image_paths:
            raise ValueError(f"文件夹中未找到图片: {folder_path}")

        results = self.predict_batch(image_paths, output_dir)

        # 转换为 DataFrame
        df = pd.DataFrame([
            {
                "文件名": Path(r["image_path"]).name,
                "预测角度": r.get("angle"),
                "处理时间(秒)": r.get("time"),
                "错误": r.get("error"),
            }
            for r in results
        ])

        # 保存 CSV
        if output_dir is not None:
            csv_path = Path(output_dir) / "predictions.csv"
            df.to_csv(str(csv_path), index=False, encoding="utf-8-sig")

        return df


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="阀门角度检测预测")

    parser.add_argument(
        "--model_path", type=str, required=True,
        help="模型权重路径（.ckpt/.pth/.onnx）"
    )
    parser.add_argument(
        "--model_name", type=str, default="convnext_base",
        help="模型架构名称"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="输入图片路径或文件夹路径"
    )
    parser.add_argument(
        "--output", type=str, default="./output",
        help="输出目录"
    )
    parser.add_argument("--image_size", type=int, default=384, help="图像尺寸")
    parser.add_argument(
        "--onnx", action="store_true",
        help="使用 ONNX 模型推理"
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="启用图像优化"
    )
    parser.add_argument(
        "--config_dir", type=str, default="./config",
        help="配置文件目录"
    )

    return parser.parse_args()


def main():
    """预测主函数"""
    args = parse_args()

    # 初始化日志
    setup_logger(log_dir="./logs")
    logger = get_logger()

    # 加载配置
    config_path = os.path.join(args.config_dir, "data_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    # 创建预测器
    predictor = ValvePredictor(
        model_path=args.model_path,
        model_name=args.model_name,
        image_size=args.image_size,
        angle_min=data_config.get("angle_min", 0.0),
        angle_max=data_config.get("angle_max", 80.0),
        use_onnx=args.onnx,
        use_optimization=args.optimize,
    )

    input_path = Path(args.input)

    if input_path.is_file():
        # 单张图片预测
        result = predictor.predict_image_path(str(input_path))
        logger.info(f"预测角度: {result['angle']}°")
        logger.info(f"处理时间: {result['time']:.4f} 秒")

        # 保存标注后的图像
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        annotated = draw_angle_on_image(result["image"], result["angle"])
        save_name = f"pred_{input_path.stem}_{result['angle']:.1f}.jpg"
        save_image(annotated, str(output_dir / save_name))
        logger.info(f"结果已保存: {output_dir / save_name}")

    elif input_path.is_dir():
        # 批量预测
        df = predictor.predict_folder(str(input_path), args.output)
        logger.info(f"批量预测完成，共 {len(df)} 张图片")
        logger.info(f"结果已保存: {Path(args.output) / 'predictions.csv'}")

        # 打印统计信息
        valid_df = df.dropna(subset=["预测角度"])
        if len(valid_df) > 0:
            logger.info(f"平均预测角度: {valid_df['预测角度'].mean():.1f}°")
            logger.info(f"预测角度范围: {valid_df['预测角度'].min():.1f}° - {valid_df['预测角度'].max():.1f}°")

    else:
        logger.error(f"输入路径不存在: {args.input}")


if __name__ == "__main__":
    main()
