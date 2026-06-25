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
    支持单张/批量预测、模型热加载、智能裁剪和多尺度推理。

    Args:
        model_path: 模型权重路径
        model_name: 模型架构名称
        image_size: 图像尺寸
        angle_min: 最小角度
        angle_max: 最大角度
        device: 计算设备
        use_onnx: 是否使用 ONNX 模型推理
        use_optimization: 是否使用图像优化
        smart_crop: 是否启用智能裁剪（远距离拍摄时自动定位并放大阀门区域）
        multi_scale: 是否启用多尺度推理（结合原图和裁剪图预测）
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
        smart_crop: bool = False,
        multi_scale: bool = False,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.image_size = image_size
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.use_onnx = use_onnx
        self.use_optimization = use_optimization
        self.smart_crop = smart_crop
        self.multi_scale = multi_scale

        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 图像优化器（智能裁剪和多尺度推理也需要）
        self.optimizer = ImageOptimizer() if (use_optimization or smart_crop or multi_scale) else None

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
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )

        # 从 checkpoint 中自动检测模型名称
        model_name = self.model_name
        if isinstance(checkpoint, dict) and "hyper_parameters" in checkpoint:
            ckpt_model_name = checkpoint["hyper_parameters"].get("model_name")
            if ckpt_model_name and ckpt_model_name != self.model_name:
                self.model_name = ckpt_model_name
                model_name = ckpt_model_name

        model = build_model(model_name=model_name, pretrained=False)

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

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """图像预处理：变换并转为模型输入张量

        Args:
            image: BGR 格式图像

        Returns:
            模型输入张量 (1, 3, H, W)
        """
        transformed = self.transform(image=image)
        input_tensor = torch.from_numpy(
            transformed["image"].transpose(2, 0, 1)
        ).float().unsqueeze(0) / 255.0
        return input_tensor.to(self.device)

    def _inference(self, input_tensor: torch.Tensor) -> float:
        """执行模型推理

        Args:
            input_tensor: 模型输入张量

        Returns:
            归一化角度预测值 [0, 1]
        """
        if self.use_onnx:
            output = self.onnx_session.run(
                None, {"input": input_tensor.cpu().numpy()}
            )[0]
            return float(output[0][0])
        else:
            output = self.model(input_tensor)
            return float(output.squeeze().cpu().numpy())

    def _denormalize_angle(self, prediction: float) -> float:
        """反归一化角度预测值

        Args:
            prediction: 归一化角度预测值 [0, 1]

        Returns:
            实际角度值
        """
        angle = prediction * (self.angle_max - self.angle_min) + self.angle_min
        return max(self.angle_min, min(self.angle_max, angle))

    @torch.no_grad()
    def predict_single(
        self,
        image: np.ndarray,
    ) -> Dict[str, Union[float, np.ndarray, bool]]:
        """单张图片预测

        支持三种模式：
        - 普通模式：直接对原图预测
        - 智能裁剪模式：检测阀门区域并裁剪放大后预测
        - 多尺度模式：同时用原图和裁剪图预测，加权融合

        Args:
            image: BGR 格式图像

        Returns:
            预测结果字典，包含预测角度、处理时间和是否裁剪
        """
        start_time = time.time()
        was_cropped = False

        # 图像优化（颜色/边缘增强）
        if self.use_optimization and self.optimizer is not None:
            image = self.optimizer.optimize(image)

        if self.multi_scale and self.optimizer is not None:
            # 多尺度推理：融合原图和裁剪图的预测结果
            angle = self._predict_multi_scale(image)
        elif self.smart_crop and self.optimizer is not None:
            # 智能裁剪：远距离时自动裁剪阀门区域
            cropped_img, was_cropped = self.optimizer.smart_crop(
                image, target_size=self.image_size
            )
            input_tensor = self._preprocess(cropped_img)
            prediction = self._inference(input_tensor)
            angle = self._denormalize_angle(prediction)
        else:
            # 普通模式
            input_tensor = self._preprocess(image)
            prediction = self._inference(input_tensor)
            angle = self._denormalize_angle(prediction)

        elapsed_time = time.time() - start_time

        return {
            "angle": round(angle, 1),
            "confidence": None,
            "time": round(elapsed_time, 4),
            "image": image,
            "cropped": was_cropped,
        }

    def _predict_multi_scale(self, image: np.ndarray) -> float:
        """多尺度推理：融合原图和裁剪图的预测结果

        当阀门在画面中占比较小时，裁剪图预测更准确；
        占比较大时，原图预测更稳定。根据阀门占比自适应加权。

        Args:
            image: BGR 格式输入图像

        Returns:
            融合后的角度值
        """
        # 原图预测
        input_tensor = self._preprocess(image)
        pred_original = self._inference(input_tensor)

        # 裁剪图预测
        cropped_img, was_cropped = self.optimizer.smart_crop(
            image, target_size=self.image_size
        )

        if not was_cropped:
            # 阀门占比足够大，直接用原图结果
            return self._denormalize_angle(pred_original)

        input_tensor_cropped = self._preprocess(cropped_img)
        pred_cropped = self._inference(input_tensor_cropped)

        # 计算阀门占比，据此确定权重
        bbox = self.optimizer.detect_valve_region(image, min_area_ratio=0.02)
        if bbox is not None:
            valve_ratio = (bbox[2] * bbox[3]) / (image.shape[0] * image.shape[1])
            # 阀门占比越小，裁剪图权重越高
            # 占比 2% → 裁剪权重 0.9，占比 15% → 裁剪权重 0.5
            crop_weight = min(0.9, max(0.5, 1.0 - valve_ratio * 3.3))
        else:
            crop_weight = 0.7

        # 加权融合
        fused_pred = pred_original * (1 - crop_weight) + pred_cropped * crop_weight
        return self._denormalize_angle(fused_pred)

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

    def predict_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        fps: Optional[float] = None,
        frame_interval: Optional[int] = None,
        save_frames: bool = False,
        save_video: bool = False,
    ) -> pd.DataFrame:
        """视频抽帧预测

        按指定频率从视频中抽帧并预测阀门角度。支持两种抽帧方式：
        - fps：每秒抽取多少帧（如 2.0 = 每秒 2 帧）
        - frame_interval：每隔多少帧抽取一帧（如 30 = 每 30 帧抽 1 帧）

        Args:
            video_path: 视频文件路径
            output_dir: 结果输出目录（可选）
            fps: 每秒抽帧数，与 frame_interval 二选一
            frame_interval: 帧间隔，与 fps 二选一
            save_frames: 是否保存标注帧为图片
            save_video: 是否输出带角度标注的视频

        Returns:
            预测结果 DataFrame，包含帧索引、时间戳、预测角度、处理时间
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 25.0  # 默认假设 25fps

        # 计算抽帧间隔
        if fps is not None and fps > 0:
            interval = max(1, int(video_fps / fps))
        elif frame_interval is not None and frame_interval > 0:
            interval = frame_interval
        else:
            # 默认每秒 1 帧
            interval = max(1, int(video_fps))

        # 准备输出
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = None

        # 视频写入器（可选）
        video_writer = None
        if save_video and output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_out_path = str(output_path / "predicted_video.mp4")
            video_writer = cv2.VideoWriter(video_out_path, fourcc, video_fps, (frame_w, frame_h))

        results = []
        frame_idx = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                # 预测
                result = self.predict_single(frame)
                angle = result["angle"]
                timestamp = frame_idx / video_fps

                results.append({
                    "帧索引": frame_idx,
                    "时间戳(秒)": round(timestamp, 2),
                    "预测角度": angle,
                    "处理时间(秒)": result["time"],
                })

                # 保存标注帧
                if save_frames and output_path is not None:
                    annotated = draw_angle_on_image(frame, angle)
                    frame_name = f"frame_{frame_idx:06d}_{angle:.1f}.jpg"
                    save_image(annotated, str(output_path / frame_name))

                # 写入标注视频
                if video_writer is not None:
                    annotated = draw_angle_on_image(frame, angle)
                    video_writer.write(annotated)

                processed_count += 1

            frame_idx += 1

        cap.release()
        if video_writer is not None:
            video_writer.release()

        # 转换为 DataFrame
        df = pd.DataFrame(results)

        # 保存 CSV
        if output_path is not None and len(df) > 0:
            csv_path = output_path / "video_predictions.csv"
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
        "--smart_crop", action="store_true",
        help="启用智能裁剪（远距离拍摄时自动定位并放大阀门区域）"
    )
    parser.add_argument(
        "--multi_scale", action="store_true",
        help="启用多尺度推理（结合原图和裁剪图预测，精度更高）"
    )
    parser.add_argument(
        "--config_dir", type=str, default="./config",
        help="配置文件目录"
    )

    # 视频抽帧参数
    parser.add_argument(
        "--fps", type=float, default=None,
        help="视频抽帧频率：每秒抽取多少帧（与 --frame_interval 二选一）"
    )
    parser.add_argument(
        "--frame_interval", type=int, default=None,
        help="视频抽帧间隔：每隔多少帧抽取一帧（与 --fps 二选一）"
    )
    parser.add_argument(
        "--save_frames", action="store_true",
        help="视频预测时保存标注帧为图片"
    )
    parser.add_argument(
        "--save_video", action="store_true",
        help="视频预测时输出带角度标注的视频"
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
        smart_crop=args.smart_crop,
        multi_scale=args.multi_scale,
    )

    input_path = Path(args.input)

    # 视频文件后缀
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

    if input_path.is_file() and input_path.suffix.lower() in video_extensions:
        # 视频抽帧预测
        logger.info(f"视频抽帧预测: {input_path}")
        if args.fps is not None:
            logger.info(f"抽帧频率: {args.fps} fps")
        elif args.frame_interval is not None:
            logger.info(f"抽帧间隔: 每 {args.frame_interval} 帧")
        else:
            logger.info("抽帧频率: 默认每秒 1 帧")

        df = predictor.predict_video(
            video_path=str(input_path),
            output_dir=args.output,
            fps=args.fps,
            frame_interval=args.frame_interval,
            save_frames=args.save_frames,
            save_video=args.save_video,
        )

        logger.info(f"视频预测完成，共处理 {len(df)} 帧")
        if len(df) > 0:
            logger.info(f"结果已保存: {Path(args.output) / 'video_predictions.csv'}")
            logger.info(f"平均预测角度: {df['预测角度'].mean():.1f}°")
            logger.info(f"预测角度范围: {df['预测角度'].min():.1f}° - {df['预测角度'].max():.1f}°")
            if args.save_video:
                logger.info(f"标注视频已保存: {Path(args.output) / 'predicted_video.mp4'}")

    elif input_path.is_file():
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
