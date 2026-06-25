"""FastAPI 应用

阀门角度检测 RESTful API 接口，提供单张/批量预测、健康检查和模型信息查询。

启动方式：
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

接口文档：
    Swagger UI: http://localhost:8000/docs
    ReDoc:      http://localhost:8000/redoc
"""

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from predict import ValvePredictor
from utils.image_utils import draw_angle_on_image
from utils.logger import setup_logger, get_logger

# 初始化日志
setup_logger(log_dir="./logs")
logger = get_logger()

# 配置
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "./weights/last.ckpt")
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "convnext_base")
DEFAULT_IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "384"))
DEFAULT_SMART_CROP = os.environ.get("SMART_CROP", "true").lower() in ("true", "1", "yes")
DEFAULT_MULTI_SCALE = os.environ.get("MULTI_SCALE", "false").lower() in ("true", "1", "yes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时加载模型，关闭时清理资源"""
    global predictor

    config_path = "./config/data_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data_config = yaml.safe_load(f)
    else:
        data_config = {}

    try:
        # 从 checkpoint 自动检测模型名称
        model_name = DEFAULT_MODEL_NAME
        if os.path.exists(DEFAULT_MODEL_PATH) and DEFAULT_MODEL_PATH.endswith((".ckpt", ".pth")):
            import torch as _torch
            try:
                _ckpt = _torch.load(DEFAULT_MODEL_PATH, map_location="cpu", weights_only=False)
                if isinstance(_ckpt, dict) and "hyper_parameters" in _ckpt:
                    _ckpt_name = _ckpt["hyper_parameters"].get("model_name")
                    if _ckpt_name:
                        model_name = _ckpt_name
                del _ckpt
            except Exception:
                pass

        predictor = ValvePredictor(
            model_path=DEFAULT_MODEL_PATH,
            model_name=model_name,
            image_size=DEFAULT_IMAGE_SIZE,
            angle_min=data_config.get("angle_min", 0.0),
            angle_max=data_config.get("angle_max", 80.0),
            use_optimization=False,
            smart_crop=DEFAULT_SMART_CROP,
            multi_scale=DEFAULT_MULTI_SCALE,
        )
        logger.info(f"模型加载成功: {DEFAULT_MODEL_PATH} (模型: {model_name})")
    except Exception as e:
        logger.warning(f"模型加载失败: {e}，API 将在首次请求时尝试加载")
        predictor = None

    yield

    # 清理资源
    if predictor is not None:
        logger.info("模型资源已释放")


# 创建 FastAPI 应用
app = FastAPI(
    title="阀门角度检测 API",
    description="工业阀门角度检测系统 RESTful API 接口",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局预测器实例
predictor: Optional[ValvePredictor] = None


class PredictResponse(BaseModel):
    """单张图片预测响应"""
    angle: float
    time: float
    cropped: bool = False
    image: Optional[str] = None


class BatchPredictItem(BaseModel):
    """批量预测单项结果"""
    filename: str
    angle: Optional[float] = None
    time: float
    cropped: bool = False
    error: Optional[str] = None


class BatchPredictResponse(BaseModel):
    """批量预测响应"""
    results: List[BatchPredictItem]
    total_time: float


class VideoFrameResult(BaseModel):
    """视频抽帧预测单项结果"""
    frame_idx: int
    timestamp: float
    angle: float
    time: float


class VideoPredictResponse(BaseModel):
    """视频抽帧预测响应"""
    total_frames: int
    processed_frames: int
    total_time: float
    results: List[VideoFrameResult]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool


class InfoResponse(BaseModel):
    """模型信息响应"""
    model_name: str
    model_path: str
    image_size: int
    angle_range: str
    device: str
    optimization_enabled: bool
    smart_crop_enabled: bool
    multi_scale_enabled: bool


def _ensure_predictor():
    """确保预测器已初始化"""
    global predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="模型未加载，请检查模型权重文件")
    return predictor


def _decode_image(file_bytes: bytes) -> np.ndarray:
    """解码上传的图片文件

    Args:
        file_bytes: 图片文件的字节数据

    Returns:
        BGR 格式的图像数组
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="图片解码失败，请检查文件格式")
    return image


def _encode_image(image: np.ndarray) -> str:
    """将图像编码为 base64 字符串

    Args:
        image: BGR 格式图像

    Returns:
        base64 编码的图片字符串
    """
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


@app.post("/predict", response_model=PredictResponse, summary="单张图片预测")
async def predict(
    file: UploadFile = File(..., description="阀门图片文件"),
    return_image: bool = True,
    smart_crop: Optional[bool] = None,
    multi_scale: Optional[bool] = None,
):
    """上传单张阀门图片，返回预测角度

    Args:
        file: 上传的图片文件
        return_image: 是否返回标注后的图片（base64 编码）
        smart_crop: 是否启用智能裁剪（None 使用默认配置）
        multi_scale: 是否启用多尺度推理（None 使用默认配置）

    Returns:
        预测角度和处理时间
    """
    pred = _ensure_predictor()

    # 临时覆盖预测模式
    original_smart_crop = pred.smart_crop
    original_multi_scale = pred.multi_scale
    if smart_crop is not None:
        pred.smart_crop = smart_crop
    if multi_scale is not None:
        pred.multi_scale = multi_scale
    # 多尺度优先级高于智能裁剪
    if pred.multi_scale:
        pred.smart_crop = False

    # 检查模型热加载
    pred.check_and_reload()

    try:
        # 读取图片
        file_bytes = await file.read()
        image = _decode_image(file_bytes)

        # 预测
        result = pred.predict_single(image)

        response = PredictResponse(
            angle=result["angle"],
            time=result["time"],
            cropped=result.get("cropped", False),
        )

        # 返回标注后的图片
        if return_image:
            annotated = draw_angle_on_image(result["image"], result["angle"])
            response.image = _encode_image(annotated)

        return response
    finally:
        # 恢复原始配置
        pred.smart_crop = original_smart_crop
        pred.multi_scale = original_multi_scale


@app.post("/predict/batch", response_model=BatchPredictResponse, summary="批量图片预测")
async def predict_batch(
    files: List[UploadFile] = File(..., description="多个阀门图片文件"),
):
    """上传多张阀门图片，批量返回预测角度

    Args:
        files: 上传的图片文件列表

    Returns:
        每张图片的预测结果列表
    """
    pred = _ensure_predictor()
    start_time = time.time()

    results = []
    for file in files:
        try:
            file_bytes = await file.read()
            image = _decode_image(file_bytes)
            result = pred.predict_single(image)

            results.append(BatchPredictItem(
                filename=file.filename,
                angle=result["angle"],
                time=result["time"],
                cropped=result.get("cropped", False),
            ))
        except Exception as e:
            results.append(BatchPredictItem(
                filename=file.filename,
                time=0.0,
                error=str(e),
            ))

    total_time = time.time() - start_time

    return BatchPredictResponse(
        results=results,
        total_time=round(total_time, 4),
    )


@app.post("/predict/video", response_model=VideoPredictResponse, summary="视频抽帧预测")
async def predict_video(
    file: UploadFile = File(..., description="视频文件（mp4/avi/mov/mkv）"),
    fps: Optional[float] = Query(None, description="每秒抽帧数（与 frame_interval 二选一）"),
    frame_interval: Optional[int] = Query(None, description="帧间隔（与 fps 二选一）"),
):
    """上传视频文件，按指定频率抽帧并预测阀门角度

    Args:
        file: 上传的视频文件
        fps: 每秒抽帧数（如 2.0 = 每秒 2 帧）
        frame_interval: 帧间隔（如 30 = 每 30 帧抽 1 帧）

    Returns:
        每帧的预测结果列表
    """
    pred = _ensure_predictor()

    # 保存上传的视频到临时文件
    import tempfile
    import os as _os

    file_bytes = await file.read()
    suffix = _os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        start_time = time.time()
        df = pred.predict_video(
            video_path=tmp_path,
            output_dir=None,
            fps=fps,
            frame_interval=frame_interval,
            save_frames=False,
            save_video=False,
        )
        total_time = time.time() - start_time

        results = []
        for _, row in df.iterrows():
            results.append(VideoFrameResult(
                frame_idx=int(row["帧索引"]),
                timestamp=float(row["时间戳(秒)"]),
                angle=float(row["预测角度"]),
                time=float(row["处理时间(秒)"]),
            ))

        return VideoPredictResponse(
            total_frames=len(df),
            processed_frames=len(df),
            total_time=round(total_time, 4),
            results=results,
        )
    finally:
        _os.unlink(tmp_path)


@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health():
    """检查服务健康状态

    Returns:
        服务状态和模型加载状态
    """
    return HealthResponse(
        status="ok" if predictor is not None else "model_not_loaded",
        model_loaded=predictor is not None,
    )


@app.get("/info", response_model=InfoResponse, summary="模型信息")
async def info():
    """返回模型和系统信息

    Returns:
        模型名称、路径、图像尺寸、角度范围等
    """
    pred = _ensure_predictor()

    return InfoResponse(
        model_name=pred.model_name,
        model_path=pred.model_path,
        image_size=pred.image_size,
        angle_range=f"{pred.angle_min}° - {pred.angle_max}°",
        device=str(pred.device),
        optimization_enabled=pred.use_optimization,
        smart_crop_enabled=pred.smart_crop,
        multi_scale_enabled=pred.multi_scale,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"API 异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误: {str(exc)}"},
    )


if __name__ == "__main__":
    import asyncio

    ssl_keyfile = os.environ.get("SSL_KEYFILE")
    ssl_certfile = os.environ.get("SSL_CERTFILE")
    https_port = int(os.environ.get("HTTPS_PORT", "8443"))
    http_port = int(os.environ.get("HTTP_PORT", "8000"))

    if ssl_keyfile and ssl_certfile:
        # 同时启动 HTTPS 和 HTTP 两个 API 服务（均提供完整接口，不重定向）
        async def serve_dual():
            https_config = uvicorn.Config(
                "api.main:app",
                host="0.0.0.0",
                port=https_port,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
            )
            http_config = uvicorn.Config(
                "api.main:app",
                host="0.0.0.0",
                port=http_port,
            )
            https_server = uvicorn.Server(https_config)
            http_server = uvicorn.Server(http_config)
            await asyncio.gather(https_server.serve(), http_server.serve())

        logger.info(f"HTTPS 服务启动: https://0.0.0.0:{https_port}")
        logger.info(f"HTTP 服务启动: http://0.0.0.0:{http_port}")
        asyncio.run(serve_dual())
    else:
        # HTTP 模式（向后兼容）
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=http_port,
            reload=True,
        )
