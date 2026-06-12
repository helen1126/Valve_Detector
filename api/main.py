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
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from predict import ValvePredictor
from utils.image_utils import draw_angle_on_image
from utils.logger import setup_logger, get_logger

# 初始化日志
setup_logger(log_dir="./logs")
logger = get_logger()

# 创建 FastAPI 应用
app = FastAPI(
    title="阀门角度检测 API",
    description="工业阀门角度检测系统 RESTful API 接口",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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

# 配置
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "./weights/last.ckpt")
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "convnext_base")
DEFAULT_IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "384"))
DEFAULT_SMART_CROP = os.environ.get("SMART_CROP", "true").lower() in ("true", "1", "yes")
DEFAULT_MULTI_SCALE = os.environ.get("MULTI_SCALE", "false").lower() in ("true", "1", "yes")


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


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型"""
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
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
