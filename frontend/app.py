"""Streamlit 前端 Demo

阀门角度检测系统的 Web 界面，支持单张/批量图片预测和结果展示。

启动方式：
    streamlit run frontend/app.py
"""

import io
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml

# 添加项目根目录到系统路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from predict import ValvePredictor
from utils.image_utils import draw_angle_on_image, bgr_to_rgb, read_image, save_image
from utils.image_optimization import ImageOptimizer

# 页面配置
st.set_page_config(
    page_title="阀门角度检测系统",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_predictor(model_path: str, model_name: str, image_size: int, use_optimization: bool):
    """加载预测器（缓存资源，避免重复加载）

    Args:
        model_path: 模型权重路径
        model_name: 模型架构名称
        image_size: 图像尺寸
        use_optimization: 是否使用图像优化

    Returns:
        ValvePredictor 实例
    """
    config_path = os.path.join(project_root, "config", "data_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data_config = yaml.safe_load(f)
    else:
        data_config = {}

    return ValvePredictor(
        model_path=model_path,
        model_name=model_name,
        image_size=image_size,
        angle_min=data_config.get("angle_min", 0.0),
        angle_max=data_config.get("angle_max", 80.0),
        use_optimization=use_optimization,
    )


def main():
    """主界面"""
    # 标题
    st.title("🔧 工业阀门角度检测系统")
    st.markdown("---")

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 参数配置")

        # 模型配置
        st.subheader("模型设置")
        model_path = st.text_input(
            "模型权重路径",
            value="./weights/last.ckpt",
            help="模型权重文件路径（.ckpt/.pth/.onnx）",
        )
        model_name = st.selectbox(
            "模型架构",
            ["convnext_base", "resnet50", "efficientnet_b4", "swin_base"],
            index=0,
            help="选择模型架构，需与权重文件匹配",
        )
        image_size = st.selectbox(
            "图像尺寸",
            [384, 224, 512],
            index=0,
            help="输入图像尺寸，精度优先选择 384 或 512",
        )

        # 图像优化
        st.subheader("图像优化")
        use_optimization = st.checkbox(
            "启用图像优化",
            value=False,
            help="启用颜色增强、边缘检测等图像优化",
        )

        # 加载模型按钮
        if st.button("🔄 加载模型", use_container_width=True):
            if os.path.exists(model_path):
                with st.spinner("正在加载模型..."):
                    try:
                        # 清除缓存，重新加载
                        st.cache_resource.clear()
                        load_predictor(model_path, model_name, image_size, use_optimization)
                        st.success("模型加载成功！")
                    except Exception as e:
                        st.error(f"模型加载失败: {e}")
            else:
                st.warning(f"模型文件不存在: {model_path}")

        st.markdown("---")
        st.info("💡 提示：首次使用请先加载模型，然后上传图片进行预测。")

    # 主界面
    tab1, tab2 = st.tabs(["📷 单张预测", "📁 批量预测"])

    # ===== 单张预测 =====
    with tab1:
        st.subheader("单张图片预测")

        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader(
                "上传阀门图片",
                type=["jpg", "jpeg", "png", "bmp"],
                key="single_upload",
            )

            if uploaded_file is not None:
                # 显示原始图片
                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if image is not None:
                    st.image(
                        bgr_to_rgb(image),
                        caption="原始图片",
                        use_container_width=True,
                    )
                    uploaded_file.seek(0)  # 重置文件指针

        with col2:
            if uploaded_file is not None and image is not None:
                if st.button("🔍 开始预测", key="predict_single", use_container_width=True):
                    if not os.path.exists(model_path):
                        st.error("请先在侧边栏配置模型路径并加载模型！")
                    else:
                        with st.spinner("正在预测..."):
                            try:
                                predictor = load_predictor(
                                    model_path, model_name, image_size, use_optimization
                                )
                                result = predictor.predict_single(image)

                                # 显示预测结果
                                st.metric("预测角度", f"{result['angle']}°")
                                st.metric("处理时间", f"{result['time']:.4f} 秒")

                                # 显示标注后的图片
                                annotated = draw_angle_on_image(
                                    result["image"], result["angle"]
                                )
                                st.image(
                                    bgr_to_rgb(annotated),
                                    caption=f"预测结果: {result['angle']}°",
                                    use_container_width=True,
                                )

                            except Exception as e:
                                st.error(f"预测失败: {e}")

    # ===== 批量预测 =====
    with tab2:
        st.subheader("批量图片预测")

        uploaded_files = st.file_uploader(
            "上传多张阀门图片",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key="batch_upload",
        )

        if uploaded_files:
            st.info(f"已上传 {len(uploaded_files)} 张图片")

            if st.button("🔍 批量预测", key="predict_batch", use_container_width=True):
                if not os.path.exists(model_path):
                    st.error("请先在侧边栏配置模型路径并加载模型！")
                else:
                    with st.spinner("正在批量预测..."):
                        try:
                            predictor = load_predictor(
                                model_path, model_name, image_size, use_optimization
                            )

                            results = []
                            progress_bar = st.progress(0)

                            for i, uploaded_file in enumerate(uploaded_files):
                                file_bytes = np.frombuffer(
                                    uploaded_file.read(), np.uint8
                                )
                                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                                if image is not None:
                                    result = predictor.predict_single(image)
                                    results.append({
                                        "文件名": uploaded_file.name,
                                        "预测角度(°)": result["angle"],
                                        "处理时间(秒)": result["time"],
                                    })

                                progress_bar.progress((i + 1) / len(uploaded_files))

                            # 显示结果表格
                            if results:
                                df = pd.DataFrame(results)
                                st.dataframe(df, use_container_width=True)

                                # 统计信息
                                st.markdown("---")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("预测数量", f"{len(results)}")
                                with col2:
                                    st.metric(
                                        "平均角度",
                                        f"{df['预测角度(°)'].mean():.1f}°",
                                    )
                                with col3:
                                    st.metric(
                                        "角度范围",
                                        f"{df['预测角度(°)'].min():.1f}° - {df['预测角度(°)'].max():.1f}°",
                                    )

                                # 下载 CSV
                                csv = df.to_csv(index=False, encoding="utf-8-sig")
                                st.download_button(
                                    "📥 下载预测结果 (CSV)",
                                    data=csv,
                                    file_name="predictions.csv",
                                    mime="text/csv",
                                )

                        except Exception as e:
                            st.error(f"批量预测失败: {e}")


if __name__ == "__main__":
    main()
