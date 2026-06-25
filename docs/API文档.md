# API 文档

阀门角度检测系统 RESTful API 接口文档。API 基于 FastAPI 构建，提供单张/批量/视频预测、健康检查和模型信息查询功能。

**基础信息**：

- 基础 URL：`http://localhost:8000`（HTTP 模式）或 `https://localhost:8443`（HTTPS 模式）
- 交互式文档：HTTP 模式 http://localhost:8000/docs ，HTTPS 模式 https://localhost:8443/docs （Swagger UI）
- 备选文档：HTTP 模式 http://localhost:8000/redoc ，HTTPS 模式 https://localhost:8443/redoc （ReDoc）

> **HTTPS 模式**：配置 `SSL_KEYFILE` 和 `SSL_CERTFILE` 环境变量后，API 以 HTTPS 模式运行，详见[部署指南](部署指南.md#https-部署)。

## 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/predict` | 单张图片预测 |
| POST | `/predict/batch` | 批量图片预测 |
| POST | `/predict/video` | 视频抽帧预测 |
| GET | `/health` | 健康检查 |
| GET | `/info` | 模型信息查询 |

---

## POST /predict

上传单张阀门图片，返回预测角度。

### 请求

- **Content-Type**：`multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 阀门图片文件（支持 jpg/jpeg/png/bmp） |
| `return_image` | bool | 否 | 是否返回标注后的图片（base64 编码），默认 `true` |
| `smart_crop` | bool | 否 | 是否启用智能裁剪（远距离拍摄时自动定位并放大阀门区域），默认为服务配置值 |
| `multi_scale` | bool | 否 | 是否启用多尺度推理（结合原图和裁剪图预测，精度更高），默认为服务配置值 |

> **注意**：`multi_scale` 优先级高于 `smart_crop`，同时启用时使用多尺度推理。

### 响应

**成功响应**（200）：

```json
{
  "angle": 25.3,
  "time": 0.0452,
  "image": "base64编码的标注图片字符串..."
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `angle` | float | 预测角度（°），范围 0.0~80.0 |
| `time` | float | 处理耗时（秒） |
| `image` | string \| null | 标注后的图片（base64 编码），`return_image=false` 时为 `null` |

### 调用示例

**Python requests**：

```python
import requests

url = "http://localhost:8000/predict"

# 基本预测（返回标注图片）
with open("valve.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})

result = response.json()
print(f"预测角度: {result['angle']}°")
print(f"处理时间: {result['time']}秒")

# 不返回图片（减少响应体积）
with open("valve.jpg", "rb") as f:
    response = requests.post(url, files={"file": f}, params={"return_image": False})

result = response.json()
print(f"预测角度: {result['angle']}°")
```

**curl**：

```bash
# 基本预测
curl -X POST "http://localhost:8000/predict" \
  -F "file=@valve.jpg"

# 不返回图片
curl -X POST "http://localhost:8000/predict?return_image=false" \
  -F "file=@valve.jpg"
```

---

## POST /predict/batch

上传多张阀门图片，批量返回预测角度。

### 请求

- **Content-Type**：`multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `files` | File[] | 是 | 多个阀门图片文件 |

### 响应

**成功响应**（200）：

```json
{
  "results": [
    {
      "filename": "valve_001.jpg",
      "angle": 25.3,
      "time": 0.0452,
      "error": null
    },
    {
      "filename": "valve_002.jpg",
      "angle": null,
      "time": 0.0,
      "error": "图片解码失败，请检查文件格式"
    }
  ],
  "total_time": 0.1234
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `results` | array | 预测结果列表 |
| `results[].filename` | string | 文件名 |
| `results[].angle` | float \| null | 预测角度（°），失败时为 `null` |
| `results[].time` | float | 单张处理耗时（秒） |
| `results[].error` | string \| null | 错误信息，成功时为 `null` |
| `total_time` | float | 总处理耗时（秒） |

### 调用示例

**Python requests**：

```python
import requests

url = "http://localhost:8000/predict/batch"

files = [
    ("files", ("valve_001.jpg", open("valve_001.jpg", "rb"), "image/jpeg")),
    ("files", ("valve_002.jpg", open("valve_002.jpg", "rb"), "image/jpeg")),
    ("files", ("valve_003.jpg", open("valve_003.jpg", "rb"), "image/jpeg")),
]

response = requests.post(url, files=files)
result = response.json()

for item in result["results"]:
    if item["angle"] is not None:
        print(f"{item['filename']}: {item['angle']}°")
    else:
        print(f"{item['filename']}: 错误 - {item['error']}")

print(f"总耗时: {result['total_time']}秒")
```

**curl**：

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@valve_001.jpg" \
  -F "files=@valve_002.jpg" \
  -F "files=@valve_003.jpg"
```

---

## POST /predict/video

上传视频文件，按指定频率抽帧并预测阀门角度。支持两种抽帧方式：按每秒帧数（fps）或按帧间隔（frame_interval）。

### 请求

- **Content-Type**：`multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 视频文件（支持 mp4/avi/mov/mkv） |
| `fps` | float | 否 | 每秒抽帧数（如 2.0 = 每秒 2 帧），与 `frame_interval` 二选一 |
| `frame_interval` | int | 否 | 帧间隔（如 30 = 每 30 帧抽 1 帧），与 `fps` 二选一 |

> **注意**：`fps` 和 `frame_interval` 都不指定时，默认每秒抽 1 帧。

### 响应

**成功响应**（200）：

```json
{
  "total_frames": 60,
  "processed_frames": 60,
  "total_time": 12.3456,
  "results": [
    {
      "frame_idx": 0,
      "timestamp": 0.0,
      "angle": 25.3,
      "time": 0.0452
    },
    {
      "frame_idx": 30,
      "timestamp": 1.0,
      "angle": 26.1,
      "time": 0.0431
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_frames` | int | 处理的总帧数 |
| `processed_frames` | int | 实际预测的帧数 |
| `total_time` | float | 总处理耗时（秒） |
| `results` | array | 预测结果列表 |
| `results[].frame_idx` | int | 帧索引（在原视频中的位置） |
| `results[].timestamp` | float | 时间戳（秒） |
| `results[].angle` | float | 预测角度（°），范围 0.0~80.0 |
| `results[].time` | float | 单帧处理耗时（秒） |

### 调用示例

**Python requests**：

```python
import requests

url = "http://localhost:8000/predict/video"

# 按每秒 2 帧抽帧
with open("valve_video.mp4", "rb") as f:
    response = requests.post(url, files={"file": f}, params={"fps": 2})

result = response.json()
print(f"总帧数: {result['total_frames']}")
print(f"总耗时: {result['total_time']}秒")

for item in result["results"]:
    print(f"  帧 {item['frame_idx']} ({item['timestamp']:.1f}s): {item['angle']}°")

# 按帧间隔抽帧（每 30 帧抽 1 帧）
with open("valve_video.mp4", "rb") as f:
    response = requests.post(url, files={"file": f}, params={"frame_interval": 30})
```

**curl**：

```bash
# 按每秒 2 帧抽帧
curl -X POST "http://localhost:8000/predict/video?fps=2" \
  -F "file=@valve_video.mp4"

# 按帧间隔抽帧
curl -X POST "http://localhost:8000/predict/video?frame_interval=30" \
  -F "file=@valve_video.mp4"
```

---

## GET /health

检查服务健康状态。

### 请求

无参数。

### 响应

**成功响应**（200）：

```json
{
  "status": "ok",
  "model_loaded": true
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 服务状态：`ok`（正常）或 `model_not_loaded`（模型未加载） |
| `model_loaded` | bool | 模型是否已加载 |

### 调用示例

**Python requests**：

```python
import requests

response = requests.get("http://localhost:8000/health")
result = response.json()
print(f"状态: {result['status']}, 模型已加载: {result['model_loaded']}")
```

**curl**：

```bash
curl http://localhost:8000/health
```

---

## GET /info

返回模型和系统信息。

### 请求

无参数。

### 响应

**成功响应**（200）：

```json
{
  "model_name": "convnext_base",
  "model_path": "./weights/last.ckpt",
  "image_size": 384,
  "angle_range": "0.0° - 80.0°",
  "device": "cuda:0",
  "optimization_enabled": false,
  "smart_crop_enabled": true,
  "multi_scale_enabled": false
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_name` | string | 模型架构名称 |
| `model_path` | string | 模型权重路径 |
| `image_size` | int | 输入图像尺寸 |
| `angle_range` | string | 角度检测范围 |
| `device` | string | 计算设备（如 `cuda:0` 或 `cpu`） |
| `optimization_enabled` | bool | 是否启用图像优化 |
| `smart_crop_enabled` | bool | 是否启用智能裁剪（远距离优化） |
| `multi_scale_enabled` | bool | 是否启用多尺度推理 |

### 调用示例

**Python requests**：

```python
import requests

response = requests.get("http://localhost:8000/info")
result = response.json()
print(f"模型: {result['model_name']}")
print(f"设备: {result['device']}")
print(f"角度范围: {result['angle_range']}")
```

**curl**：

```bash
curl http://localhost:8000/info
```

---

## 错误码说明

| HTTP 状态码 | 说明 | 触发场景 |
|-------------|------|----------|
| 200 | 成功 | 请求处理成功 |
| 400 | 请求错误 | 图片解码失败、文件格式不支持 |
| 422 | 参数验证失败 | 缺少必填参数、参数类型错误 |
| 500 | 服务器内部错误 | 模型推理异常、未知错误 |
| 503 | 服务不可用 | 模型未加载 |

### 错误响应格式

```json
{
  "detail": "错误描述信息"
}
```

### 常见错误示例

**模型未加载**（503）：

```json
{
  "detail": "模型未加载，请检查模型权重文件"
}
```

**图片解码失败**（400）：

```json
{
  "detail": "图片解码失败，请检查文件格式"
}
```

**服务器内部错误**（500）：

```json
{
  "detail": "服务器内部错误: [具体错误信息]"
}
```

---

## 完整调用示例

### Python 完整示例

```python
import requests
import base64
import json

BASE_URL = "http://localhost:8000"

# 1. 健康检查
health = requests.get(f"{BASE_URL}/health").json()
print(f"服务状态: {health['status']}")

# 2. 查看模型信息
info = requests.get(f"{BASE_URL}/info").json()
print(f"模型: {info['model_name']}, 设备: {info['device']}")

# 3. 单张预测
with open("valve.jpg", "rb") as f:
    result = requests.post(
        f"{BASE_URL}/predict",
        files={"file": f},
        params={"return_image": True}
    ).json()
print(f"预测角度: {result['angle']}°, 耗时: {result['time']}秒")

# 保存返回的标注图片
if result["image"]:
    img_data = base64.b64decode(result["image"])
    with open("result.jpg", "wb") as f:
        f.write(img_data)

# 4. 批量预测
files = [
    ("files", ("v1.jpg", open("valve_001.jpg", "rb"), "image/jpeg")),
    ("files", ("v2.jpg", open("valve_002.jpg", "rb"), "image/jpeg")),
]
batch_result = requests.post(f"{BASE_URL}/predict/batch", files=files).json()
for item in batch_result["results"]:
    print(f"  {item['filename']}: {item['angle']}°")

# 5. 视频抽帧预测（按每秒 2 帧抽帧）
with open("valve_video.mp4", "rb") as f:
    video_result = requests.post(
        f"{BASE_URL}/predict/video",
        files={"file": f},
        params={"fps": 2}
    ).json()
print(f"视频总帧数: {video_result['total_frames']}, 总耗时: {video_result['total_time']}秒")
for item in video_result["results"]:
    print(f"  帧 {item['frame_idx']} ({item['timestamp']:.1f}s): {item['angle']}°")
```

### curl 完整示例

```bash
# 健康检查
curl http://localhost:8000/health

# 模型信息
curl http://localhost:8000/info

# 单张预测
curl -X POST "http://localhost:8000/predict" \
  -F "file=@valve.jpg" \
  -o result.json

# 批量预测
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@valve_001.jpg" \
  -F "files=@valve_002.jpg" \
  -o batch_result.json
```
