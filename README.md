# MCP Multivision Server

为大模型提供**图片与视频分析**能力的 MCP 服务器,**纯云端**:所有视觉理解都交给
OpenAI 兼容的云端视觉大模型完成,**视频原生交给模型处理(如 Qwen3.5,通过 DashScope `video_url`)——
本地不做任何抽帧/转码**,镜像轻量、无 ffmpeg / OpenCV 依赖。

> 设计:视频的抽帧与时序对齐由模型服务端完成。一份 `base_url + api_key + model` 配置即可切换
> 通义千问3.5 / Qwen-VL / GLM-4V / GPT-4o(GPT-4o 仅图片,不支持 `video_url`)。

## 功能

- **图片理解问答**:描述、问答、OCR、图表解读、UI 分析、示意图理解、报错诊断(`vision_analyze_image`,含任务预设)
- **视频理解(原生)**:把视频直接交给云端多模态大模型,按时间顺序分析场景/对象/动作/事件(`vision_analyze_video`)
- **图片元信息(本地)**:EXIF/尺寸/格式,Pillow 解析(`vision_image_metadata`)
- 支持 stdio 与 SSE 两种传输
- 输入支持:本地绝对路径 / `file://` / `http(s)://` / base64(data URI);视频推荐传 http(s) URL

## 快速开始

### 1. 配置云端视觉模型(OpenAI 兼容)

视频需选**支持原生视频**的模型:

| 平台 | MCP_VISION_BASE_URL | 示例 MCP_VISION_MODEL | 视频 |
|------|--------------------|----------------------|:--:|
| 通义千问3.5 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3.5-plus` | ✓ |
| 通义千问VL | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-vl-max` | ✓ |
| 智谱 GLM-4V | `https://open.bigmodel.cn/api/paas/v4` | `glm-4v` | 视模型 |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` | ✗(仅图片) |

> `video_url` 是 DashScope 对 OpenAI 协议的扩展,因此原生视频当前主要在通义千问系列可用;
> OpenAI 官方 GPT-4o 不支持 `video_url`,只能做图片。

### 2. 本地开发运行

无需 ffmpeg,纯 Python 依赖:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[sse]"

# stdio（本地 MCP 客户端）
mcp-multivision-server

# SSE（远程 MCP 客户端）
MCP_TRANSPORT=sse MCP_PORT=8093 mcp-multivision-server
```

## 部署

```bash
cp .env.example .env       # 填入 MCP_VISION_BASE_URL / API_KEY / MODEL
docker compose up -d
```

镜像基于 `python:3.11-slim`,无 ffmpeg / OpenCV / 系统库依赖,体积约 200MB。

推送 `v*` tag 触发 GitHub Actions:原生 amd64 + arm64 构建、推送 Harbor、多架构 manifest、GitHub Release。
需配置仓库 secrets `HARBOR_USERNAME` / `HARBOR_PASSWORD`。

## MCP 客户端配置

SSE:

```json
{ "mcpServers": { "multivision": { "url": "http://<your-server>:8093/sse" } } }
```

stdio:

```json
{
  "mcpServers": {
    "multivision": {
      "command": "mcp-multivision-server",
      "env": {
        "MCP_VISION_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "MCP_VISION_API_KEY": "your_api_key",
        "MCP_VISION_MODEL": "qwen3.5-plus"
      }
    }
  }
}
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MCP_TRANSPORT` | `stdio` | `stdio` 或 `sse` |
| `MCP_HOST` / `MCP_PORT` | `0.0.0.0` / `8093` | SSE 监听地址 |
| `MCP_VISION_PROVIDER` | `openai` | provider 名（当前支持 openai 兼容） |
| `MCP_VISION_BASE_URL` | — | 视觉模型 API 基址 |
| `MCP_VISION_API_KEY` | — | 视觉模型 API Key |
| `MCP_VISION_MODEL` | — | 模型名（视频需支持原生视频，如 qwen3.5-plus） |
| `MCP_VISION_MAX_TOKENS` | `1024` | 生成上限 |
| `MCP_VISION_TEMPERATURE` | `0.2` | 采样温度 |
| `MCP_VISION_TIMEOUT` | `120` | 请求超时（秒），视频较慢 |
| `MCP_VISION_MAX_RETRIES` | `3` | 失败重试次数 |
| `MCP_VISION_MAX_IMAGE_SIZE` | `20971520` | 单图 base64 字节上限（20MB） |
| `MCP_VISION_MAX_VIDEO_SIZE` | `104857600` | 本地视频转 base64 上限（100MB），更大请传 URL |
| `MCP_VISION_ALLOWED_IMAGE_FORMATS` | `jpeg,png,webp,gif,bmp,tiff` | 允许的图片格式 |
| `MCP_VISION_CACHE_ENABLED` | `false` | 是否缓存分析结果 |
| `MCP_VISION_CACHE_DIR` | `/tmp/mcp-vision-cache` | 缓存目录 |

## MCP 工具列表

| 工具 | 说明 | 是否需云端 key |
|------|------|:--:|
| `vision_analyze_image` | 图片描述/问答/OCR/图表/UI/报错（含 preset） | 是 |
| `vision_analyze_video` | 视频原生交给云端模型理解（video_url） | 是 |
| `vision_image_metadata` | 尺寸/格式/EXIF/GPS（本地 Pillow） | 否 |
| `vision_get_server_status` | 配置与可用性自检 | 否 |

`vision_analyze_image` 的 `preset` 可选:`describe` / `ocr` / `chart` / `ui` / `diagram` / `error`;提供 `prompt` 时覆盖 preset。

## 项目结构

```
03-mcp-multivision-server/
├── Dockerfile / docker-compose.yaml / .env.example
├── pyproject.toml / requirements.txt
├── .github/workflows/build-release.yaml
└── src/mcp_multivision_server/
    ├── server.py                 # 入口 + 4 个工具
    ├── providers/                # 云端 VLM
    │   ├── base.py               # BaseVisionProvider / VisionResult / ProviderError
    │   └── openai_compat.py      # OpenAI 兼容 provider（含 video_url）
    └── media/
        ├── inputs.py             # 输入解析：图片/视频 → image_url / video_url
        └── imageinfo.py          # 本地 EXIF/尺寸（Pillow）
```

## 许可

MIT
