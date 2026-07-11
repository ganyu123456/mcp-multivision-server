# MCP Multivision Server

为大模型提供**图片与视频分析**能力的 MCP 服务器：本地用 ffmpeg / OpenCV 做抽帧与结构化 CV 预处理，
云端用**任意 OpenAI 兼容的视觉大模型**（通义千问VL / GLM-4V / 豆包 / GPT-4o 等）做理解与问答。

> 设计理念：宿主 LLM 本身能看图，本服务的价值在于它做不到的部分——**视频**（LLM 不能直接吃视频，
> 靠本地抽关键帧并标注时间戳再喂给视觉模型）以及**结构化 CV 原语**（人脸坐标、EXIF、图像相似度）。
> OCR / 目标识别由 `vision_analyze_image` 通过 prompt 完成（视觉大模型在这方面很强），避免引入重依赖。

## 功能

- **图片理解问答**：描述、问答、OCR、图表解读、UI 分析、示意图理解、报错诊断（`vision_analyze_image`，含任务预设）
- **视频理解**：本地抽关键帧 + 时间戳接地，交云端模型按时间顺序分析（`vision_analyze_video`）
- **视频预处理（本地）**：元信息、抽帧、抽音轨、场景切分
- **图片专门 CV（本地）**：EXIF/尺寸元信息、人脸检测、图像相似度对比
- 支持 stdio 与 SSE 两种传输
- 输入支持四种格式：本地绝对路径 / `file://` / `http(s)://` / base64(data URI)

## 快速开始

### 1. 配置云端视觉模型（OpenAI 兼容）

一份配置靠 `base_url + api_key + model` 切任意一家：

| 平台 | MCP_VISION_BASE_URL | 示例 MCP_VISION_MODEL |
|------|--------------------|----------------------|
| 通义千问VL | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-vl-max` |
| 智谱 GLM-4V | `https://open.bigmodel.cn/api/paas/v4` | `glm-4v` |
| 豆包 | `https://ark.cn-beijing.volces.com/api/v3` | `doubao-vision-pro-32k` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |

> 本地 CV / 视频预处理工具无需 key 即可使用；仅 `vision_analyze_image` / `vision_analyze_video` 需要配置。

### 2. 本地开发运行

需要本机安装 **ffmpeg**（`brew install ffmpeg` / `apt install ffmpeg`）。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[sse]"

# stdio（本地 MCP 客户端）
mcp-multivision-server

# SSE（远程 MCP 客户端）
MCP_TRANSPORT=sse MCP_PORT=8093 mcp-multivision-server
```

## 部署

### Docker Compose

```bash
cp .env.example .env       # 填入 MCP_VISION_BASE_URL / API_KEY / MODEL
docker compose up -d
```

镜像内已内置静态 ffmpeg 与 OpenCV 运行库，无需额外安装。

### CI/CD 自动构建发布

推送 `v*` tag 触发 GitHub Actions：原生 amd64 + arm64 构建、推送 Harbor、合成多架构 manifest、创建 GitHub Release。
需在仓库配置 `HARBOR_USERNAME` / `HARBOR_PASSWORD` secrets。

## MCP 客户端配置

SSE：

```json
{
  "mcpServers": {
    "multivision": { "url": "http://<your-server>:8093/sse" }
  }
}
```

stdio：

```json
{
  "mcpServers": {
    "multivision": {
      "command": "mcp-multivision-server",
      "env": {
        "MCP_VISION_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "MCP_VISION_API_KEY": "your_api_key",
        "MCP_VISION_MODEL": "qwen-vl-max"
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
| `MCP_VISION_MODEL` | — | 模型名 |
| `MCP_VISION_MAX_TOKENS` | `1024` | 生成上限 |
| `MCP_VISION_TEMPERATURE` | `0.2` | 采样温度 |
| `MCP_VISION_TIMEOUT` | `60` | 请求超时（秒） |
| `MCP_VISION_MAX_RETRIES` | `3` | 失败重试次数 |
| `MCP_VISION_MAX_FRAMES` | `8` | analyze_video 送模型的最大帧数 |
| `MCP_VISION_MAX_IMAGE_SIZE` | `20971520` | 单图字节上限（20MB） |
| `MCP_VISION_ALLOWED_IMAGE_FORMATS` | `jpeg,png,webp,gif,bmp,tiff` | 允许的图片格式 |
| `MCP_VISION_CACHE_ENABLED` | `false` | 是否缓存分析结果 |
| `MCP_VISION_CACHE_DIR` | `/tmp/mcp-vision-cache` | 缓存目录 |
| `MCP_VISION_WORK_DIR` | `/tmp/mcp-vision` | 抽帧/音轨临时目录 |
| `MCP_FFMPEG_BIN` / `MCP_FFPROBE_BIN` | PATH | ffmpeg/ffprobe 路径 |

## MCP 工具列表

| 工具 | 说明 | 是否需云端 key |
|------|------|:--:|
| `vision_analyze_image` | 图片描述/问答/OCR/图表/UI/报错（含 preset） | 是 |
| `vision_analyze_video` | 抽关键帧+时间戳，按时间分析视频 | 是 |
| `vision_video_info` | ffprobe 视频元信息 | 否 |
| `vision_extract_frames` | 抽帧为图片文件，返回路径+时间戳 | 否 |
| `vision_extract_audio` | 抽音轨为 mp3/wav | 否 |
| `vision_scene_detect` | 场景切换时间戳 | 否 |
| `vision_image_metadata` | 尺寸/格式/EXIF/GPS | 否 |
| `vision_detect_faces` | OpenCV 人脸检测（数量+坐标） | 否 |
| `vision_compare_images` | 感知哈希+直方图相似度 | 否 |
| `vision_get_server_status` | 配置与可用性自检 | 否 |

`vision_analyze_image` 的 `preset` 可选：`describe`（描述）、`ocr`（文字识别）、`chart`（图表）、
`ui`（界面）、`diagram`（示意图）、`error`（报错诊断）；提供 `prompt` 时覆盖 preset。

## 项目结构

```
03-mcp-multivision-server/
├── Dockerfile / docker-compose.yaml / .env.example
├── pyproject.toml / requirements.txt
├── .github/workflows/build-release.yaml
└── src/mcp_multivision_server/
    ├── server.py                 # 入口 + 10 个工具
    ├── providers/                # 云端 VLM
    │   ├── base.py               # BaseVisionProvider / VisionResult / ProviderError
    │   └── openai_compat.py      # OpenAI 兼容 provider
    └── media/                    # 本地处理
        ├── inputs.py             # 四格式输入解析 + 校验
        ├── ffmpeg.py             # 探测/抽帧/抽音轨/场景切分
        └── cv.py                 # EXIF/人脸/相似度
```

## 许可

MIT
