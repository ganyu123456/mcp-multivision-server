FROM mwader/static-ffmpeg:7.1 AS ffmpeg

FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

# OpenCV (headless) 运行期所需系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 静态编译的 ffmpeg / ffprobe，无需 apt 安装
COPY --from=ffmpeg /ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg /ffprobe /usr/local/bin/ffprobe

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY src/ ./src/
COPY pyproject.toml ./

RUN pip install --no-cache-dir -e ".[sse]"

ENV MCP_TRANSPORT=sse \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8093 \
    MCP_VISION_PROVIDER=openai \
    MCP_FFMPEG_BIN=/usr/local/bin/ffmpeg \
    MCP_FFPROBE_BIN=/usr/local/bin/ffprobe \
    MCP_VISION_WORK_DIR=/tmp/mcp-vision

EXPOSE 8093

ENTRYPOINT ["python", "-m", "mcp_multivision_server.server"]
