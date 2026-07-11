#!/usr/bin/env python3
"""MCP Multivision Server - Image and video analysis tools for LLM agents.

Usage:
    mcp-multivision-server                    # Run with default settings (stdio)
    MCP_TRANSPORT=sse mcp-multivision-server  # Run as an HTTP/SSE server
    MCP_PORT=8093 mcp-multivision-server      # Custom SSE port

Local ffmpeg/OpenCV handle frame extraction and structured CV; a cloud
OpenAI-compatible vision model handles image/video understanding. Configure the
model via MCP_VISION_BASE_URL, MCP_VISION_API_KEY and MCP_VISION_MODEL.
"""

import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from typing import Any, Optional

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent

from .providers.base import ProviderError, VisionResult
from .providers.openai_compat import OpenAICompatProvider
from .media import cv, ffmpeg, inputs
from .media.cv import CVError
from .media.ffmpeg import FFmpegError
from .media.inputs import InputError

load_dotenv()

PROVIDER_NAME = os.getenv("MCP_VISION_PROVIDER", "openai").strip()
BASE_URL = os.getenv("MCP_VISION_BASE_URL", "").strip()
API_KEY = os.getenv("MCP_VISION_API_KEY", "").strip()
MODEL = os.getenv("MCP_VISION_MODEL", "").strip()
WORK_DIR = os.getenv("MCP_VISION_WORK_DIR", "/tmp/mcp-vision")
CACHE_ENABLED = os.getenv("MCP_VISION_CACHE_ENABLED", "false").strip().lower() in ("1", "true", "yes")
CACHE_DIR = os.getenv("MCP_VISION_CACHE_DIR", "/tmp/mcp-vision-cache")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


MAX_TOKENS = _env_int("MCP_VISION_MAX_TOKENS", 1024)
TEMPERATURE = _env_float("MCP_VISION_TEMPERATURE", 0.2)
TIMEOUT = _env_float("MCP_VISION_TIMEOUT", 60.0)
MAX_RETRIES = _env_int("MCP_VISION_MAX_RETRIES", 3)
MAX_FRAMES = _env_int("MCP_VISION_MAX_FRAMES", 8)

server = Server("mcp-multivision-server")

_provider = None

PROMPT_PRESETS = {
    "describe": "请详细描述这张图片的内容，包括主要对象、场景、颜色以及所有可见细节。",
    "ocr": "请提取图片中所有可见的文字，尽量保持原始布局与顺序，不要翻译或改写。",
    "chart": "这是一张图表。请判断其类型，说明坐标轴/图例含义，并总结关键数据与趋势。",
    "ui": "这是一张 UI 界面截图。请分析其布局结构、主要组件、配色方案与交互元素。",
    "diagram": "这是一张示意图或流程图。请解释其中的节点、连接关系与整体流程。",
    "error": "这是一张报错截图。请识别其中的错误信息，分析可能的原因，并给出修复建议。",
}


def _init_provider():
    global _provider
    factory = {
        "openai": lambda: OpenAICompatProvider(
            base_url=BASE_URL,
            api_key=API_KEY,
            model=MODEL,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
            default_max_tokens=MAX_TOKENS,
            default_temperature=TEMPERATURE,
        ),
    }
    builder = factory.get(PROVIDER_NAME)
    _provider = builder() if builder else None


def _get_provider():
    if _provider is None:
        raise ProviderError(PROVIDER_NAME, f"Unknown or unconfigured provider '{PROVIDER_NAME}'.")
    if not _provider.is_available():
        raise ProviderError(
            PROVIDER_NAME,
            "Provider not configured. Set MCP_VISION_BASE_URL, MCP_VISION_API_KEY and MCP_VISION_MODEL.",
        )
    return _provider


# ---------------------------------------------------------------- tool schemas

_IMAGE_PROP = {
    "type": "string",
    "description": "图片输入，支持：本地绝对路径 / file:// URL / http(s) URL / base64(data URI)",
}
_VIDEO_PROP = {
    "type": "string",
    "description": "视频输入，支持：本地绝对路径 / file:// URL / http(s) URL",
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="vision_analyze_image",
            description="图片理解问答：调用云端视觉大模型对图片进行描述、问答、OCR、图表/UI/报错分析等。可用 prompt 自由提问，或用 preset 选择任务类型。",
            inputSchema={
                "type": "object",
                "properties": {
                    "image": _IMAGE_PROP,
                    "prompt": {"type": "string", "description": "自由指令/问题；提供后覆盖 preset", "default": ""},
                    "preset": {
                        "type": "string",
                        "enum": list(PROMPT_PRESETS.keys()),
                        "description": "任务预设：describe 描述 / ocr 文字识别 / chart 图表 / ui 界面 / diagram 示意图 / error 报错诊断",
                        "default": "describe",
                    },
                    "max_tokens": {"type": "integer", "description": "可选，本次生成上限"},
                    "temperature": {"type": "number", "description": "可选，采样温度"},
                },
                "required": ["image"],
            },
        ),
        Tool(
            name="vision_analyze_video",
            description="视频理解：本地抽取关键帧并逐帧标注时间戳后，交由云端视觉大模型按时间顺序分析视频内容（场景、对象、动作、事件）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "video": _VIDEO_PROP,
                    "prompt": {"type": "string", "description": "自由指令/问题，如'这段视频里发生了什么？'", "default": ""},
                    "max_frames": {"type": "integer", "description": f"抽取并送入模型的最大帧数（默认 {MAX_FRAMES}，上限 32）", "default": MAX_FRAMES},
                    "frame_mode": {
                        "type": "string",
                        "enum": ["count", "keyframe", "interval"],
                        "description": "抽帧方式：count 均匀取N帧 / keyframe 关键帧 / interval 按秒间隔",
                        "default": "count",
                    },
                    "interval": {"type": "number", "description": "frame_mode=interval 时的秒间隔", "default": 5.0},
                    "max_tokens": {"type": "integer", "description": "可选，本次生成上限"},
                    "temperature": {"type": "number", "description": "可选，采样温度"},
                },
                "required": ["video"],
            },
        ),
        Tool(
            name="vision_video_info",
            description="视频元信息：用 ffprobe 读取时长、分辨率、帧率、编码、音视频流等（纯本地，不调用云端）。",
            inputSchema={
                "type": "object",
                "properties": {"video": _VIDEO_PROP},
                "required": ["video"],
            },
        ),
        Tool(
            name="vision_extract_frames",
            description="视频抽帧：本地用 ffmpeg 抽取帧并保存为图片文件，返回帧文件路径与时间戳（纯本地）。可配合 vision_analyze_image 逐帧分析。",
            inputSchema={
                "type": "object",
                "properties": {
                    "video": _VIDEO_PROP,
                    "mode": {
                        "type": "string",
                        "enum": ["count", "keyframe", "interval"],
                        "description": "抽帧方式：count 均匀取N帧 / keyframe 关键帧 / interval 按秒间隔",
                        "default": "count",
                    },
                    "count": {"type": "integer", "description": "count/keyframe 模式的帧数量", "default": 8},
                    "interval": {"type": "number", "description": "interval 模式的秒间隔", "default": 5.0},
                },
                "required": ["video"],
            },
        ),
        Tool(
            name="vision_extract_audio",
            description="视频抽音轨：本地用 ffmpeg 提取音频为 mp3/wav 文件（供后续转写），返回音频路径（纯本地）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "video": _VIDEO_PROP,
                    "format": {"type": "string", "enum": ["mp3", "wav"], "description": "输出音频格式", "default": "mp3"},
                },
                "required": ["video"],
            },
        ),
        Tool(
            name="vision_scene_detect",
            description="场景切分：本地用 ffmpeg 检测视频中的场景切换时间点，返回时间戳列表（纯本地）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "video": _VIDEO_PROP,
                    "threshold": {"type": "number", "description": "场景变化阈值 0~1，越大越不敏感", "default": 0.4},
                },
                "required": ["video"],
            },
        ),
        Tool(
            name="vision_image_metadata",
            description="图片元信息：读取尺寸、格式、颜色模式与 EXIF（含拍摄时间、GPS 若有），纯本地 Pillow 解析。",
            inputSchema={
                "type": "object",
                "properties": {"image": _IMAGE_PROP},
                "required": ["image"],
            },
        ),
        Tool(
            name="vision_detect_faces",
            description="人脸检测：本地用 OpenCV Haar 级联检测正面人脸，返回人脸数量与边界框坐标（纯本地，无需联网）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "image": _IMAGE_PROP,
                    "min_neighbors": {"type": "integer", "description": "检测严格度，越大越保守", "default": 5},
                },
                "required": ["image"],
            },
        ),
        Tool(
            name="vision_compare_images",
            description="图片相似度：本地用感知哈希与颜色直方图对比两张图片，返回 0~1 相似度（纯本地）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_a": {**_IMAGE_PROP, "description": "第一张图片（路径/URL/base64）"},
                    "image_b": {**_IMAGE_PROP, "description": "第二张图片（路径/URL/base64）"},
                },
                "required": ["image_a", "image_b"],
            },
        ),
        Tool(
            name="vision_get_server_status",
            description="查询服务状态：云端视觉模型是否已配置、ffmpeg 是否可用、当前配置项。",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        result = await _handle_tool(name, arguments)
        return [TextContent(type="text", text=result)]
    except ProviderError as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e), "provider": e.provider}, ensure_ascii=False))]
    except (InputError, FFmpegError, CVError) as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]


async def _handle_tool(name: str, args: dict[str, Any]) -> str:
    if name == "vision_analyze_image":
        return await _analyze_image(args)
    elif name == "vision_analyze_video":
        return await _analyze_video(args)
    elif name == "vision_video_info":
        return await _video_info(args)
    elif name == "vision_extract_frames":
        return await _extract_frames(args)
    elif name == "vision_extract_audio":
        return await _extract_audio(args)
    elif name == "vision_scene_detect":
        return await _scene_detect(args)
    elif name == "vision_image_metadata":
        return await _image_metadata(args)
    elif name == "vision_detect_faces":
        return await _detect_faces(args)
    elif name == "vision_compare_images":
        return await _compare_images(args)
    elif name == "vision_get_server_status":
        return _server_status()
    else:
        return json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)


# --------------------------------------------------------------- input helpers

async def _prepare_video(src: str) -> tuple[str, bool]:
    """Return (path_or_url, is_temp). http(s) URLs are passed to ffmpeg directly."""
    src = src.strip()
    if inputs.is_http_url(src):
        return src, False
    if inputs.is_data_uri(src) or inputs.looks_like_base64(src):
        return await inputs.materialize(src, WORK_DIR, suffix=".mp4")
    return inputs.local_path(src), False


async def _local_image(src: str) -> tuple[str, bool]:
    """Materialize any image input to a local file path for OpenCV/Pillow."""
    return await inputs.materialize(src, WORK_DIR)


def _optional(args: dict, key: str):
    value = args.get(key)
    return value if value not in (None, "") else None


# ----------------------------------------------------------------- VLM handlers

async def _analyze_image(args: dict) -> str:
    provider = _get_provider()
    preset = args.get("preset", "describe")
    prompt = args.get("prompt") or PROMPT_PRESETS.get(preset, PROMPT_PRESETS["describe"])
    image_url = await inputs.resolve_image_url(args["image"])
    parts = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    result = await _cached_analyze(provider, parts, args)
    out = result.to_dict()
    out["preset"] = preset if not args.get("prompt") else None
    return json.dumps(out, ensure_ascii=False)


async def _analyze_video(args: dict) -> str:
    provider = _get_provider()
    prompt = args.get("prompt") or (
        "请分析这段视频的内容：场景、主要对象、发生的动作与事件，并按时间顺序概述。"
    )
    max_frames = min(max(1, int(args.get("max_frames", MAX_FRAMES))), 32)
    mode = args.get("frame_mode", "count")
    interval = float(args.get("interval", 5.0))

    path, is_temp = await _prepare_video(args["video"])
    frame_dir = os.path.join(WORK_DIR, f"frames_{uuid.uuid4().hex}")
    try:
        frames = await ffmpeg.extract_frames(
            path, frame_dir, mode=mode, count=max_frames, interval=interval
        )
        frames = frames[:max_frames]
        parts: list[dict] = [
            {
                "type": "text",
                "text": prompt + f"\n\n以下是从视频中按时间抽取的 {len(frames)} 帧，每帧前标注了时间戳：",
            }
        ]
        for fr in frames:
            with open(fr["path"], "rb") as f:
                data = f.read()
            parts.append({"type": "text", "text": f"[{fr['label']}]"})
            parts.append({"type": "image_url", "image_url": {"url": inputs.to_data_uri(data, "jpeg")}})

        result = await _cached_analyze(provider, parts, args)
        out = result.to_dict()
        out["frame_mode"] = mode
        out["frames_used"] = [{"timestamp": fr["timestamp"], "label": fr["label"]} for fr in frames]
        return json.dumps(out, ensure_ascii=False)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)
        _cleanup_temp(path, is_temp)


async def _cached_analyze(provider, parts: list[dict], args: dict) -> VisionResult:
    max_tokens = _optional(args, "max_tokens")
    temperature = _optional(args, "temperature")
    if not CACHE_ENABLED:
        return await provider.analyze(
            parts,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            temperature=float(temperature) if temperature is not None else None,
        )

    key_src = json.dumps(
        {"model": MODEL, "parts": parts, "max_tokens": max_tokens, "temperature": temperature},
        ensure_ascii=False, sort_keys=True,
    )
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.isfile(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return VisionResult(**cached)
        except Exception:
            pass

    result = await provider.analyze(
        parts,
        max_tokens=int(max_tokens) if max_tokens is not None else None,
        temperature=float(temperature) if temperature is not None else None,
    )
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False)
    except Exception:
        pass
    return result


# --------------------------------------------------------------- local handlers

async def _video_info(args: dict) -> str:
    path, is_temp = await _prepare_video(args["video"])
    try:
        info = await ffmpeg.probe(path)
        return json.dumps(info, ensure_ascii=False)
    finally:
        _cleanup_temp(path, is_temp)


async def _extract_frames(args: dict) -> str:
    path, is_temp = await _prepare_video(args["video"])
    mode = args.get("mode", "count")
    count = max(1, int(args.get("count", 8)))
    interval = float(args.get("interval", 5.0))
    out_dir = os.path.join(WORK_DIR, f"frames_{uuid.uuid4().hex}")
    try:
        frames = await ffmpeg.extract_frames(path, out_dir, mode=mode, count=count, interval=interval)
        return json.dumps(
            {"count": len(frames), "mode": mode, "output_dir": out_dir, "frames": frames},
            ensure_ascii=False,
        )
    finally:
        _cleanup_temp(path, is_temp)


async def _extract_audio(args: dict) -> str:
    path, is_temp = await _prepare_video(args["video"])
    fmt = args.get("format", "mp3")
    out_path = os.path.join(WORK_DIR, f"audio_{uuid.uuid4().hex}.{fmt}")
    try:
        info = await ffmpeg.probe(path)
        if not info.get("has_audio"):
            return json.dumps({"error": "video has no audio stream", "path": args["video"]}, ensure_ascii=False)
        os.makedirs(WORK_DIR, exist_ok=True)
        result_path = await ffmpeg.extract_audio(path, out_path, fmt=fmt)
        return json.dumps({"audio_path": result_path, "format": fmt}, ensure_ascii=False)
    finally:
        _cleanup_temp(path, is_temp)


async def _scene_detect(args: dict) -> str:
    path, is_temp = await _prepare_video(args["video"])
    threshold = float(args.get("threshold", 0.4))
    try:
        times = await ffmpeg.scene_detect(path, threshold=threshold)
        scenes = [{"timestamp": t, "label": ffmpeg.format_ts(t)} for t in times]
        return json.dumps({"count": len(scenes), "threshold": threshold, "scenes": scenes}, ensure_ascii=False)
    finally:
        _cleanup_temp(path, is_temp)


async def _image_metadata(args: dict) -> str:
    path, is_temp = await _local_image(args["image"])
    try:
        info = await asyncio.to_thread(cv.image_metadata, path)
        return json.dumps(info, ensure_ascii=False)
    finally:
        _cleanup_temp(path, is_temp)


async def _detect_faces(args: dict) -> str:
    path, is_temp = await _local_image(args["image"])
    min_neighbors = int(args.get("min_neighbors", 5))
    try:
        result = await asyncio.to_thread(cv.detect_faces, path, min_neighbors=min_neighbors)
        return json.dumps(result, ensure_ascii=False)
    finally:
        _cleanup_temp(path, is_temp)


async def _compare_images(args: dict) -> str:
    path_a, temp_a = await _local_image(args["image_a"])
    path_b, temp_b = await _local_image(args["image_b"])
    try:
        result = await asyncio.to_thread(cv.compare_images, path_a, path_b)
        return json.dumps(result, ensure_ascii=False)
    finally:
        _cleanup_temp(path_a, temp_a)
        _cleanup_temp(path_b, temp_b)


def _server_status() -> str:
    available = _provider is not None and _provider.is_available()
    return json.dumps(
        {
            "provider": PROVIDER_NAME,
            "provider_available": available,
            "model": MODEL or None,
            "base_url_configured": bool(BASE_URL),
            "api_key_configured": bool(API_KEY),
            "ffmpeg_available": ffmpeg.ffmpeg_available(),
            "work_dir": WORK_DIR,
            "cache_enabled": CACHE_ENABLED,
            "max_frames": MAX_FRAMES,
            "max_tokens": MAX_TOKENS,
            "allowed_image_formats": sorted(inputs._allowed_formats()),
            "max_image_size": inputs._max_size(),
        },
        ensure_ascii=False,
    )


def _cleanup_temp(path: str, is_temp: bool) -> None:
    if is_temp and path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


# ----------------------------------------------------------------- entrypoints

async def main():
    """Run the MCP Multivision Server. Supports stdio and SSE transports."""
    _init_provider()
    os.makedirs(WORK_DIR, exist_ok=True)

    if _provider is not None and _provider.is_available():
        print(f"[mcp-multivision-server] Vision provider '{PROVIDER_NAME}' ready (model={MODEL}).", file=sys.stderr)
    else:
        print(
            "[mcp-multivision-server] Vision provider not configured. "
            "Local CV/video tools work; set MCP_VISION_BASE_URL/API_KEY/MODEL for analysis tools.",
            file=sys.stderr,
        )
    if not ffmpeg.ffmpeg_available():
        print("[mcp-multivision-server] WARNING: ffmpeg/ffprobe not found on PATH; video tools will fail.", file=sys.stderr)

    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "sse":
        await _run_sse()
    else:
        await _run_stdio()


async def _run_stdio():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


async def _run_sse():
    try:
        from starlette.applications import Starlette
        from starlette.responses import Response
        from starlette.routing import Mount, Route as StarletteRoute
        import uvicorn
    except ImportError:
        print(
            "[mcp-multivision-server] SSE transport requires starlette and uvicorn. "
            "Install with: pip install mcp-multivision-server[sse]",
            file=sys.stderr,
        )
        sys.exit(1)

    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8093"))

    transport_instance = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with transport_instance.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
        return Response()

    app = Starlette(
        routes=[
            StarletteRoute("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=transport_instance.handle_post_message),
        ]
    )

    print(f"[mcp-multivision-server] SSE server starting on http://{host}:{port}", file=sys.stderr)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


def run():
    """Synchronous entrypoint for the console script."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
