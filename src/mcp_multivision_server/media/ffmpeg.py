"""ffmpeg / ffprobe helpers for local video preprocessing.

All calls shell out to the ffmpeg/ffprobe binaries (path from MCP_FFMPEG_BIN /
MCP_FFPROBE_BIN or PATH) via asyncio subprocesses, so nothing blocks the loop.
"""

import asyncio
import json
import os
import re


class FFmpegError(Exception):
    """Raised when an ffmpeg/ffprobe invocation fails."""


def ffmpeg_bin() -> str:
    return os.getenv("MCP_FFMPEG_BIN") or "ffmpeg"


def ffprobe_bin() -> str:
    return os.getenv("MCP_FFPROBE_BIN") or "ffprobe"


def ffmpeg_available() -> bool:
    from shutil import which

    return which(ffmpeg_bin()) is not None and which(ffprobe_bin()) is not None


def format_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


async def _run(cmd: list[str]) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout, stderr


def _parse_fps(rate: str) -> float:
    try:
        if "/" in rate:
            num, den = rate.split("/")
            den = float(den)
            return round(float(num) / den, 3) if den else 0.0
        return round(float(rate), 3)
    except (ValueError, ZeroDivisionError):
        return 0.0


async def probe(path: str) -> dict:
    """Return a cleaned summary of a media file's format and streams."""
    cmd = [
        ffprobe_bin(), "-v", "error",
        "-show_format", "-show_streams",
        "-of", "json", path,
    ]
    rc, stdout, stderr = await _run(cmd)
    if rc != 0:
        raise FFmpegError(f"ffprobe failed: {stderr.decode('utf-8', 'replace')[:300]}")

    data = json.loads(stdout or b"{}")
    fmt = data.get("format", {})
    video, audio = None, None
    for stream in data.get("streams", []):
        kind = stream.get("codec_type")
        if kind == "video" and video is None:
            video = {
                "codec": stream.get("codec_name", ""),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "fps": _parse_fps(stream.get("r_frame_rate", "0/0")),
                "nb_frames": stream.get("nb_frames"),
                "duration": _to_float(stream.get("duration")),
            }
        elif kind == "audio" and audio is None:
            audio = {
                "codec": stream.get("codec_name", ""),
                "sample_rate": stream.get("sample_rate"),
                "channels": stream.get("channels"),
            }

    return {
        "path": path,
        "duration": _to_float(fmt.get("duration")),
        "size_bytes": _to_int(fmt.get("size")),
        "format_name": fmt.get("format_name", ""),
        "bit_rate": _to_int(fmt.get("bit_rate")),
        "video": video,
        "audio": audio,
        "has_audio": audio is not None,
    }


async def _duration(path: str) -> float:
    info = await probe(path)
    dur = info.get("duration") or (info.get("video") or {}).get("duration") or 0.0
    return float(dur or 0.0)


def _plan_timestamps(duration: float, mode: str, count: int, interval: float) -> list[float]:
    if duration <= 0:
        return [0.0]
    if mode == "interval":
        step = max(0.1, float(interval))
        ts, t = [], 0.0
        while t < duration and len(ts) < 600:
            ts.append(round(t, 3))
            t += step
        return ts or [0.0]
    # mode == "count": evenly spaced, avoiding the very first/last frames
    n = max(1, int(count))
    if n == 1:
        return [round(duration / 2, 3)]
    start = duration * 0.02
    end = duration * 0.90
    return [round(start + (end - start) * i / (n - 1), 3) for i in range(n)]


async def extract_frames(
    path: str,
    out_dir: str,
    *,
    mode: str = "count",
    count: int = 8,
    interval: float = 5.0,
) -> list[dict]:
    """Extract frames as JPEGs. Returns [{path, timestamp, label}] ordered by time."""
    os.makedirs(out_dir, exist_ok=True)

    if mode == "keyframe":
        timestamps = await _keyframe_timestamps(path, limit=max(1, count))
        if not timestamps:
            timestamps = _plan_timestamps(await _duration(path), "count", count, interval)
    else:
        timestamps = _plan_timestamps(await _duration(path), mode, count, interval)

    frames = []
    for idx, ts in enumerate(timestamps):
        out_path = os.path.join(out_dir, f"frame_{idx:04d}.jpg")
        cmd = [
            ffmpeg_bin(), "-y", "-ss", f"{ts}", "-i", path,
            "-frames:v", "1", "-q:v", "2", out_path,
        ]
        rc, _, stderr = await _run(cmd)
        if rc != 0 or not os.path.isfile(out_path):
            continue
        frames.append({"path": out_path, "timestamp": ts, "label": format_ts(ts)})
    if not frames:
        raise FFmpegError("no frames could be extracted")
    return frames


async def _keyframe_timestamps(path: str, *, limit: int) -> list[float]:
    cmd = [
        ffprobe_bin(), "-v", "error",
        "-select_streams", "v:0",
        "-skip_frame", "nokey",
        "-show_entries", "frame=pts_time",
        "-of", "json", path,
    ]
    rc, stdout, _ = await _run(cmd)
    if rc != 0:
        return []
    try:
        frames = json.loads(stdout or b"{}").get("frames", [])
        times = sorted({round(float(f["pts_time"]), 3) for f in frames if f.get("pts_time")})
    except (ValueError, KeyError):
        return []
    if len(times) <= limit:
        return times
    # 均匀下采样到 limit 个
    step = len(times) / limit
    return [times[int(i * step)] for i in range(limit)]


async def extract_audio(path: str, out_path: str, *, fmt: str = "mp3") -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if fmt == "wav":
        codec = ["-acodec", "pcm_s16le"]
    else:
        codec = ["-acodec", "libmp3lame", "-q:a", "2"]
    cmd = [ffmpeg_bin(), "-y", "-i", path, "-vn", *codec, out_path]
    rc, _, stderr = await _run(cmd)
    if rc != 0 or not os.path.isfile(out_path):
        raise FFmpegError(f"audio extraction failed: {stderr.decode('utf-8', 'replace')[:300]}")
    return out_path


_SHOWINFO_PTS = re.compile(r"pts_time:([0-9.]+)")


async def scene_detect(path: str, *, threshold: float = 0.4) -> list[float]:
    """Return timestamps (seconds) where a scene change above threshold occurs."""
    threshold = min(1.0, max(0.0, float(threshold)))
    cmd = [
        ffmpeg_bin(), "-i", path,
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]
    rc, _, stderr = await _run(cmd)
    if rc != 0:
        raise FFmpegError(f"scene detection failed: {stderr.decode('utf-8', 'replace')[:300]}")
    text = stderr.decode("utf-8", "replace")
    times = [round(float(m), 3) for m in _SHOWINFO_PTS.findall(text)]
    return sorted(set(times))


def _to_float(value):
    try:
        return round(float(value), 3) if value is not None else None
    except (ValueError, TypeError):
        return None


def _to_int(value):
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None
