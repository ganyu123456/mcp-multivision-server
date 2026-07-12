"""Unified media input handling.

Accepts four input forms for images and videos:
  1. absolute local path        /path/to/img.jpg
  2. file:// URL                 file:///path/to/img.jpg
  3. http(s):// URL              https://example.com/img.jpg
  4. data URI / bare base64      data:image/png;base64,iVBOR...  (or the raw base64)

Everything is resolved to a URL suitable for an OpenAI-compatible ``image_url``
or DashScope ``video_url`` field: http(s) URLs pass through (the model fetches
them), local files / base64 become base64 data URIs. No local media processing.
"""

import base64
import binascii
import os
import re
from io import BytesIO
from typing import Optional
from urllib.parse import unquote, urlparse

import httpx

_DEFAULT_MAX_SIZE = 20 * 1024 * 1024
_DEFAULT_MAX_VIDEO_SIZE = 100 * 1024 * 1024
_DEFAULT_ALLOWED = "jpeg,png,webp,gif,bmp,tiff"
_DATA_URI_RE = re.compile(r"^data:(?P<mime>[\w./+-]+)?;base64,(?P<data>.+)$", re.DOTALL)
_VIDEO_EXT_MIME = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".flv": "video/x-flv",
    ".wmv": "video/x-ms-wmv",
}


class InputError(Exception):
    """Raised when an image/video input cannot be read or fails validation."""


def _max_size() -> int:
    try:
        return int(os.getenv("MCP_VISION_MAX_IMAGE_SIZE", str(_DEFAULT_MAX_SIZE)))
    except ValueError:
        return _DEFAULT_MAX_SIZE


def _max_video_size() -> int:
    try:
        return int(os.getenv("MCP_VISION_MAX_VIDEO_SIZE", str(_DEFAULT_MAX_VIDEO_SIZE)))
    except ValueError:
        return _DEFAULT_MAX_VIDEO_SIZE


def _allowed_formats() -> set[str]:
    raw = os.getenv("MCP_VISION_ALLOWED_IMAGE_FORMATS", _DEFAULT_ALLOWED)
    return {f.strip().lower() for f in raw.split(",") if f.strip()}


def is_http_url(src: str) -> bool:
    return src.startswith("http://") or src.startswith("https://")


def is_data_uri(src: str) -> bool:
    return src.startswith("data:")


def looks_like_base64(src: str) -> bool:
    if len(src) < 32:
        return False
    return re.fullmatch(r"[A-Za-z0-9+/=\s]+", src) is not None


def _normalize_path(src: str) -> str:
    if src.startswith("file://"):
        parsed = urlparse(src)
        path = unquote(parsed.path)
    else:
        path = src
    if "\x00" in path:
        raise InputError("path contains a null byte")
    return os.path.abspath(os.path.expanduser(path))


def existing_path(src: str) -> Optional[str]:
    """Return a normalized path if src refers to an existing file, else None."""
    if src.startswith("data:"):
        return None
    try:
        path = _normalize_path(src)
    except InputError:
        return None
    return path if os.path.isfile(path) else None


def local_path(src: str) -> str:
    """Normalize a path / file:// URL to a validated local file path."""
    path = _normalize_path(src)
    if not os.path.isfile(path):
        raise InputError(f"file not found: {path}")
    return path


def check_size(data: bytes) -> None:
    limit = _max_size()
    if len(data) > limit:
        raise InputError(f"input exceeds MCP_VISION_MAX_IMAGE_SIZE ({len(data)} > {limit} bytes)")


def sniff_image_format(data: bytes) -> str:
    """Return a lowercase image format (e.g. 'jpeg') using Pillow; validates it is an image."""
    from PIL import Image

    try:
        with Image.open(BytesIO(data)) as img:
            fmt = (img.format or "").lower()
    except Exception as e:
        raise InputError(f"not a readable image: {e}")
    if fmt == "jpg":
        fmt = "jpeg"
    if not fmt:
        raise InputError("could not determine image format")
    return fmt


def check_allowed_format(fmt: str) -> None:
    allowed = _allowed_formats()
    if fmt not in allowed:
        raise InputError(f"image format '{fmt}' not allowed (allowed: {sorted(allowed)})")


def decode_base64(src: str) -> bytes:
    payload = src
    m = _DATA_URI_RE.match(src)
    if m:
        payload = m.group("data")
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as e:
        raise InputError(f"invalid base64 data: {e}")


async def fetch_bytes(url: str, *, timeout: float = 30.0) -> bytes:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url)
    if resp.status_code != 200:
        raise InputError(f"failed to fetch {url}: HTTP {resp.status_code}")
    return resp.content


def to_data_uri(data: bytes, fmt: str) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/{fmt};base64,{b64}"


async def load_image_bytes(src: str) -> tuple[bytes, str]:
    """Load an image input into (bytes, format), validating size and format."""
    src = src.strip()
    if is_http_url(src):
        data = await fetch_bytes(src)
    elif is_data_uri(src):
        data = decode_base64(src)
    else:
        path = existing_path(src)
        if path:
            with open(path, "rb") as f:
                data = f.read()
        elif looks_like_base64(src):
            data = decode_base64(src)
        else:
            raise InputError(f"file not found: {_normalize_path(src)}")
    check_size(data)
    fmt = sniff_image_format(data)
    check_allowed_format(fmt)
    return data, fmt


async def resolve_image_url(src: str) -> str:
    """Return a value suitable for a VLM ``image_url.url`` field.

    http(s) URLs pass through (the model fetches them); everything else is
    validated locally and converted to a base64 data URI.
    """
    src = src.strip()
    if is_http_url(src):
        return src
    data, fmt = await load_image_bytes(src)
    return to_data_uri(data, fmt)


async def resolve_video_url(src: str) -> str:
    """Return a value suitable for a DashScope ``video_url.url`` field.

    http(s) URLs and data URIs pass through; local files / bare base64 become a
    ``data:video/<type>;base64,...`` URI. The cloud model does frame sampling.
    """
    src = src.strip()
    if is_http_url(src) or is_data_uri(src):
        return src

    path = existing_path(src)
    if path:
        size = os.path.getsize(path)
        limit = _max_video_size()
        if size > limit:
            raise InputError(
                f"video exceeds MCP_VISION_MAX_VIDEO_SIZE ({size} > {limit} bytes); "
                "pass an http(s) URL instead of a local file for large videos"
            )
        ext = os.path.splitext(path)[1].lower()
        mime = _VIDEO_EXT_MIME.get(ext, "video/mp4")
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"

    if looks_like_base64(src):
        return f"data:video/mp4;base64,{src}"

    raise InputError(f"file not found: {_normalize_path(src)}")
