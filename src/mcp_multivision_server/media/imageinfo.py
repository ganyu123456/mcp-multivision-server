"""Lightweight image metadata (Pillow only, no OpenCV / ffmpeg)."""

import os


class ImageInfoError(Exception):
    """Raised when image metadata cannot be read."""


def image_metadata(path: str) -> dict:
    """Return dimensions, format and EXIF metadata (incl. GPS if present)."""
    from PIL import ExifTags, Image

    try:
        with Image.open(path) as img:
            info = {
                "path": path,
                "format": img.format,
                "mode": img.mode,
                "width": img.width,
                "height": img.height,
                "size_bytes": os.path.getsize(path),
                "exif": {},
                "gps": {},
            }
            raw_exif = img.getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                    info["exif"][tag] = _exif_safe(value)
                gps_ifd = raw_exif.get_ifd(ExifTags.IFD.GPSInfo) if hasattr(ExifTags, "IFD") else {}
                for gps_id, value in (gps_ifd or {}).items():
                    tag = ExifTags.GPSTAGS.get(gps_id, str(gps_id))
                    info["gps"][tag] = _exif_safe(value)
    except Exception as e:
        raise ImageInfoError(f"failed to read image metadata: {e}")
    return info


def _exif_safe(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")[:200]
    if isinstance(value, (tuple, list)):
        return [_exif_safe(v) for v in value]
    try:
        import json

        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)
