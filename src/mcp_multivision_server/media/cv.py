"""Local computer-vision utilities (Pillow + OpenCV), no cloud calls.

All functions are blocking and are meant to be run via ``asyncio.to_thread``.
"""

import os


class CVError(Exception):
    """Raised when a local CV operation fails."""


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
        raise CVError(f"failed to read image metadata: {e}")
    return info


def detect_faces(path: str, *, scale_factor: float = 1.1, min_neighbors: int = 5) -> dict:
    """Detect frontal faces with OpenCV's bundled Haar cascade (no model download)."""
    import cv2

    image = cv2.imread(path)
    if image is None:
        raise CVError(f"could not read image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise CVError("failed to load Haar cascade")
    faces = cascade.detectMultiScale(
        gray, scaleFactor=max(1.01, scale_factor), minNeighbors=max(1, min_neighbors)
    )
    boxes = [
        {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        for (x, y, w, h) in faces
    ]
    h, w = gray.shape[:2]
    return {
        "path": path,
        "image_width": int(w),
        "image_height": int(h),
        "face_count": len(boxes),
        "faces": boxes,
    }


def compare_images(path_a: str, path_b: str) -> dict:
    """Compare two images via perceptual hashing + colour histogram correlation."""
    import cv2
    import numpy as np

    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)
    if img_a is None:
        raise CVError(f"could not read image: {path_a}")
    if img_b is None:
        raise CVError(f"could not read image: {path_b}")

    ahash_a, ahash_b = _ahash(img_a, cv2, np), _ahash(img_b, cv2, np)
    dhash_a, dhash_b = _dhash(img_a, cv2, np), _dhash(img_b, cv2, np)
    ahash_sim = 1.0 - _hamming(ahash_a, ahash_b) / ahash_a.size
    dhash_sim = 1.0 - _hamming(dhash_a, dhash_b) / dhash_a.size
    hist_sim = _hist_correlation(img_a, img_b, cv2, np)

    overall = round((ahash_sim + dhash_sim + hist_sim) / 3, 4)
    return {
        "similarity": overall,
        "ahash_similarity": round(float(ahash_sim), 4),
        "dhash_similarity": round(float(dhash_sim), 4),
        "histogram_similarity": round(float(hist_sim), 4),
        "identical": overall >= 0.999,
    }


def _ahash(image, cv2, np, size: int = 8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return small > small.mean()


def _dhash(image, cv2, np, size: int = 8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return small[:, 1:] > small[:, :-1]


def _hamming(a, b) -> int:
    import numpy as np

    return int(np.count_nonzero(a != b))


def _hist_correlation(img_a, img_b, cv2, np) -> float:
    def hist(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
        return h

    corr = cv2.compareHist(hist(img_a), hist(img_b), cv2.HISTCMP_CORREL)
    return max(0.0, min(1.0, float(corr)))


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
