"""Отрисовка рамок и подписей на кадре (PIL — чтобы работала кириллица)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = {
    "vehicle": (220, 40, 40),
    "license_plate": (255, 200, 0),
    "person": (40, 200, 60),
}

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def imread(path: str | Path) -> np.ndarray | None:
    """cv2.imread не понимает кириллицу в путях на Windows — читаем через буфер."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: str | Path, img: np.ndarray, quality: int = 90) -> None:
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError(f"Не удалось закодировать изображение {path}")
    buf.tofile(str(path))


def _font(size: int) -> ImageFont.ImageFont:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_detections(img_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    w = img.width
    thick = max(2, w // 700)
    font = _font(max(18, w // 90))

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = COLORS.get(det["category"], (200, 200, 200))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thick)
        label = det.get("label") or det["category"]
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        ty = y1 - th - 6 if y1 - th - 6 > 0 else y2 + 4
        draw.rectangle([x1, ty, x1 + tw + 10, ty + th + 6], fill=color)
        draw.text((x1 + 5, ty + 2), label, fill=(0, 0, 0) if sum(color) > 400 else (255, 255, 255), font=font)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def crop_with_margin(img_bgr: np.ndarray, bbox: list[int], margin: float = 0.3, min_side: int = 0) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    crop = img_bgr[y1:y2, x1:x2]
    if min_side and min(crop.shape[:2]) < min_side:
        scale = min_side / max(1, min(crop.shape[:2]))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop
