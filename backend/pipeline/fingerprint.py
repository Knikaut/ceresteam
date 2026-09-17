"""Отпечаток внешности машины — цветовая гистограмма вырезки.

Нужен, чтобы узнать машину, когда номер не читается: цвет кабины и кузова виден
и спереди, и сзади. Это не нейросетевой эмбеддинг, а простая и объяснимая мера.
"""
from __future__ import annotations

import cv2
import numpy as np

BINS = (12, 6, 4)  # H, S, V


def compute(crop_bgr: np.ndarray) -> list[float]:
    h, w = crop_bgr.shape[:2]
    # Центральная часть вырезки: меньше фона и дороги.
    crop = crop_bgr[int(h * 0.1):int(h * 0.85), int(w * 0.1):int(w * 0.9)]
    if crop.size == 0:
        crop = crop_bgr
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(BINS), [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return [round(float(v), 5) for v in hist]


def similarity(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    ha = np.array(a, dtype=np.float32)
    hb = np.array(b, dtype=np.float32)
    return float(max(0.0, cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)))
