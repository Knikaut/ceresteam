"""Детекция техники и людей на кадре (YOLOv8, классы COCO)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from backend import config

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
PERSON_CLASS = "person"


@dataclass
class Detection:
    cls: str
    conf: float
    bbox: list[int]  # x1, y1, x2, y2

    @property
    def area(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1])


_model = None
_device: str | None = None


def resolve_device(name: str | None = None) -> str:
    """Куда класть модель: "auto" — видеокарта, если она есть, иначе процессор."""
    name = (name or os.environ.get("YOLO_DEVICE") or "auto").strip().lower()
    if name != "auto":
        return name
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():  # Apple GPU
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_device() -> str:
    global _device
    if _device is None:
        _device = resolve_device()
    return _device


def set_device(name: str | None) -> str:
    """Задать устройство до первого кадра (из run_batch.py)."""
    global _device, _model
    new = resolve_device(name)
    if new != _device:
        _device, _model = new, None  # модель уже лежит на прежнем устройстве
    return _device


def _fallback_to_cpu(err: Exception) -> None:
    global _device, _model
    print(f"Видеокарта ({_device}) недоступна ({type(err).__name__}: {err}) — считаем на процессоре")
    _device, _model = "cpu", None


def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        config.limit_cpu_threads()
        model = YOLO(config.YOLO_MODEL)
        try:
            model.to(get_device())
        except Exception as e:
            _fallback_to_cpu(e)
            model.to("cpu")
        _model = model
    return _model


def _iou(a: list[int], b: list[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _contained(inner: list[int], outer: list[int]) -> float:
    """Какая доля inner лежит внутри outer."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / float(area) if area else 0.0


def merge_vehicles(dets: list[Detection]) -> list[Detection]:
    """Тягач и прицеп/кузов YOLO часто выдаёт двумя рамками — сливаем пересекающиеся."""
    vehicles = sorted([d for d in dets if d.cls in VEHICLE_CLASSES], key=lambda d: -d.area)
    merged: list[Detection] = []
    for d in vehicles:
        for m in merged:
            if _iou(d.bbox, m.bbox) > 0.4 or _contained(d.bbox, m.bbox) > 0.7:
                m.bbox = [min(m.bbox[0], d.bbox[0]), min(m.bbox[1], d.bbox[1]),
                          max(m.bbox[2], d.bbox[2]), max(m.bbox[3], d.bbox[3])]
                m.conf = max(m.conf, d.conf)
                if d.cls == "truck":
                    m.cls = "truck"
                break
        else:
            merged.append(Detection(d.cls, d.conf, list(d.bbox)))
    return merged


def detect(img_bgr: np.ndarray, conf: float = 0.25) -> list[Detection]:
    model = get_model()
    try:
        res = model.predict(img_bgr, imgsz=config.YOLO_IMGSZ, conf=conf, device=get_device(), verbose=False)[0]
    except Exception as e:
        if get_device() == "cpu":
            raise
        # Видеокарта отвалилась на середине прогона — доканчиваем на процессоре, а не падаем.
        _fallback_to_cpu(e)
        res = get_model().predict(img_bgr, imgsz=config.YOLO_IMGSZ, conf=conf, device="cpu", verbose=False)[0]
    names = res.names
    dets: list[Detection] = []
    for box in res.boxes:
        cls = names[int(box.cls[0])]
        if cls not in VEHICLE_CLASSES and cls != PERSON_CLASS:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        dets.append(Detection(cls, float(box.conf[0]), [x1, y1, x2, y2]))
    people = [d for d in dets if d.cls == PERSON_CLASS]
    return merge_vehicles(dets) + people
