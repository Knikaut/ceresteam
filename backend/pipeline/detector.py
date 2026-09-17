"""Детекция техники и людей на кадре (YOLOv8, классы COCO)."""
from __future__ import annotations

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


def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(config.YOLO_MODEL)
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
    res = model.predict(img_bgr, imgsz=config.YOLO_IMGSZ, conf=conf, verbose=False)[0]
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
