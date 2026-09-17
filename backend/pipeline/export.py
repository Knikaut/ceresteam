"""Сохранение отчётов в JSON и CSV."""
from __future__ import annotations

import csv
import json
import os
import textwrap
from collections.abc import Iterable
from pathlib import Path

CSV_FIELDS = [
    "source_image", "timestamp_on_frame", "id", "category", "on_scale", "equipment_type",
    "manufacturer", "manufacturer_country", "model", "year", "make_model_source", "plate_number", "plate_format", "plate_country",
    "plate_region_code", "color", "load_state", "has_trailer",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "detection_confidence", "notes",
]


def detection_rows(report: dict) -> list[dict]:
    rows = []
    for d in report["detections"]:
        plate = d.get("license_plate") or {}
        app = d.get("appearance") or {}
        bb = d["bounding_box_px"]
        rows.append({
            "source_image": report["source_image"],
            "timestamp_on_frame": report.get("timestamp_on_frame"),
            "id": d["id"],
            "category": d["category"],
            "on_scale": d.get("on_scale", ""),
            "equipment_type": d.get("equipment_type"),
            "manufacturer": d.get("manufacturer"),
            "manufacturer_country": d.get("manufacturer_country"),
            "model": d.get("model"),
            "year": d.get("year"),
            "make_model_source": d.get("manufacturer_confidence"),
            "plate_number": plate.get("formatted"),
            "plate_format": plate.get("format"),
            "plate_country": plate.get("country"),
            "plate_region_code": plate.get("region_code"),
            "color": app.get("color"),
            "load_state": app.get("load_state"),
            "has_trailer": app.get("has_trailer"),
            "bbox_x1": bb["x1"], "bbox_y1": bb["y1"], "bbox_x2": bb["x2"], "bbox_y2": bb["y2"],
            "detection_confidence": d.get("detection_confidence"),
            "notes": d.get("notes"),
        })
    return rows


def _tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def write_json(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Пишем через временный файл: обрыв на середине не оставит обрезанный отчёт,
    # иначе пакетный прогон примет его за посчитанный кадр и не пересчитает.
    tmp = _tmp_path(path)
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_json_stream(reports: Iterable[dict], path: str | Path) -> Path:
    """{"images": [...]} по одному отчёту — память не зависит от числа кадров."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with tmp.open("w", encoding="utf-8") as f:
        f.write('{\n  "images": [\n')
        first = True
        for report in reports:
            if not first:
                f.write(",\n")
            first = False
            f.write(textwrap.indent(json.dumps(report, ensure_ascii=False, indent=2), "    "))
        f.write("\n  ]\n}\n")
    os.replace(tmp, path)
    return path


def write_csv(reports: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in reports:
            writer.writerows(detection_rows(r))
    os.replace(tmp, path)
    return path
