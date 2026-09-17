"""Полный анализ одного кадра: техника -> номер -> марка/модель -> отчёт.

Результат — dict: отчёт по кадру (техника, номера, люди, сводка); поля для CSV см. export.CSV_FIELDS.
"""
from __future__ import annotations

import time
from pathlib import Path

from backend.pipeline import annotate, detector, makes, plates, registry, vlm

TYPE_BY_CLASS = {
    "truck": "Грузовой автомобиль",
    "car": "Легковой автомобиль",
    "bus": "Автобус",
    "motorcycle": "Мотоцикл",
}

# Самая крупная техника считается стоящей на весах, если занимает хотя бы такую долю кадра.
MAIN_AREA_FRACTION = 0.004
# Остальная техника считается "на площадке", если сопоставима по размеру с главной.
SECONDARY_AREA_RATIO = 0.4

# Надёжность прочтения номера: согласие разных прочтений + уверенность OCR - штрафы.
# Веса и пороги подобраны перебором по эталонной разметке 25 кадров весовой, а не на глаз.
PLATE_W_VOTES = 0.5
PLATE_W_CONF = 0.5
# Несуществующий код региона — почти всегда ошибка чтения, а не редкий номер.
PLATE_BAD_REGION_PENALTY = 0.8
# Каждый символ, исправленный при подгонке под шаблон, удешевляет прочтение.
PLATE_FIX_PENALTY = 0.2
# Ниже порога честнее сказать «номер не прочитан», чем показать выдуманный.
PLATE_READABLE_MIN = 0.55
PLATE_SURE_MIN = 0.75


def plate_reliability(p) -> float:
    return (PLATE_W_VOTES * p.votes + PLATE_W_CONF * p.confidence
            - (0.0 if p.region_valid else PLATE_BAD_REGION_PENALTY)
            - PLATE_FIX_PENALTY * p.fixes)


def _fmt_conf(value: float) -> str:
    """Словесная оценка. Границы совпадают с порогом читаемости: показанный номер не бывает «низкой» уверенности."""
    return ("высокая" if value >= PLATE_SURE_MIN else
            "средняя" if value >= PLATE_READABLE_MIN else "низкая")


def set_country(entry: dict) -> None:
    """Страна производителя: по справочнику марок, иначе оценка VLM."""
    country = makes.country_for(entry.get("manufacturer")) or entry.get("vlm_country")
    entry["manufacturer_country"] = country


def apply_registry(entry: dict, record: dict | None) -> None:
    """Подставляет марку/модель/год из реестра data.egov.kz в запись о машине."""
    if not record:
        return
    entry["registry"] = record
    if record.get("manufacturer"):
        if entry.get("manufacturer") and not makes.same_make(entry["manufacturer"], record["manufacturer"]):
            entry["notes"] += (f"Надпись на технике: {entry['manufacturer']}, реестр: {record['manufacturer']}. ")
            if not record.get("model"):
                # Модель относилась к другой марке — сбрасываем.
                entry["model"] = "не определена"
                entry["model_confidence"] = None
        entry["manufacturer"] = record["manufacturer"]
        entry["manufacturer_confidence"] = "высокая (реестр data.egov.kz по госномеру)"
    if record.get("model"):
        entry["model"] = record["model"]
        entry["model_confidence"] = "высокая (реестр data.egov.kz по госномеру)"
    if record.get("year"):
        entry["year"] = record["year"]
    if record.get("type") and entry.get("equipment_type") in (None, "Техника", "Грузовой автомобиль"):
        entry["equipment_type"] = record["type"].capitalize()
    set_country(entry)


def analyze_image(image_path: str | Path, use_vlm: bool = True, use_registry: bool = True) -> dict:
    image_path = Path(image_path)
    img = annotate.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось открыть изображение: {image_path}")
    h, w = img.shape[:2]
    t0 = time.time()

    dets = detector.detect(img)
    vehicles = sorted([d for d in dets if d.cls in detector.VEHICLE_CLASSES], key=lambda d: -d.area)
    people = [d for d in dets if d.cls == detector.PERSON_CLASS]

    detections: list[dict] = []
    next_id = 1
    main_area = vehicles[0].area if vehicles else 0

    for idx, v in enumerate(vehicles):
        if idx == 0:
            is_main = v.area >= MAIN_AREA_FRACTION * w * h
        else:
            is_main = v.area >= max(SECONDARY_AREA_RATIO * main_area, 0.01 * w * h)
        vehicle_id = next_id
        next_id += 1

        entry = {
            "id": vehicle_id,
            "category": "vehicle",
            "on_scale": is_main,
            "yolo_class": v.cls,
            "equipment_type": TYPE_BY_CLASS.get(v.cls, "Техника"),
            "manufacturer": None,
            "model": None,
            "manufacturer_confidence": None,
            "model_confidence": None,
            "license_plate": None,
            "bounding_box_px": {"x1": v.bbox[0], "y1": v.bbox[1], "x2": v.bbox[2], "y2": v.bbox[3]},
            "detection_confidence": round(v.conf, 3),
            "appearance": None,
            "notes": "",
        }

        plate_entries: list[dict] = []
        if is_main:
            # OCR по увеличенной вырезке техники: номер + надписи (марка).
            x1, y1, x2, y2 = v.bbox
            crop = img[y1:y2, x1:x2]
            found, boxes = plates.read_vehicle(crop, offset=(x1, y1))
            brand_text = plates.brand_from_boxes(boxes)
            if brand_text:
                entry["manufacturer"] = brand_text
                entry["manufacturer_confidence"] = "высокая (надпись прочитана на технике)"

            if found:
                best = found[0]
                reliability = plate_reliability(best)
                readable = reliability >= PLATE_READABLE_MIN
                entry["license_plate"] = {
                    # Ненадёжное прочтение не выдаём за номер: выдуманный номер хуже пустого поля.
                    "text": best.text if readable else None,
                    "formatted": best.formatted if readable else None,
                    "country": "Казахстан (KZ)",
                    "region_code": best.region_code if readable else None,
                    "format": best.kind,
                    "readable": readable,
                    "confidence": _fmt_conf(reliability),
                    "ocr_confidence": round(best.confidence, 3),
                    "agreement": round(best.votes, 2),
                    "alternatives": [p.formatted for p in found[1:4]] if readable else [],
                }
                if not readable:
                    entry["license_plate"]["notes"] = ("Номер мелкий/нечёткий: прочтение ненадёжное, "
                                                       "номер должен ввести охранник")
                    # Догадку OCR прячем в отдельное поле — как подсказку, а не как результат.
                    entry["license_plate"]["ocr_guess"] = {"text": best.text, "formatted": best.formatted,
                                                           "reliability": round(reliability, 2)}
                # Рамка рисуется только для надёжного прочтения; остальные варианты — в alternatives.
                for p in found[:1] if readable else []:
                    plate_entries.append({
                        "id": None,
                        "category": "license_plate",
                        "equipment_type": ("Государственный регистрационный знак" if p.kind != "painted"
                                           else "Номер, нанесённый краской на борт") + f" транспортного средства id={vehicle_id}",
                        "manufacturer": None,
                        "model": None,
                        "license_plate": {"text": p.text, "formatted": p.formatted, "country": "Казахстан (KZ)",
                                          "region_code": p.region_code, "format": p.kind},
                        "bounding_box_px": {"x1": p.bbox[0], "y1": p.bbox[1], "x2": p.bbox[2], "y2": p.bbox[3]},
                        "detection_confidence": round(p.confidence, 3),
                        "readable": True,
                        "vehicle_id": vehicle_id,
                    })
            else:
                entry["license_plate"] = {"text": None, "readable": False,
                                          "confidence": None, "notes": "Номер не найден или не читается"}

            # Марка/модель по приоритету: реестр по номеру -> надпись -> VLM.
            if use_registry and entry["license_plate"].get("readable"):
                status, record = registry.lookup_status(entry["license_plate"]["text"])
                entry["registry_status"] = status
                apply_registry(entry, record)

            if use_vlm and vlm.enabled():
                hints = f"надпись на технике: {brand_text}" if brand_text else ""
                desc = vlm.describe_vehicle(annotate.crop_with_margin(img, v.bbox, margin=0.15), hints)
                if desc:
                    entry["equipment_type"] = desc.get("equipment_type") or entry["equipment_type"]
                    known = entry.get("manufacturer")
                    if desc.get("manufacturer") and not known:
                        entry["manufacturer"] = desc["manufacturer"]
                        entry["manufacturer_confidence"] = desc.get("manufacturer_confidence")
                    elif desc.get("manufacturer") and known and not makes.same_make(desc["manufacturer"], known):
                        entry["notes"] += f"VLM считает производителем {desc['manufacturer']}, по данным реестра/надписи — {known}. "
                    if desc.get("model") and not entry.get("model"):
                        # Модель VLM берём, только если она про ту же марку, иначе получится «МАЗ Actros».
                        if not known or makes.same_make(desc.get("manufacturer"), known):
                            entry["model"] = desc.get("model")
                            entry["model_confidence"] = desc.get("model_confidence")
                        else:
                            entry["notes"] += f"Модель по VLM: {desc['model']} (для марки {desc.get('manufacturer') or '—'}). "
                    if desc.get("country"):
                        entry["vlm_country"] = desc["country"]
                    entry["appearance"] = {
                        "color": desc.get("color"),
                        "has_trailer": desc.get("has_trailer"),
                        "load_state": desc.get("load_state"),
                        "cargo": desc.get("cargo"),
                        "view": desc.get("view"),
                        "description": desc.get("appearance"),
                        "source": desc.get("_model"),
                    }
            if entry["model"] is None:
                entry["model"] = "не определена"
            set_country(entry)
        else:
            entry["notes"] = "Техника на заднем плане, вне весовой платформы."

        detections.append(entry)
        for p in plate_entries:
            p["id"] = next_id
            next_id += 1
            detections.append(p)

    for p in people:
        detections.append({
            "id": next_id,
            "category": "person",
            "equipment_type": "Человек (не техника, сотрудник весовой/водитель)",
            "manufacturer": None,
            "model": None,
            "license_plate": None,
            "bounding_box_px": {"x1": p.bbox[0], "y1": p.bbox[1], "x2": p.bbox[2], "y2": p.bbox[3]},
            "detection_confidence": round(p.conf, 3),
            "notes": "Включён для полноты сцены.",
        })
        next_id += 1

    timestamp = plates.read_overlay_timestamp(img)

    main = next((d for d in detections if d["category"] == "vehicle" and d["on_scale"]), None)
    return {
        "source_image": image_path.name,
        "image_resolution": {"width": w, "height": h},
        "camera_id": "Camera 01",
        "timestamp_on_frame": timestamp,
        "note_on_timestamp": "Дата/время прочитаны с наложенной надписи камеры и могут отличаться от реального времени.",
        "processing_seconds": round(time.time() - t0, 2),
        "detections": detections,
        "summary": {
            "total_objects_detected": len(detections),
            "vehicles_or_equipment": sum(1 for d in detections if d["category"] == "vehicle"),
            "vehicles_on_scale": sum(1 for d in detections if d["category"] == "vehicle" and d["on_scale"]),
            "people": len(people),
            "plates_read": sum(1 for d in detections if d["category"] == "license_plate"),
            "main_vehicle_id": main["id"] if main else None,
        },
    }


def label_for(det: dict) -> str:
    cat = det["category"]
    if cat == "vehicle":
        parts = [f"{det['id']}:"]
        if det.get("manufacturer"):
            parts.append(det["manufacturer"])
        parts.append(det.get("equipment_type") or "техника")
        if not det.get("on_scale"):
            parts.append("(фон)")
        return " ".join(parts)
    if cat == "license_plate":
        formatted = (det.get("license_plate") or {}).get("formatted")
        if not formatted:
            return f"{det['id']}: номер не прочитан"
        return f"{det['id']}: номер {formatted}"
    if cat == "person":
        return f"{det['id']}: человек"
    return f"{det['id']}: {det.get('equipment_type', cat)}"


def render_outputs(image_path: str | Path, report: dict, out_dir: str | Path) -> dict:
    """Сохраняет размеченный кадр и увеличенный номер. Возвращает пути."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = annotate.imread(image_path)
    stem = Path(image_path).stem

    drawn = [{"bbox": [d["bounding_box_px"][k] for k in ("x1", "y1", "x2", "y2")],
              "category": d["category"], "label": label_for(d)} for d in report["detections"]]
    annotated = annotate.draw_detections(img, drawn)
    annotated_path = out_dir / f"{stem}_annotated.jpg"
    annotate.imwrite(annotated_path, annotated, 88)
    paths = {"annotated": str(annotated_path)}

    plate = next((d for d in report["detections"] if d["category"] == "license_plate"), None)
    if plate:
        bbox = [plate["bounding_box_px"][k] for k in ("x1", "y1", "x2", "y2")]
        zoom = annotate.crop_with_margin(img, bbox, margin=0.6, min_side=160)
        zoom_path = out_dir / f"{stem}_plate.jpg"
        annotate.imwrite(zoom_path, zoom, 92)
        paths["plate_zoom"] = str(zoom_path)

    main_id = report["summary"].get("main_vehicle_id")
    main = next((d for d in report["detections"] if d["id"] == main_id), None)
    if main:
        bbox = [main["bounding_box_px"][k] for k in ("x1", "y1", "x2", "y2")]
        thumb = annotate.crop_with_margin(img, bbox, margin=0.1)
        thumb_path = out_dir / f"{stem}_vehicle.jpg"
        annotate.imwrite(thumb_path, thumb, 85)
        paths["vehicle_crop"] = str(thumb_path)
    return paths
