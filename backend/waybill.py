"""Путевой лист машины — ИМИТАЦИЯ.

В компании путевой лист выписывает диспетчер до выезда: водитель, машина, задание (что и откуда
везти), отметки медика и механика. Весовая берёт из него водителя и груз, весовщику вводить их не
нужно. Реальных путевых листов у нас нет, поэтому лист придумывается правдоподобно и стабильно:
за машиной закреплён один водитель (зависит только от бортового номера), а груз, поле и время
выезда — свои у каждого рейса; на заезде и выезде рейса лист один и тот же (хранится в рейсе).
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from backend.pipeline import makes

ORGANIZATION = "ТОО «Ceres Агро»"   # хозяйство-заказчик в демо, не реальная компания

SURNAMES = ["Ахметов", "Жумабаев", "Сейткали", "Исмаилов", "Кусаинов", "Бекмурзин", "Нургалиев",
            "Оспанов", "Иванов", "Петренко", "Шевченко", "Кузнецов", "Смирнов", "Мельник",
            "Ковальчук", "Баймуханов", "Тулеуов", "Абенов", "Горбунов", "Садыков"]
INITIALS = ["А.С.", "Б.К.", "Е.М.", "Н.Т.", "С.П.", "В.И.", "Д.А.", "М.Ж.", "Р.Б.", "К.Е.", "Т.Н.", "О.В."]
STAFF = {"медик": ["Сарсенова Г.А.", "Литвиненко О.П.", "Ермекова Д.С."],
         "механик": ["Петров В.И.", "Касымов Б.Т.", "Шульц А.А."],
         "диспетчер": ["Ахметова Г.К.", "Волкова Н.С.", "Жакупова А.М."]}
# Культуры Костанайской области; пшеница и ячмень — основной объём
CROPS = [("Пшеница", 50), ("Ячмень", 25), ("Овёс", 8), ("Лён", 7), ("Подсолнечник", 6), ("Рапс", 4)]
PLOTS = ["Северный", "Южный", "Береговой", "Степной", "Озёрный", "Дальний"]
TRAILERS = ["зерновоз 2ПТС-4", "самосвальный СЗАП-8551", "тракторный 2ПТС-9"]


def _kind(equipment_type: str | None) -> str:
    t = (equipment_type or "").lower()
    if "трактор" in t:
        return "tractor"
    if "легков" in t or "пикап" in t or "внедорож" in t:
        return "car"
    return "truck"


def _driver(vehicle_id: str, kind: str) -> dict:
    """Водитель закреплён за машиной: один и тот же при каждом приезде."""
    rnd = random.Random(f"driver|{vehicle_id}")
    surname, initials = rnd.choice(SURNAMES), rnd.choice(INITIALS)
    return {
        "name": f"{surname} {initials}",
        "personnel_no": f"{rnd.randint(100, 999)}",
        "license": f"{rnd.choice('ABCEHKMPTX')}{rnd.choice('ABCEHKMPTX')} {rnd.randint(100000, 999999)}",
        "category": {"tractor": "тракторист-машинист, кат. D", "car": "B"}.get(kind, "C, CE"),
    }


def issue(vehicle_id: str, vehicle: dict, event_time: str | None, visit_no: int,
          has_trailer: bool | None = None) -> dict:
    """Путевой лист на рейс. vehicle: manufacturer, model, equipment_type, plate_formatted."""
    kind = _kind(vehicle.get("equipment_type"))
    try:
        at = datetime.strptime(event_time or "", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        at = datetime.now()
    rnd = random.Random(f"trip|{vehicle_id}|{at:%Y-%m-%d}|{visit_no}")
    crop = rnd.choices([c for c, _ in CROPS], weights=[w for _, w in CROPS])[0]
    departed = at - timedelta(minutes=rnd.randint(35, 150))
    make, model = vehicle.get("manufacturer"), vehicle.get("model")
    model = model if model and model != "не определена" else None
    if make and model and makes.same_make(make, model):
        make = None   # «КамАЗ-65115», а не «КамАЗ КамАЗ-65115»
    make_model = " ".join(x for x in (make, model) if x) or "—"
    return {
        "number": f"{at:%y%m%d}-{str(vehicle_id).zfill(3)}{'' if visit_no <= 1 else f'/{visit_no}'}",
        "date": f"{at:%d.%m.%Y}",
        "kind": kind,
        "title": "Путевой лист трактора" if kind == "tractor" else "Путевой лист грузового автомобиля",
        "organization": ORGANIZATION,
        "vehicle": {"make_model": make_model, "type": vehicle.get("equipment_type") or "—",
                    "plate": vehicle.get("plate_formatted") or "—",
                    "garage_no": f"Г-{str(vehicle_id).zfill(2)}",
                    "trailer": rnd.choice(TRAILERS) if has_trailer else "—"},
        "driver": _driver(str(vehicle_id), kind),
        "task": {"cargo": crop, "from": f"Поле №{rnd.randint(1, 24)} · участок «{rnd.choice(PLOTS)}»",
                 "trips_planned": rnd.randint(1, 4)},
        "departure": {"time": f"{departed:%H:%M}",
                      ("engine_hours" if kind == "tractor" else "odometer_km"):
                          rnd.randint(1200, 9800) if kind == "tractor" else rnd.randint(40000, 480000)},
        "marks": {"medic": f"{rnd.choice(STAFF['медик'])} · {(departed - timedelta(minutes=12)):%H:%M} · к работе допущен",
                  "mechanic": f"{rnd.choice(STAFF['механик'])} · {(departed - timedelta(minutes=6)):%H:%M} · технически исправен, выезд разрешён",
                  "dispatcher": rnd.choice(STAFF["диспетчер"])},
        "imitation": True,
    }
