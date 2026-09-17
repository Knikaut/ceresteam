"""Имитация весов: правдоподобные значения по типу техники.

Реальных весов в демо нет; вес либо вводит охранник, либо генерируется здесь.
Тара одной и той же машины стабильна (небольшой разброс), брутто зависит от груза.
"""
from __future__ import annotations

import random

# (тара, полезная нагрузка) в тоннах по семейству техники.
PROFILES = {
    "газ": (3.4, 4.5),
    "зил": (4.3, 6.0),
    "маз": (9.5, 13.0),
    "камаз": (8.8, 12.0),
    "урал": (9.0, 10.0),
    "кировец": (13.5, 14.0),   # трактор с прицепом
    "мтз": (4.0, 6.0),
    "scania": (9.0, 20.0),
    "volvo": (9.0, 20.0),
    "man": (9.0, 20.0),
    "default_truck": (8.0, 11.0),
    "default_car": (1.6, 0.4),
}


def _profile(manufacturer: str | None, equipment_type: str | None) -> tuple[float, float]:
    key = (manufacturer or "").lower()
    for name, prof in PROFILES.items():
        if name in key:
            return prof
    if equipment_type and "легков" in equipment_type.lower():
        return PROFILES["default_car"]
    return PROFILES["default_truck"]


def stable_tare(vehicle_id: str, manufacturer: str | None, equipment_type: str | None) -> float:
    """Тара конкретной машины: зависит от ID, чтобы повторные выезды сходились."""
    tare, _ = _profile(manufacturer, equipment_type)
    rnd = random.Random(vehicle_id)
    return round(tare * rnd.uniform(0.95, 1.05), 2)


def generate(kind: str, vehicle_id: str, manufacturer: str | None, equipment_type: str | None,
             load_state: str | None = None) -> float:
    """kind: 'entry' (гружёная) или 'exit' (пустая)."""
    tare = stable_tare(vehicle_id, manufacturer, equipment_type)
    _, payload = _profile(manufacturer, equipment_type)
    if kind == "exit" and load_state != "гружёный":
        return round(tare + random.uniform(-0.05, 0.08), 2)   # пустая, небольшой дрейф весов
    if load_state == "пустой":
        return round(tare + random.uniform(-0.05, 0.08), 2)
    return round(tare + payload * random.uniform(0.75, 1.0), 2)
