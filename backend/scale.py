"""Весы на COM-порту — ИМИТАЦИЯ.

На настоящей весовой весовой терминал подключён к компьютеру по COM-порту (RS-232, обычно
9600 бод, 8N1) и непрерывно передаёт показание; программа берёт вес, когда он стабилизировался.
Реального терминала и его данных у нас нет, поэтому показание придумывается правдоподобно по
типу техники (weights.generate: тара машины стабильна, брутто зависит от груза) — и везде
подписано как имитация, чтобы его не приняли за настоящее взвешивание.
"""
from __future__ import annotations

import threading
from datetime import datetime

from backend import config, weights

_lock = threading.Lock()
_last: dict | None = None   # последнее показание: {"weight_t", "at", "stable", "kind", "vehicle_id"}


def source() -> str:
    """Подпись источника веса в рейсе, отчёте и JSON."""
    return f"весы {config.SCALE_PORT} (имитация показаний)"


def read(kind: str, vehicle_id: str, manufacturer: str | None, equipment_type: str | None,
         load_state: str | None = None) -> float:
    """Стабилизированное показание весов для машины на платформе: брутто на заезде, тара на выезде."""
    global _last
    weight = weights.generate(kind, vehicle_id, manufacturer, equipment_type, load_state)
    with _lock:
        _last = {"weight_t": weight, "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "stable": True, "kind": kind, "vehicle_id": vehicle_id}
    return weight


def status() -> dict:
    """Что показать в интерфейсе: порт, параметры связи, последнее показание."""
    with _lock:
        last = dict(_last) if _last else None
    return {"port": config.SCALE_PORT, "baud": config.SCALE_BAUD, "frame": "8N1",
            "connected": True, "imitation": True, "last": last}
