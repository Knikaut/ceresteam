"""Склады и весовые — имитация. Координаты условные: поля под Костанаем
(код региона 10 на номерах машин из датасета)."""

WAREHOUSES = [
    {"id": "W1", "name": "Склад №1 · Элеватор", "scale": "деревянная весовая",
     "capacity_t": 800, "lat": 53.2205, "lon": 63.6280},
    {"id": "W2", "name": "Склад №2 · Ток", "scale": "весы с синими бортами",
     "capacity_t": 400, "lat": 53.2412, "lon": 63.6710},
    {"id": "W3", "name": "Склад №3 · Ангар", "scale": "весы под арочным навесом",
     "capacity_t": 600, "lat": 53.1980, "lon": 63.6790},
    {"id": "W4", "name": "Склад №4 · МТМ", "scale": "весы у мастерских",
     "capacity_t": 300, "lat": 53.2090, "lon": 63.5860},
]

BY_ID = {w["id"]: w for w in WAREHOUSES}
DEFAULT_ID = "W1"


def get(warehouse_id: str | None) -> dict:
    return BY_ID.get(warehouse_id or DEFAULT_ID, BY_ID[DEFAULT_ID])
