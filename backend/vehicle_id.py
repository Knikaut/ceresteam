"""Бортовой номер машины: порядковый номер 1, 2, 3… и подпись «№1 · МАЗ 5551»."""
from __future__ import annotations

UNKNOWN_MODEL = {None, "", "не определена"}


def display_name(v: dict) -> str:
    make = v.get("manufacturer") or ""
    model = v.get("model") if v.get("model") not in UNKNOWN_MODEL else ""
    title = " ".join(x for x in (make, model) if x) or (v.get("equipment_type") or "техника")
    return f"№{v['id']} · {title}"
