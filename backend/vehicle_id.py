"""Бортовой номер машины: порядковый номер 1, 2, 3… и подпись «№1 · МАЗ 5551»."""
from __future__ import annotations

import re

from backend.pipeline import makes

UNKNOWN_MODEL = {None, "", "не определена"}


def display_name(v: dict) -> str:
    make = v.get("manufacturer") or ""
    model = v.get("model") if v.get("model") not in UNKNOWN_MODEL else ""
    # Пояснения VLM в скобках — в таблицу ответа, не в название: «МАЗ 5551», а не «МАЗ 5551 (или аналог, с прицепом)»
    model = re.sub(r"\s*\([^)]*\)", "", model).strip()
    if make and model and makes.same_make(make, model):
        make = ""   # модель уже начинается с марки: «КамАЗ-55111», а не «КамАЗ КамАЗ-55111»
    title = " ".join(x for x in (make, model) if x) or (v.get("equipment_type") or "техника")
    return f"№{v['id']} · {title}"
