"""Реестр data.egov.kz: поиск по локальному снимку, без обращений в сеть."""
from __future__ import annotations

import json

import pytest

from backend.pipeline import registry


@pytest.fixture
def снимок(tmp_path, monkeypatch):
    """Маленький снимок набора вместо настоящих 24 759 записей."""
    path = tmp_path / "registry.jsonl"
    строки = [
        {"reg_number": "536AEH01", "marka": "HYUNDAI SANTA FE", "model": "HYUNDAI SANTA FE",
         "id": "1", "type": "2", "year_issue": "2022", "number_doc": "ZI00067884"},
        {"reg_number": "651ADB10", "marka": "SHACMAN SX3258DR384", "model": "SHACMAN SX3258DR384",
         "id": "2", "type": "3", "year_issue": "2022", "number_doc": "ZI00090107"},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in строки), encoding="utf-8")
    monkeypatch.setenv("REGISTRY_SNAPSHOT", str(path))
    monkeypatch.setattr(registry, "_snapshot", None)
    monkeypatch.setattr(registry, "_cache", {})
    return path


def test_номер_находится_в_снимке_без_сети(снимок, monkeypatch):
    monkeypatch.setattr(registry, "_get_json", лови_сеть)
    статус, запись = registry.lookup_status("536 AEH 01")

    assert статус == registry.FOUND, статус
    assert запись["manufacturer"] == "HYUNDAI", запись
    assert запись["year"] == "2022"


def test_номера_нет_в_снимке_и_это_обычный_ответ(снимок, monkeypatch):
    """В наборе только регистрации 2022 года, номеров с кадров весовой там нет."""
    monkeypatch.setattr(registry, "_get_json", лови_сеть)
    статус, запись = registry.lookup_status("041 AHF 10")

    assert статус == registry.NOT_FOUND, статус
    assert запись is None


def test_грузовик_получает_свой_тип(снимок, monkeypatch):
    monkeypatch.setattr(registry, "_get_json", лови_сеть)
    _, запись = registry.lookup_status("651ADB10")
    assert запись["type"] == "грузовой автомобиль", запись


def лови_сеть(*args, **kwargs):
    raise AssertionError("Поиск полез в интернет, хотя снимок набора есть на диске")
