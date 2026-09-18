"""Запасной классификатор: когда зовётся и что имеет право подставить в отчёт. Настоящая модель не грузится."""
from __future__ import annotations

import numpy as np
import pytest

from backend.pipeline import analyze, fallback, makes


def ответ(make=None, model=None, kind=None, уверен=True, conf=0.9):
    def поле(value):
        return {"value": value, "confidence": conf, "threshold": 0.5, "confident": уверен} if value else None
    return {k: v for k, v in {"make": поле(make), "model": поле(model), "type": поле(kind)}.items() if v}


def запись(**поля):
    entry = {"manufacturer": None, "model": None, "manufacturer_confidence": None, "model_confidence": None,
             "equipment_type": "Грузовой автомобиль", "license_plate": {"readable": True}, "notes": ""}
    entry.update(поля)
    return entry


@pytest.fixture(autouse=True)
def марки_моделей(monkeypatch):
    monkeypatch.setattr(fallback, "_state", {"model_make": {"К-744": "Кировец", "ГАЗ-3307": "ГАЗ",
                                                            "ВАЗ-2121 Нива": "Lada/ВАЗ"}})


# ---------- когда зовётся ----------

def test_номер_прочитан_и_марка_с_моделью_есть_модель_не_нужна():
    assert not analyze.needs_fallback(запись(manufacturer="КамАЗ", model="КамАЗ-65115"))


@pytest.mark.parametrize("поля", [
    {"license_plate": {"readable": False}, "manufacturer": "КамАЗ", "model": "КамАЗ-65115"},
    {"manufacturer": None},
    {"manufacturer": "КамАЗ", "model": None},
])
def test_номер_плохо_виден_или_марка_модель_пусты_модель_нужна(поля):
    assert analyze.needs_fallback(запись(**поля))


# ---------- что подставляет ----------

def test_уверенный_ответ_заполняет_пустую_марку_и_модель():
    entry = запись()
    analyze.apply_fallback(entry, ответ(make="Кировец", model="К-744"))
    assert entry["manufacturer"] == "Кировец" and entry["model"] == "К-744"
    assert "обученная модель" in entry["manufacturer_confidence"]
    assert entry["fallback"]["make"]["value"] == "Кировец", "Ответ модели не сохранён в отчёте"


def test_неуверенный_ответ_ничего_не_подставляет():
    entry = запись()
    analyze.apply_fallback(entry, ответ(make="Кировец", model="К-744", уверен=False))
    assert entry["manufacturer"] is None and entry["model"] is None


def test_марку_из_реестра_не_перетирает_а_пишет_расхождение():
    entry = запись(manufacturer="КамАЗ", manufacturer_confidence="высокая (реестр data.egov.kz по госномеру)")
    analyze.apply_fallback(entry, ответ(make="МАЗ"))
    assert entry["manufacturer"] == "КамАЗ"
    assert "МАЗ" in entry["notes"], "Расхождение с реестром не отмечено"


def test_уверенная_модель_при_неуверенной_марке_даёт_марку():
    entry = запись()
    guess = ответ(model="ГАЗ-3307")
    guess["make"] = {"value": fallback.UNKNOWN, "confidence": 0.4, "threshold": 0.2, "confident": False}
    analyze.apply_fallback(entry, guess)
    assert entry["manufacturer"] == "ГАЗ" and entry["model"] == "ГАЗ-3307"


def test_модель_чужой_марки_не_подставляется():
    """Реестр сказал КамАЗ, модель по виду — К-744 (Кировец): «КамАЗ К-744» быть не должно."""
    entry = запись(manufacturer="КамАЗ")
    analyze.apply_fallback(entry, ответ(model="К-744"))
    assert entry["model"] is None


def test_лада_записывается_как_ваз_и_страна_находится():
    entry = запись()
    analyze.apply_fallback(entry, ответ(make="Lada/ВАЗ", model="ВАЗ-2121 Нива"))
    assert entry["manufacturer"] == "ВАЗ (Lada)" and entry["model"] == "ВАЗ-2121 Нива"
    assert makes.country_for(entry["manufacturer"]) == "Россия"


def test_тип_ставится_только_вместо_общего_типа_детектора():
    общий = запись()
    analyze.apply_fallback_type(общий, ответ(kind="самосвал"), "Грузовой автомобиль")
    assert общий["equipment_type"] == "Самосвал"

    от_vlm = запись(equipment_type="Бортовой грузовик с тентом")
    analyze.apply_fallback_type(от_vlm, ответ(kind="самосвал"), "Грузовой автомобиль")
    assert от_vlm["equipment_type"] == "Бортовой грузовик с тентом", "Тип от VLM перетёрт"


@pytest.mark.parametrize("make, model, title", [
    ("КамАЗ", "КамАЗ-55111", "№1 · КамАЗ-55111"),
    ("ВАЗ (Lada)", "ВАЗ-2121 Нива", "№1 · ВАЗ-2121 Нива"),
    ("Кировец", "К-744", "№1 · Кировец К-744"),
    ("МАЗ", "5551", "№1 · МАЗ 5551"),
    ("МАЗ", "5551 (или аналог, с прицепом)", "№1 · МАЗ 5551"),
    ("МАЗ", "5337/5551 (примерно)", "№1 · МАЗ 5337/5551"),
])
def test_марка_в_названии_не_повторяется(make, model, title):
    from backend.vehicle_id import display_name
    assert display_name({"id": 1, "manufacturer": make, "model": model}) == title


# ---------- вероятности ----------

def test_не_видно_и_другое_не_считаются_ответом_даже_при_высокой_вероятности():
    heads = {"make": {"classes": ["КамАЗ", fallback.UNKNOWN], "coef": np.array([[0.0, 0.0], [5.0, 0.0]], np.float32),
                      "intercept": np.zeros(2, np.float32), "threshold": 0.3}}
    res = fallback.predict(heads, np.array([1.0, 0.0], np.float32))
    assert res["make"]["value"] == fallback.UNKNOWN and res["make"]["confidence"] > 0.9
    assert res["make"]["confident"] is False


def test_вероятности_это_softmax_и_порог_применяется():
    heads = {"type": {"classes": ["самосвал", "трактор"], "coef": np.array([[1.0], [0.0]], np.float32),
                      "intercept": np.zeros(2, np.float32), "threshold": 0.8}}
    res = fallback.predict(heads, np.array([1.0], np.float32))
    assert res["type"]["value"] == "самосвал"
    assert res["type"]["confidence"] == pytest.approx(np.e / (np.e + 1), abs=1e-3)
    assert res["type"]["confident"] is False, "0.73 ниже порога 0.8 — ответ не должен считаться уверенным"


def test_вырезка_как_при_обучении_поля_и_длинная_сторона_512():
    img = np.zeros((1520, 2688, 3), np.uint8)
    crop = fallback.training_crop(img, [1000, 200, 2000, 900])
    assert max(crop.shape[:2]) == fallback.CROP_LONG_SIDE
    мелкая = fallback.training_crop(img, [100, 100, 300, 200])
    # 200×100 плюс поля по 8% с каждой стороны: 232×116, без увеличения
    assert мелкая.shape[:2] == (116, 232), "Мелкую вырезку нельзя увеличивать — при обучении её не увеличивали"
