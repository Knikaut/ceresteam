"""Общие приспособления для тестов весовой.

Правила набора:
* быстрые тесты не грузят настоящие модели (YOLO, EasyOCR) и не ходят в сеть;
* всё, что пишется на диск, уходит во временную папку — рабочие results/ не трогаем;
* тяжёлые тесты живут под маркером `models` и по умолчанию не запускаются.
"""
from __future__ import annotations

import copy
import json
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import config, db  # noqa: E402
from backend.pipeline import analyze, annotate, detector, plates, registry, vlm  # noqa: E402

# Реальные кадры и эталонная разметка (для тестов под маркером `models`).
FRAMES_DIR = Path("/Users/bauyrzanesekeev/PhpstormProjects/project/data/Auto")
GROUND_TRUTH = Path("/Users/bauyrzanesekeev/PhpstormProjects/project/eval/ground_truth.json")
# Снимок реестра data.egov.kz, скачанный tools/fetch_registry.py.
REGISTRY_SNAPSHOT = ROOT / "results" / "registry" / "registered_vehicles.jsonl"


# ---------- изоляция: сеть и тяжёлые модели ----------

@pytest.fixture(autouse=True)
def без_сети(request, monkeypatch):
    """Быстрые тесты не должны ходить в интернет: любое соединение — ошибка теста."""
    if request.node.get_closest_marker("models"):
        return

    def _отказ(*args, **kwargs):
        raise AssertionError(
            "Тест попытался выйти в сеть. Сетевой вызов нужно подменять заглушкой: "
            "офлайн-прогон обязан работать без интернета."
        )

    monkeypatch.setattr(socket.socket, "connect", _отказ, raising=False)
    monkeypatch.setattr(socket.socket, "connect_ex", _отказ, raising=False)
    monkeypatch.setattr(socket, "create_connection", _отказ, raising=False)


@pytest.fixture(autouse=True)
def без_тяжёлых_моделей(request, monkeypatch):
    """Ловим случайную загрузку YOLO/EasyOCR: в быстром наборе их быть не должно."""
    if request.node.get_closest_marker("models"):
        return

    def _отказ(*args, **kwargs):
        raise AssertionError(
            "Тест дошёл до настоящей модели (YOLO/EasyOCR). В быстром наборе модели "
            "подменяются заглушками; тест с настоящими моделями помечается маркером `models`."
        )

    monkeypatch.setattr(plates, "get_reader", _отказ)
    monkeypatch.setattr(detector, "get_model", _отказ)
    monkeypatch.setattr(detector, "resolve_device", lambda name=None: "cpu")


# ---------- изоляция: файлы и база ----------

@pytest.fixture(autouse=True)
def временные_папки(tmp_path, monkeypatch):
    """RESULTS_DIR, DB_PATH и DATA_DIR на время теста уезжают во временную папку."""
    results = tmp_path / "results"
    frames = tmp_path / "frames"
    results.mkdir(parents=True, exist_ok=True)
    frames.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "RESULTS_DIR", results)
    monkeypatch.setattr(config, "DB_PATH", results / "weighbridge.db")
    monkeypatch.setattr(config, "DATA_DIR", frames)

    # Константы, посчитанные при импорте модулей от старого RESULTS_DIR.
    import backend.service as service
    monkeypatch.setattr(service, "CAPTURES_DIR", results / "captures")
    monkeypatch.setattr(service, "REPORTS_DIR", results / "reports")
    monkeypatch.setattr(registry, "CACHE_PATH", results / "registry_cache.json")
    monkeypatch.setattr(registry, "_cache", None, raising=False)
    monkeypatch.setattr(registry, "_offline_until", 0.0, raising=False)
    monkeypatch.setattr(vlm, "CACHE_PATH", results / "vlm_cache.json")

    yield results

    # База закрывается руками: глобальное соединение переживает monkeypatch.
    if getattr(db, "_con", None) is not None:
        try:
            db._con.close()
        except Exception:
            pass
        db._con = None


@pytest.fixture
def база(временные_папки):
    """Чистая SQLite-база во временной папке."""
    db._con = None
    db.init()
    yield db
    if db._con is not None:
        db._con.close()
        db._con = None


# ---------- синтетические данные ----------

def картинка(width: int = 240, height: int = 180, seed: int = 0) -> np.ndarray:
    """Кадр-заглушка: цветной шум с фиксированным зерном (нужен отпечаток внешности)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


@pytest.fixture
def кадр_на_диске(tmp_path):
    """Фабрика: кладёт синтетический кадр в файл и возвращает путь."""
    def _создать(name: str = "1.jpg", width: int = 240, height: int = 180, seed: int = 0) -> Path:
        path = tmp_path / "frames" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        annotate.imwrite(path, картинка(width, height, seed))
        return path
    return _создать


def номер(text: str = "041AHF10", formatted: str | None = None, kind: str = "kz_2012",
          region: str | None = "10", conf: float = 0.9, votes: float = 1.0,
          region_valid: bool = True, fixes: int = 0, bbox: list[int] | None = None):
    """Готовый plates.PlateResult для подстановки вместо OCR."""
    result = plates.PlateResult(
        text=text,
        formatted=formatted or f"{text[:3]} {text[3:-2]} {text[-2:]}",
        kind=kind,
        region_code=region,
        confidence=conf,
        bbox=bbox or [30, 40, 90, 60],
        raw=text,
        region_valid=region_valid,
        fixes=fixes,
    )
    result.votes = votes
    return result


def отчёт(source_image: str = "1.jpg", *, plate: str | None = "041AHF10",
          plate_readable: bool = True, manufacturer: str | None = "МАЗ",
          model: str | None = "5551", timestamp: str | None = "2019-05-02 21:58:00",
          appearance: dict | None = None, bbox: tuple[int, int, int, int] = (20, 20, 200, 160),
          equipment_type: str = "Грузовой автомобиль", people: int = 0) -> dict:
    """Готовый отчёт по кадру — как его отдаёт analyze.analyze_image, только без моделей."""
    x1, y1, x2, y2 = bbox
    plate_block: dict | None
    if plate is None:
        plate_block = {"text": None, "readable": False, "confidence": None,
                       "notes": "Номер не найден или не читается"}
    else:
        formatted = f"{plate[:3]} {plate[3:-2]} {plate[-2:]}"
        plate_block = {
            "text": plate if plate_readable else None,
            "formatted": formatted if plate_readable else None,
            "country": "Казахстан (KZ)",
            "region_code": plate[-2:] if plate_readable else None,
            "format": "kz_2012",
            "readable": plate_readable,
            "confidence": "высокая" if plate_readable else "низкая",
            "ocr_confidence": 0.9,
            "agreement": 1.0,
            "alternatives": [],
        }

    vehicle = {
        "id": 1, "category": "vehicle", "on_scale": True, "yolo_class": "truck",
        "equipment_type": equipment_type, "manufacturer": manufacturer, "model": model,
        "manufacturer_confidence": "высокая (надпись прочитана на технике)" if manufacturer else None,
        "model_confidence": None, "license_plate": plate_block,
        "bounding_box_px": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "detection_confidence": 0.91, "appearance": appearance, "notes": "",
        "manufacturer_country": "Беларусь" if manufacturer == "МАЗ" else None,
    }
    detections = [vehicle]
    next_id = 2
    if plate and plate_readable:
        detections.append({
            "id": next_id, "category": "license_plate",
            "equipment_type": "Государственный регистрационный знак транспортного средства id=1",
            "manufacturer": None, "model": None,
            "license_plate": {"text": plate, "formatted": plate_block["formatted"],
                              "country": "Казахстан (KZ)", "region_code": plate[-2:], "format": "kz_2012"},
            "bounding_box_px": {"x1": x1 + 10, "y1": y1 + 40, "x2": x1 + 60, "y2": y1 + 60},
            "detection_confidence": 0.9, "readable": True, "vehicle_id": 1,
        })
        next_id += 1
    for i in range(people):
        detections.append({
            "id": next_id + i, "category": "person",
            "equipment_type": "Человек (не техника, сотрудник весовой/водитель)",
            "manufacturer": None, "model": None, "license_plate": None,
            "bounding_box_px": {"x1": 5, "y1": 5, "x2": 15, "y2": 40},
            "detection_confidence": 0.7, "notes": "Включён для полноты сцены.",
        })

    return {
        "source_image": source_image,
        "image_resolution": {"width": 240, "height": 180},
        "camera_id": "Camera 01",
        "timestamp_on_frame": timestamp,
        "note_on_timestamp": "Дата/время прочитаны с наложенной надписи камеры.",
        "processing_seconds": 0.01,
        "detections": detections,
        "summary": {
            "total_objects_detected": len(detections),
            "vehicles_or_equipment": 1,
            "vehicles_on_scale": 1,
            "people": people,
            "plates_read": sum(1 for d in detections if d["category"] == "license_plate"),
            "main_vehicle_id": 1,
        },
    }


# ---------- стенд весовой ----------

class Весовая:
    """Весовая с подменённым анализом кадра: рейсы, предупреждения, отмена и правка."""

    def __init__(self, service_mod, tmp_path, monkeypatch):
        self.service = service_mod
        self.tmp_path = tmp_path
        self.отчёты: dict[str, dict] = {}
        self.вызовы_анализа: list[str] = []

        def _анализ(image_path, use_vlm=True, use_registry=True):
            stem = Path(image_path).stem
            self.вызовы_анализа.append(stem)
            if stem not in self.отчёты:
                raise ValueError(f"Для снимка {stem} тест не подготовил отчёт")
            return copy.deepcopy(self.отчёты[stem])

        def _файлы(image_path, report, out_dir):
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            marked = out_dir / f"{Path(image_path).stem}_annotated.jpg"
            annotate.imwrite(marked, картинка(60, 40))
            return {"annotated": str(marked)}

        monkeypatch.setattr(analyze, "analyze_image", _анализ)
        monkeypatch.setattr(analyze, "render_outputs", _файлы)
        monkeypatch.setattr(registry, "lookup_status", lambda plate, use_cache=True: (registry.NOT_FOUND, None))

    def снимок(self, report: dict, *, seed: int = 0) -> str:
        """Кладёт «кадр с камеры» и регистрирует отчёт, который вернёт анализ."""
        src = self.tmp_path / "frames" / f"src_{seed}_{len(self.отчёты)}.jpg"
        src.parent.mkdir(parents=True, exist_ok=True)
        annotate.imwrite(src, картинка(seed=seed))
        capture = self.service.save_capture(src, src.name)
        self.отчёты[capture["capture_id"]] = report
        return capture["capture_id"]

    def отправить(self, report: dict | None = None, *, capture_id: str | None = None,
                  seed: int = 0, warehouse_id: str = "W1", weight=None, driver=None,
                  crop=None, note=None, plate_override=None) -> dict:
        if capture_id is None:
            capture_id = self.снимок(report if report is not None else отчёт(), seed=seed)
        guard = {"driver": driver, "crop": crop, "warehouse_id": warehouse_id, "weight": weight,
                 "note": note, "plate_override": plate_override, "frame": None}
        return self.service.process_capture(capture_id, guard)

    @staticmethod
    def предупреждения(result: dict) -> list[str]:
        return list((result["ai_message"].get("payload") or {}).get("alerts") or [])

    @staticmethod
    def текст(result: dict) -> str:
        return result["ai_message"]["text"]


@pytest.fixture
def весовая(база, tmp_path, monkeypatch) -> Весовая:
    import backend.service as service
    return Весовая(service, tmp_path, monkeypatch)


# ---------- эталонная разметка ----------

@pytest.fixture(scope="session")
def эталон() -> list[dict]:
    if not GROUND_TRUTH.exists():
        pytest.skip(f"Эталонная разметка не найдена: {GROUND_TRUTH}")
    return json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))


def эталонные_номера(only_readable: bool = True) -> list[tuple[str, str]]:
    """[(кадр, номер)] из эталонной разметки."""
    if not GROUND_TRUTH.exists():
        return []
    out = []
    for image in json.loads(GROUND_TRUTH.read_text(encoding="utf-8")):
        for obj in image.get("objects", []):
            plate = obj.get("plate") or {}
            if plate.get("text") and (plate.get("readable") or not only_readable):
                out.append((image["image"], plate["text"]))
    return out
