"""Настройки проекта. Всё можно переопределить переменными окружения."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Читает KEY=VALUE из .env (секреты вроде ANTHROPIC_API_KEY). Переменные окружения важнее файла."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")

# Папка с кадрами камеры. В контейнере организаторов это /data/Auto.
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "фото машин на взевшивании" / "Auto"))
if not DATA_DIR.exists() and Path("/data/Auto").exists():
    DATA_DIR = Path("/data/Auto")

# Куда складываем результаты: размеченные кадры, JSON, CSV, база.
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
DB_PATH = Path(os.environ.get("DB_PATH", RESULTS_DIR / "weighbridge.db"))

# Детектор техники (COCO-классы car/bus/truck/person). Скачивается при первом запуске.
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8s.pt")
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "1280"))

# OCR: английский для номеров, русский для надписей на технике (МАЗ, КИРОВЕЦ ...).
OCR_LANGS = ["en", "ru"]
# Где считать OCR: "auto" — видеокарта NVIDIA, если она есть, иначе процессор;
# "cpu"/"off" — только процессор; "gpu"/"on" — любой ускоритель (на маке это mps);
# можно указать устройство явно: "cuda", "cuda:1", "mps".
OCR_GPU = os.environ.get("OCR_GPU", "auto")
# Чем читать номера: "auto" — специализированной моделью, если пакеты установлены,
# иначе прежним EasyOCR; "easyocr" — принудительно прежним путём.
PLATE_READER = os.environ.get("PLATE_READER", "auto")

# Реестр data.egov.kz: марка/модель/год по номеру. Ключ необязателен (без него — поиск как на сайте).
REGISTRY_ENABLED = os.environ.get("REGISTRY_ENABLED", "on")
EGOV_API_KEY = os.environ.get("EGOV_API_KEY", "")

# VLM (описание внешности, марка/модель). "auto" = включить, если есть доступ к API; "off" — не использовать.
# Провайдер: "openrouter" (ключ OPENROUTER_API_KEY), "anthropic" (ANTHROPIC_API_KEY) или "auto" — что найдётся.
VLM_PROVIDER = os.environ.get("VLM_PROVIDER", "auto")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# Модель для OpenRouter: Claude Sonnet 5 — хорошее качество на технике СНГ за ~0,5 цента/кадр;
# запасная — Gemini 3.5 Flash, если основная недоступна.
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-5")
OPENROUTER_FALLBACK = os.environ.get("OPENROUTER_FALLBACK", "google/gemini-3.5-flash")
VLM_MODEL = os.environ.get("VLM_MODEL", "claude-opus-5")
VLM_ENABLED = os.environ.get("VLM_ENABLED", "auto")
# Глубина рассуждений: low дешевле и быстрее; medium/high — если качество марки/модели не устроит.
VLM_EFFORT = os.environ.get("VLM_EFFORT", "low")

# Весы на COM-порту. Реального весового терминала нет — показания имитируются (backend/scale.py).
SCALE_PORT = os.environ.get("SCALE_PORT", "COM3")
SCALE_BAUD = int(os.environ.get("SCALE_BAUD", "9600"))

# Запасной классификатор (SigLIP2 + обученные головы, backend/pipeline/fallback.py): тип, марка и модель
# по виду машины — когда номер не прочитан или марка/модель не определены. "auto" — включить, если есть
# файл голов и пакет transformers; "off" — не использовать. Веса SigLIP2 скачиваются при первом запуске.
FALLBACK_ENABLED = os.environ.get("FALLBACK_ENABLED", "auto")
FALLBACK_HEAD = os.environ.get("FALLBACK_HEAD", str(ROOT / "backend" / "pipeline" / "fallback_head.npz"))

def _cpu_limit() -> int:
    """Сколько ядер процессора реально доступно. В контейнере организаторов видно 8 ядер, а квота
    cgroup — 2: библиотеки запускают 8 потоков, система их всё время притормаживает, и номер
    читается 36 с вместо долей секунды. Вне контейнера квоты нет — берём все ядра."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


CPU_THREADS = int(os.environ.get("CPU_THREADS") or _cpu_limit())


def cpu_threads_limited() -> bool:
    return CPU_THREADS < (os.cpu_count() or 1)


def limit_cpu_threads() -> None:
    """Потоки torch и OpenCV — по квоте процессора. Вызывается при загрузке моделей."""
    if cpu_threads_limited():
        import cv2
        import torch
        torch.set_num_threads(CPU_THREADS)
        cv2.setNumThreads(CPU_THREADS)


HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
