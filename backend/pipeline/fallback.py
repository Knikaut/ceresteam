"""Запасной классификатор техники: тип, марка и модель по внешнему виду машины.

Нужен, только когда обычный путь ответа не дал: номер не прочитан (значит, реестр по номеру
недоступен) или марка/модель так и не определены. Во всех остальных случаях работает то, что было:
реестр data.egov.kz по номеру, надпись на технике, VLM.

Как устроен: SigLIP2 (google/siglip2-so400m-patch16-384, веса заморожены) превращает вырезку машины
в вектор, поверх обучены три логистические регрессии — тип, марка, модель (ml/train_head.py).
Их веса лежат в fallback_head.npz рядом с этим файлом. Ответ считается уверенным, если вероятность не
ниже порога, подобранного на кросс-валидации так, чтобы уверенные ответы были верны в ≥90% случаев.
Нет transformers, весов SigLIP2 или файла голов — классификатор выключается, анализ идёт как раньше.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np

from backend import config
from backend.pipeline import annotate

HEAD_PATH = Path(config.FALLBACK_HEAD)
# Вырезка — ровно как при обучении (ml/extract_crops.py): поля 8%, длинная сторона не больше 512 px,
# сохранение в JPEG 90. Сжатие тоже повторяем: без него вектор отличается от обучающего
# (сходство 0.956–0.98 на 20 машинах), с ним совпадает до 1.0000.
CROP_MARGIN = 0.08
CROP_LONG_SIDE = 512
CROP_JPEG_QUALITY = 90
UNKNOWN = "не видно / не определить"
OTHER = "другое"
FIELDS = ("type", "make", "model")
# Марка в разметке -> как её пишет приложение (для страны и сравнения с реестром и надписью)
MAKE_NAMES = {"Lada/ВАЗ": "ВАЗ (Lada)"}

_lock = threading.Lock()
_state: dict | None = None
_failed: str | None = None


def training_crop(img_bgr: np.ndarray, bbox) -> np.ndarray:
    crop = annotate.crop_with_margin(img_bgr, list(bbox), margin=CROP_MARGIN)
    h, w = crop.shape[:2]
    scale = CROP_LONG_SIDE / max(h, w)
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, CROP_JPEG_QUALITY])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else crop


def display_make(name: str | None) -> str | None:
    return MAKE_NAMES.get(name, name) if name else None


def enabled() -> bool:
    if str(config.FALLBACK_ENABLED).strip().lower() in ("off", "no", "false", "0"):
        return False
    return HEAD_PATH.exists() and _failed is None


def load_heads(path: Path = HEAD_PATH) -> tuple[str, dict, dict]:
    """(имя модели-«глаз», головы по полям, модель машины -> марка) из файла ml/train_head.py."""
    data = np.load(path)
    heads = {}
    for f in FIELDS:
        if f"{f}_coef" in data:
            heads[f] = {"classes": [str(c) for c in data[f"{f}_classes"]],
                        "coef": data[f"{f}_coef"].astype(np.float32),
                        "intercept": data[f"{f}_intercept"].astype(np.float32),
                        "threshold": float(data[f"{f}_threshold"])}
    return str(data["encoder"]), heads, json.loads(str(data["model_make"]))


def predict(heads: dict, x: np.ndarray) -> dict:
    """Ответы голов для одного L2-нормированного вектора: значение, вероятность, уверен ли."""
    out = {}
    for field, h in heads.items():
        z = h["coef"] @ x + h["intercept"]
        p = np.exp(z - z.max())
        p /= p.sum()
        i = int(p.argmax())
        value, conf = h["classes"][i], float(p[i])
        out[field] = {"value": value, "confidence": round(conf, 3), "threshold": round(h["threshold"], 3),
                      # «не видно» и «другое» — не ответ: в отчёт такое не подставляем
                      "confident": conf >= h["threshold"] and value not in (UNKNOWN, OTHER)}
    return out


def _load() -> dict | None:
    global _state, _failed
    with _lock:
        if _state is None and _failed is None:
            try:
                import torch
                from transformers import AutoImageProcessor, SiglipVisionModel

                encoder, heads, model_make = load_heads()
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                # Только «зрительная» часть SigLIP2: она и даёт вектор картинки (get_image_features),
                # текстовая здесь не нужна и заняла бы ещё гигабайт памяти.
                model = SiglipVisionModel.from_pretrained(encoder, dtype=dtype).to(device).eval()
                processor = AutoImageProcessor.from_pretrained(encoder)
                _state = {"model": model, "processor": processor, "device": device, "dtype": dtype,
                          "heads": heads, "model_make": model_make, "encoder": encoder}
                print(f"Запасной классификатор готов: {encoder} на {device}, головы {', '.join(heads)}")
            except Exception as e:  # нет пакетов/весов — работаем без него
                _failed = f"{type(e).__name__}: {e}"
                print(f"Запасной классификатор недоступен ({_failed}) — марку по виду машины не определяем")
        return _state


def embed(img_bgr: np.ndarray, bbox) -> np.ndarray | None:
    st = _load()
    if st is None:
        return None
    import torch
    from PIL import Image

    rgb = cv2.cvtColor(training_crop(img_bgr, bbox), cv2.COLOR_BGR2RGB)
    px = st["processor"](images=Image.fromarray(rgb), return_tensors="pt")["pixel_values"]
    with _lock, torch.inference_mode():
        feats = st["model"](pixel_values=px.to(st["device"], st["dtype"])).pooler_output.float()
    return torch.nn.functional.normalize(feats, dim=-1)[0].cpu().numpy()


def classify(img_bgr: np.ndarray, bbox) -> dict | None:
    """Тип, марка, модель по виду машины в рамке bbox. None — классификатор выключен или недоступен."""
    if not enabled():
        return None
    x = embed(img_bgr, bbox)
    if x is None:
        return None
    return predict(_state["heads"], x)


def make_of_model(model: str) -> str | None:
    """Марка, к которой относится модель машины по разметке («К-744» -> «Кировец»)."""
    return (_state or {}).get("model_make", {}).get(model)
