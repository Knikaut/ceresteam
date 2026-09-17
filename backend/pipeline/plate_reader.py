"""Поиск и чтение госномеров: детектор номера (YOLOv9, ONNX) + OCR (fast-plate-ocr, ONNX).

Номер на обзорной камере весовой занимает 50–150 px из 2688, поэтому детектор
запускается не на всём кадре, а на вырезке каждого найденного ТС. OCR
выполняется в нескольких вариантах (две модели × несколько предобработок, для
двухстрочных номеров — построчно), а результат выбирается голосованием с учётом
форматов номеров РК (см. kz_plates).

Перенесено из соседнего решения команды (agro_alpr). Прежний путь через EasyOCR читал
4 номера из 16 на кадрах весовой, этот — 12 из 16 на тех же кадрах: общая модель для
текста на фотографиях не справляется с табличками в 35-130 пикселей.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.pipeline import kz_plates

CPU = ["CPUExecutionProvider"]


@dataclass
class PlateBox:
    box: tuple[int, int, int, int]
    conf: float


class PlateReader:
    def __init__(self, detector_model: str = "yolo-v9-s-608-license-plate-end2end",
                 ocr_models: tuple[str, ...] = ("cct-xs-v2-global-model", "cct-s-v2-global-model"),
                 det_conf: float = 0.15, min_conf: float = 0.3):
        from fast_plate_ocr import LicensePlateRecognizer
        from open_image_models import LicensePlateDetector

        self.detector = LicensePlateDetector(detection_model=detector_model, conf_thresh=det_conf, providers=CPU)
        self.ocrs = [LicensePlateRecognizer(m, providers=CPU) for m in ocr_models]
        self.min_conf = min_conf

    # ---------- детекция ----------
    def detect(self, image: np.ndarray, region: tuple[int, int, int, int]) -> list[PlateBox]:
        """Номера внутри области ТС. Крупные области дополнительно режутся на
        перекрывающиеся фрагменты: детектор работает на входе 608 px, и мелкий
        номер на большой вырезке иначе теряется."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = region
        pad_x, pad_y = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        windows = [(x1, y1, x2, y2)]
        rw, rh = x2 - x1, y2 - y1
        if max(rw, rh) > 700:
            nx, ny = (2 if rw > 700 else 1), (2 if rh > 700 else 1)
            tw, th = int(rw / nx * 1.3) if nx > 1 else rw, int(rh / ny * 1.3) if ny > 1 else rh
            for i in range(nx):
                for j in range(ny):
                    tx = x1 + (0 if nx == 1 else int(i * (rw - tw) / (nx - 1)))
                    ty = y1 + (0 if ny == 1 else int(j * (rh - th) / (ny - 1)))
                    windows.append((tx, ty, tx + tw, ty + th))
        boxes = []
        for wx1, wy1, wx2, wy2 in windows:
            crop = image[wy1:wy2, wx1:wx2]
            if crop.size == 0:
                continue
            for d in self.detector.predict(crop):
                b = d.bounding_box
                boxes.append(PlateBox((b.x1 + wx1, b.y1 + wy1, b.x2 + wx1, b.y2 + wy1), float(d.confidence)))
        return _nms(boxes)

    # ---------- чтение ----------
    def _probs(self, crop_bgr: np.ndarray) -> list[tuple[np.ndarray, str, str]]:
        """Матрицы вероятностей (слоты × алфавит) от каждой OCR-модели."""
        from fast_plate_ocr.core.process import preprocess_image
        from fast_plate_ocr.inference.plate_recognizer import _load_image_from_source

        out = []
        for ocr in self.ocrs:
            cfg = ocr.config
            code = cv2.COLOR_BGR2GRAY if cfg.image_color_mode == "grayscale" else cv2.COLOR_BGR2RGB
            x = preprocess_image(_load_image_from_source(cv2.cvtColor(crop_bgr, code), cfg))
            raw = ocr.model.run([ocr.plate_output_name], {"input": x})[0]
            out.append((raw.reshape(-1, cfg.max_plate_slots, len(cfg.alphabet))[0], cfg.alphabet, cfg.pad_char))
        return out

    @staticmethod
    def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
        if not angle:
            return img
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _is_blue(img: np.ndarray) -> bool:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue = (hsv[..., 0] > 85) & (hsv[..., 0] < 135) & (hsv[..., 1] > 50) & (hsv[..., 2] > 60)
        return blue.mean() > 0.25

    def _variants(self, crop: np.ndarray) -> list[np.ndarray]:
        """Варианты вырезки: поворот (номера на широкоугольной камере наклонены),
        увеличение, а для синих тракторных номеров — инверсия яркости."""
        big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        out = [self._rotate(big, a) for a in (0, -12, -6, 6, 12)] + [crop]
        if self._is_blue(crop):
            inv = [cv2.cvtColor(255 - cv2.cvtColor(v, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) for v in out]
            out += inv
        return out

    def read(self, image: np.ndarray, box: tuple[int, int, int, int]) -> kz_plates.PlateResult:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 6:
            return kz_plates.PlateResult()
        crop = image[max(0, y1 - bh // 6):min(h, y2 + bh // 6), max(0, x1 - bw // 12):min(w, x2 + bw // 12)]
        reads, groups = [], []
        # Однострочное чтение выполняется всегда: из-за ракурса рамка обычного номера
        # (520×112 мм) бывает почти квадратной. Двухстрочный номер модель тоже иногда
        # читает «в одну строку» — такие прочтения разбираются двухстрочными раскладками.
        for v in self._variants(crop):
            for P, a, pad in self._probs(v):
                reads.append(kz_plates.decode_single(P, a, pad)); groups.append(0)
        if bw / bh < 3.0:  # возможно двухстрочный номер (280×200 мм) — читаем построчно
            ch = crop.shape[0]
            top, bottom = _widen(crop[: int(ch * 0.55)]), _widen(crop[int(ch * 0.45):])
            for vt, vb in zip(self._variants(top), self._variants(bottom)):
                for (pt, a, pad), (pb, _, _) in zip(self._probs(vt), self._probs(vb)):
                    reads.append(kz_plates.decode_two_rows(pt, pb, a, pad)); groups.append(1)
        return kz_plates.vote(reads, self.min_conf, groups)


def _widen(row: np.ndarray, aspect: float = 3.5) -> np.ndarray:
    """Строка двухстрочного номера почти квадратная, а OCR обучен на вытянутых номерах:
    добавляем поля по бокам."""
    h, w = row.shape[:2]
    extra = max(0, int(h * aspect) - w)
    return cv2.copyMakeBorder(row, 0, 0, extra // 2, extra - extra // 2, cv2.BORDER_REPLICATE)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _nms(boxes: list[PlateBox], thr: float = 0.4) -> list[PlateBox]:
    kept: list[PlateBox] = []
    for b in sorted(boxes, key=lambda b: -b.conf):
        if all(_iou(b.box, k.box) < thr for k in kept):
            kept.append(b)
    return kept
