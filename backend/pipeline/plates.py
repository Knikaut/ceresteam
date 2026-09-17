"""Чтение государственных номеров Казахстана по OCR.

Два прохода:
1. OCR по всей вырезке техники — находим текстовые области и надписи (марка).
2. Каждую область-кандидат (текст из OCR + белые прямоугольники нужных пропорций)
   увеличиваем и читаем повторно с ограниченным алфавитом A-Z0-9, затем
   проверяем по шаблонам номеров РК.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from backend import config

# Кириллические буквы, которые OCR путает с латинскими на номерах.
_CYR_TO_LAT = str.maketrans("АВЕКМНОРСТУХаеорсух", "ABEKMHOPCTYXaeopcyx")
# Исправление символов по позиции: в позиции цифры / в позиции буквы.
_TO_DIGIT = str.maketrans("OQDIlZSBGTA", "00011258674")
_TO_LETTER = str.maketrans("012456789", "OIZASGTBQ")

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PLATE_ALLOWLIST = LETTERS + "0123456789 "
VALID_REGIONS = {f"{i:02d}" for i in range(1, 21)}  # коды регионов РК 01..20

# Шаблоны: D = цифра, L = буква.
PLATE_TEMPLATES = [
    ("kz_2012", "DDDLLLDD", "{0} {1} {2}"),   # 041 AHF 10 (стандарт с 2012 г.)
    ("kz_2012", "DDDLLDD", "{0} {1} {2}"),    # 681 AS 10
    ("painted", "DDDDDLL", "{0} {1} {2}"),    # 10 625 AZ (нарисован на борту)
    # Старый формат «A 123 BCD» (до 2012 г.) отключён: шаблон слишком свободный
    # и ловит мусор с мелких номеров. Включить при необходимости.
]

# Ключевые слова производителей, которые OCR может прочитать на облицовке/стекле.
BRAND_KEYWORDS = {
    "МАЗ": ["МАЗ", "MAZ"],
    "КамАЗ": ["КАМАЗ", "KAMAZ"],
    "ГАЗ": ["ГАЗ", "GAZ"],
    "Кировец": ["КИРОВЕЦ", "KIROVETS", "KIROVEC"],
    "ЗИЛ": ["ЗИЛ", "ZIL"],
    "Урал": ["УРАЛ", "URAL"],
    "Scania": ["SCANIA"],
    "Volvo": ["VOLVO"],
    "MAN": ["MAN"],
    "Mercedes-Benz": ["MERCEDES", "ACTROS"],
    "Howo": ["HOWO", "SINOTRUK"],
    "Shacman": ["SHACMAN", "SHAANXI"],
    "Isuzu": ["ISUZU"],
    "Hyundai": ["HYUNDAI"],
    "DAF": ["DAF"],
    "Iveco": ["IVECO"],
    "МТЗ": ["МТЗ", "BELARUS", "БЕЛАРУС"],
}


@dataclass
class OcrBox:
    text: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def h(self) -> int:
        return self.y2 - self.y1


@dataclass
class PlateResult:
    text: str                 # компактно: 041AHF10
    formatted: str            # 041 AHF 10
    kind: str                 # kz_2012 / painted
    region_code: str | None
    confidence: float
    bbox: list[int]           # в координатах исходного кадра
    raw: str = ""
    region_valid: bool = True

    votes: float = 0.0        # доля прочтений, совпавших с этим вариантом (0..1)

    @property
    def score(self) -> float:
        order = {"kz_2012": 2.0, "painted": 0.5}.get(self.kind, 0.0)
        return order + (0.5 if self.region_valid else -1.0) + self.confidence + 2 * self.votes


_reader = None


def get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(config.OCR_LANGS, gpu=False, verbose=False)
    return _reader


def ocr_image(img_bgr: np.ndarray, min_conf: float = 0.1, allowlist: str | None = None) -> list[OcrBox]:
    reader = get_reader()
    kwargs = {"allowlist": allowlist} if allowlist else {}
    out = []
    for box, text, conf in reader.readtext(img_bgr, detail=1, paragraph=False, **kwargs):
        if conf < min_conf or not text.strip():
            continue
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        out.append(OcrBox(text, float(conf), min(xs), min(ys), max(xs), max(ys)))
    return out


# ---------- нормализация и шаблоны ----------

def normalize(text: str) -> str:
    t = text.upper().translate(_CYR_TO_LAT)
    t = re.sub(r"[^A-Z0-9]", "", t)
    # Флаг/код страны на номере читается как KZ в начале или конце.
    t = re.sub(r"^KZ", "", t)
    t = re.sub(r"KZ$", "", t)
    return t


def fit_template(s: str, template: str) -> str | None:
    """Подгоняет строку под шаблон, исправляя похожие символы по позиции."""
    if len(s) != len(template):
        return None
    out = []
    for ch, kind in zip(s, template):
        if kind == "D":
            ch2 = ch if ch.isdigit() else ch.translate(_TO_DIGIT)
            if not ch2.isdigit():
                return None
        else:
            ch2 = ch if ch.isalpha() else ch.translate(_TO_LETTER)
            if not ch2.isalpha():
                return None
        out.append(ch2)
    return "".join(out)


def match_plate(raw: str) -> tuple[str, str, str, str | None, bool] | None:
    """Возвращает (kind, compact, formatted, region, region_valid) либо None."""
    s = normalize(raw)
    if not s:
        return None
    for kind, template, fmt in PLATE_TEMPLATES:
        fixed = fit_template(s, template)
        if fixed is None:
            continue
        groups = re.findall(r"\d+|[A-Z]+", fixed)
        if kind == "painted":
            region = groups[0][:2]
            formatted = f"{region} {groups[0][2:]} {groups[1]}"
        else:
            region = groups[-1]
            formatted = fmt.format(*groups)
        return kind, fixed, formatted, region, region in VALID_REGIONS
    return None


def group_lines(boxes: list[OcrBox]) -> list[list[OcrBox]]:
    """Склеивает блоки, стоящие на одной строке (номер часто читается частями)."""
    boxes = sorted(boxes, key=lambda b: (b.cy, b.x1))
    lines: list[list[OcrBox]] = []
    for b in boxes:
        for line in lines:
            ref = line[-1]
            same_row = abs(b.cy - ref.cy) < max(ref.h, b.h) * 0.7
            close = b.x1 - ref.x2 < max(ref.h, b.h) * 2.5
            if same_row and close:
                line.append(b)
                break
        else:
            lines.append([b])
    return lines


def _candidate_strings(boxes: list[OcrBox]):
    for b in boxes:
        yield b.text, [b]
    for line in group_lines(boxes):
        if len(line) > 1:
            yield " ".join(b.text for b in line), line
            if len(line) > 2:
                yield " ".join(b.text for b in line[1:]), line[1:]
                yield " ".join(b.text for b in line[:-1]), line[:-1]
    if len(boxes) > 1:
        ordered = sorted(boxes, key=lambda b: b.x1)
        yield " ".join(b.text for b in ordered), ordered


def plates_from_boxes(boxes: list[OcrBox], offset=(0, 0), scale: float = 1.0) -> list[PlateResult]:
    """Выбирает номера из OCR-блоков; offset/scale переводят координаты в исходный кадр."""
    found: dict[str, PlateResult] = {}
    for raw, parts in _candidate_strings(boxes):
        m = match_plate(raw)
        if not m:
            continue
        kind, compact, formatted, region, valid = m
        conf = float(np.mean([p.conf for p in parts]))
        x1 = min(p.x1 for p in parts); y1 = min(p.y1 for p in parts)
        x2 = max(p.x2 for p in parts); y2 = max(p.y2 for p in parts)
        bbox = [int(x1 / scale + offset[0]), int(y1 / scale + offset[1]),
                int(x2 / scale + offset[0]), int(y2 / scale + offset[1])]
        res = PlateResult(compact, formatted, kind, region, conf, bbox, raw=raw, region_valid=valid)
        if compact not in found or found[compact].score < res.score:
            found[compact] = res
    return sorted(found.values(), key=lambda r: -r.score)


# ---------- поиск областей-кандидатов ----------

def _white_rectangles(crop_bgr: np.ndarray) -> list[list[int]]:
    """Светлые прямоугольники с пропорциями номерной таблички (≈4.6:1)."""
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch < 8 or cw < 0.04 * w or cw > 0.7 * w:
            continue
        aspect = cw / float(ch)
        if 2.2 <= aspect <= 7.5 and cv2.contourArea(c) > 0.55 * cw * ch:
            rects.append([x, y, x + cw, y + ch])
    return rects


def _expand(rect: list[int], w: int, h: int, fx: float = 0.25, fy: float = 0.5) -> list[int]:
    x1, y1, x2, y2 = rect
    mx, my = int((x2 - x1) * fx), int((y2 - y1) * fy)
    return [max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)]


def _dedupe(rects: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for r in rects:
        for o in out:
            ix = max(0, min(r[2], o[2]) - max(r[0], o[0]))
            iy = max(0, min(r[3], o[3]) - max(r[1], o[1]))
            inter = ix * iy
            if inter > 0.5 * min((r[2] - r[0]) * (r[3] - r[1]), (o[2] - o[0]) * (o[3] - o[1])):
                o[:] = [min(r[0], o[0]), min(r[1], o[1]), max(r[2], o[2]), max(r[3], o[3])]
                break
        else:
            out.append(list(r))
    return out


def _variants(img: np.ndarray):
    """Варианты предобработки: на мелких номерах OCR читает их по-разному, устраиваем голосование."""
    yield img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
    yield clahe
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield otsu


def _letters_between(img: np.ndarray, boxes: list[OcrBox]) -> tuple[str, float] | None:
    """Номер РК: 3 цифры, буквы, 2 цифры региона. Буквы в середине OCR часто читает
    как цифры — дочитываем полосу между цифровыми блоками с алфавитом только из букв."""
    digits = [b for b in boxes if normalize(b.text).isdigit()]
    first = [b for b in digits if len(normalize(b.text)) == 3]
    last = [b for b in digits if len(normalize(b.text)) == 2]
    if not first or not last:
        return None
    f = min(first, key=lambda b: b.x1)
    l = max(last, key=lambda b: b.x1)
    if l.x1 - f.x2 < f.h * 0.8:
        return None
    y1 = max(0, min(f.y1, l.y1) - f.h // 4)
    y2 = min(img.shape[0], max(f.y2, l.y2) + f.h // 4)
    strip = img[y1:y2, max(0, f.x2 - f.h // 6):min(img.shape[1], l.x1 + f.h // 6)]
    if strip.size == 0:
        return None
    strip, _ = upscale_for_ocr(strip, target_width=300, max_scale=4.0)
    reads = ocr_image(strip, min_conf=0.05, allowlist=LETTERS + " ")
    letters = normalize("".join(b.text for b in sorted(reads, key=lambda b: b.x1)))
    if 2 <= len(letters) <= 3 and letters.isalpha():
        conf = float(np.mean([b.conf for b in reads]))
        return normalize(f.text) + letters + normalize(l.text), conf
    return None


def read_plate_region(sub_bgr: np.ndarray, offset=(0, 0)) -> list[PlateResult]:
    """Читает номер в области-кандидате несколькими способами и голосует."""
    tally: dict[str, list] = {}   # compact -> [PlateResult, votes]
    total = 0
    for target_w in (650,):
        up, s = upscale_for_ocr(sub_bgr, target_width=target_w, max_scale=14.0)
        for variant in _variants(up):
            total += 1
            boxes = ocr_image(variant, min_conf=0.05, allowlist=PLATE_ALLOWLIST)
            reads = plates_from_boxes(boxes, offset, s)
            refined = _letters_between(variant, boxes)
            if refined:
                m = match_plate(refined[0])
                if m:
                    kind, compact, formatted, region, valid = m
                    x1 = min(b.x1 for b in boxes); y1 = min(b.y1 for b in boxes)
                    x2 = max(b.x2 for b in boxes); y2 = max(b.y2 for b in boxes)
                    bbox = [int(x1 / s + offset[0]), int(y1 / s + offset[1]),
                            int(x2 / s + offset[0]), int(y2 / s + offset[1])]
                    reads.insert(0, PlateResult(compact, formatted, kind, region, refined[1], bbox,
                                                raw=refined[0], region_valid=valid))
            seen = set()
            for p in reads:
                if p.text in seen:
                    continue
                seen.add(p.text)
                if p.text not in tally:
                    tally[p.text] = [p, 0.0]
                tally[p.text][1] += 1.0
                if p.confidence > tally[p.text][0].confidence:
                    tally[p.text][0] = p
    out = []
    for p, votes in tally.values():
        p.votes = votes / max(total, 1)
        out.append(p)
    return sorted(out, key=lambda p: -p.score)


def read_vehicle(crop_bgr: np.ndarray, offset=(0, 0)) -> tuple[list[PlateResult], list[OcrBox]]:
    """Возвращает (номера, все OCR-блоки первого прохода) для вырезки техники."""
    h, w = crop_bgr.shape[:2]
    up, scale = upscale_for_ocr(crop_bgr, target_width=1300)
    boxes = ocr_image(up)

    # Кандидаты (по приоритету): блоки с цифрами -> светлые прямоугольники -> прочий текст.
    ranked: list[tuple[int, list[int]]] = []
    for b in boxes:
        txt = normalize(b.text)
        rect = [int(b.x1 / scale), int(b.y1 / scale), int(b.x2 / scale), int(b.y2 / scale)]
        if sum(ch.isdigit() for ch in txt) >= 2:
            ranked.append((0, rect))
        elif len(txt) >= 2:
            ranked.append((2, rect))
    for line in group_lines(boxes):
        if len(line) > 1:
            ranked.append((0, [int(min(b.x1 for b in line) / scale), int(min(b.y1 for b in line) / scale),
                               int(max(b.x2 for b in line) / scale), int(max(b.y2 for b in line) / scale)]))
    for r in _white_rectangles(crop_bgr):
        ranked.append((1, r))
    ranked.sort(key=lambda t: t[0])
    rects = _dedupe([_expand(r, w, h) for _, r in ranked])[:4]

    results: dict[str, PlateResult] = {}
    # Первый проход тоже может дать номер (крупные надписи на борту).
    for p in plates_from_boxes(boxes, offset, scale):
        results[p.text] = p
    for r in rects:
        sub = crop_bgr[r[1]:r[3], r[0]:r[2]]
        if sub.size == 0:
            continue
        for p in read_plate_region(sub, (offset[0] + r[0], offset[1] + r[1])):
            if p.text not in results or results[p.text].score < p.score:
                results[p.text] = p
    plates = sorted(results.values(), key=lambda p: -p.score)
    return plates, boxes


def brand_from_boxes(boxes: list[OcrBox]) -> str | None:
    text = " ".join(b.text for b in boxes).upper()
    for brand, keys in BRAND_KEYWORDS.items():
        for k in keys:
            if re.search(rf"(?<![A-ZА-Я]){re.escape(k)}(?![A-ZА-Я])", text):
                return brand
    return None


def upscale_for_ocr(crop: np.ndarray, target_width: int = 1100, max_scale: float = 4.0) -> tuple[np.ndarray, float]:
    h, w = crop.shape[:2]
    scale = min(max_scale, max(1.0, target_width / max(w, 1)))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop, scale


# ---------- дата/время с наложенной надписи камеры ----------

DATE_RE = re.compile(r"(\d{2})-?(\d{2})-?(20\d{2})")
TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})")


def read_overlay_timestamp(img_bgr: np.ndarray) -> str | None:
    """Читает дату/время с надписи камеры (левый верхний угол, формат MM-DD-YYYY HH:MM:SS)."""
    h, w = img_bgr.shape[:2]
    region = img_bgr[0:int(h * 0.12), 0:int(w * 0.45)]
    region, _ = upscale_for_ocr(region, target_width=1800, max_scale=2.5)
    boxes = ocr_image(region, min_conf=0.05, allowlist="0123456789-: ")
    text = " ".join(b.text.replace(" ", "") for b in sorted(boxes, key=lambda b: b.x1))
    d, t = DATE_RE.search(text), TIME_RE.search(text)
    if not d or not t:
        return None
    mm, dd, yyyy = d.groups()
    hh, mi, ss = t.groups()
    if not (1 <= int(mm) <= 12 and 1 <= int(dd) <= 31 and int(hh) < 24 and int(mi) < 60 and int(ss) < 60):
        return None
    return f"{yyyy}-{mm}-{dd} {hh}:{mi}:{ss}"
