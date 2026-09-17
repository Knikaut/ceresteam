"""Нормализация и валидация госномеров Республики Казахстан.

Перенесено из соседнего решения команды (agro_alpr) без изменений логики: там этот разбор
измерен на тех же 25 кадрах весовой и дал 12 точных прочтений из 16 против 4 у прежнего.

OCR часто путает похожие символы (O/0, B/8, I/1 ...). У номера РК жёсткая
позиционная структура: для каждой раскладки известно, где цифра, где буква
и где код региона. Поэтому символы исправляются по позиции, а из всех
интерпретаций выбирается та, что даёт корректный код региона.

Источники: СТ РК 986-2012, приказ МВД №1040 (коды 01–17), МВД 22.06.2022
(коды 18–20), СТ РК 1176 и Приказ МСХ №4-3/267 (номера тракторов).
С 2018 года допустимы все 26 латинских букв.
"""
from __future__ import annotations

import re

import numpy as np
from dataclasses import dataclass, field

REGIONS = {
    "01": "г. Астана", "02": "г. Алматы", "03": "Акмолинская область",
    "04": "Актюбинская область", "05": "Алматинская область", "06": "Атырауская область",
    "07": "Западно-Казахстанская область", "08": "Жамбылская область",
    "09": "Карагандинская область", "10": "Костанайская область",
    "11": "Кызылординская область", "12": "Мангистауская область",
    "13": "Туркестанская область", "14": "Павлодарская область",
    "15": "Северо-Казахстанская область", "16": "Восточно-Казахстанская область",
    "17": "г. Шымкент", "18": "область Абай", "19": "область Жетісу", "20": "область Ұлытау",
}

# Буквенные коды регионов номеров образца 1993 года (O, W, K, U, V – упразднённые области).
OLD_1993_LETTERS = {
    "Z": "г. Астана", "A": "г. Алматы", "C": "Акмолинская область",
    "D": "Актюбинская область", "B": "Алматинская область", "E": "Атырауская область",
    "L": "Западно-Казахстанская область", "H": "Жамбылская область",
    "M": "Карагандинская область", "P": "Костанайская область",
    "N": "Кызылординская область", "R": "Мангистауская область",
    "X": "Туркестанская область / г. Шымкент", "S": "Павлодарская область",
    "T": "Северо-Казахстанская область", "F": "Восточно-Казахстанская область",
    "O": "Кокшетауская область (упразднена)", "W": "Тургайская область (упразднена)",
    "K": "Жезказганская область (упразднена)", "U": "Семипалатинская область (упразднена)",
    "V": "Талдыкорганская область (упразднена)",
}

# Буквенные коды регионов на номерах тракторов и самоходной техники (СТ РК 1176, Приложение 9).
TRACTOR_LETTERS = {
    "Z": "г. Астана", "A": "г. Алматы", "C": "Акмолинская область",
    "D": "Актюбинская область", "B": "Алматинская область", "E": "Атырауская область",
    "L": "Западно-Казахстанская область", "H": "Жамбылская область",
    "M": "Карагандинская область", "P": "Костанайская область",
    "N": "Кызылординская область", "R": "Мангистауская область",
    "X": "Туркестанская область", "S": "Павлодарская область",
    "T": "Северо-Казахстанская область", "F": "Восточно-Казахстанская область",
    "Y": "г. Шымкент", "G": "область Абай", "Q": "область Жетісу", "W": "область Ұлытау",
}

# Символы маски: D – цифра, L – буква, R – цифра кода региона, r – буква региона.
DIGIT_SLOTS, LETTER_SLOTS = "DR", "Lr"


@dataclass(frozen=True)
class Layout:
    name: str
    title: str
    mask: str             # порядок символов так, как их читает OCR
    order: tuple          # индексы групп для канонического вида
    groups: tuple         # размеры групп в порядке чтения
    rows: int = 1
    split: int = 0        # символов в верхней строке (для двухстрочных)

    def canonical(self, text: str) -> tuple[str, str]:
        parts, i = [], 0
        for g in self.groups:
            parts.append(text[i:i + g]); i += g
        ordered = [parts[k] for k in self.order]
        return "".join(ordered), " ".join(ordered)


LAYOUTS = (
    # Однострочные
    Layout("car_2012", "Номер 2012 (физлицо)", "DDDLLLRR", (0, 1, 2), (3, 3, 2)),
    Layout("car_2012_legal", "Номер 2012 (юрлицо)", "DDDLLRR", (0, 1, 2), (3, 2, 2)),
    Layout("trailer_2012", "Номер прицепа 2012", "DDLLLRR", (0, 1, 2), (2, 3, 2)),
    Layout("old_1993", "Номер образца 1993", "rDDDLLL", (0, 1, 2), (1, 3, 3)),
    Layout("old_1993_legal", "Номер образца 1993 (юрлицо)", "rDDDLL", (0, 1, 2), (1, 3, 2)),
    # Двухстрочные (верхняя строка + нижняя)
    Layout("truck_2012_2row", "Задний номер грузовика 2012", "DDDRRLLL", (0, 2, 1), (3, 2, 3), 2, 3),
    Layout("truck_2012_legal_2row", "Задний номер грузовика 2012 (юрлицо)", "DDDRRLL", (0, 2, 1), (3, 2, 2), 2, 3),
    Layout("trailer_2012_2row", "Номер прицепа 2012 (квадратный)", "LLLRRDD", (2, 0, 1), (3, 2, 2), 2, 3),
    Layout("tractor_2row", "Номер трактора / тракторного прицепа", "LLLrDDD", (1, 2, 0), (3, 1, 3), 2, 3),
    Layout("moto_2012_2row", "Номер мотоцикла 2012", "DDRRLL", (0, 2, 1), (2, 2, 2), 2, 2),
)

TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1", "T": "7",
            "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T", "4": "A"}
CYR_TO_LAT = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")

# Редкие на весовой раскладки получают небольшой штраф.
LAYOUT_PENALTY = {"moto_2012_2row": 0.1, "old_1993_legal": 0.03}

HOME_REGIONS = {"10", "P"}  # Костанайская область – небольшой приоритет при равенстве


@dataclass
class Candidate:
    text: str              # канонический номер без пробелов
    formatted: str
    layout: Layout
    region_code: str
    region_name: str
    score: float
    fixes: int


@dataclass
class PlateResult:
    text: str | None = None
    formatted: str | None = None
    layout: str | None = None
    layout_title: str | None = None
    region_code: str | None = None
    region_name: str | None = None
    confidence: float = 0.0
    readable: bool = False
    raw_reads: list = field(default_factory=list)


def clean(raw: str) -> str:
    s = (raw or "").upper().translate(CYR_TO_LAT)
    return re.sub(r"[^A-Z0-9]", "", s)


def _fit(s: str, probs, layout: Layout):
    out, fixes, conf = [], 0, 0.0
    for i, (ch, kind) in enumerate(zip(s, layout.mask)):
        p = probs[i] if probs is not None and i < len(probs) else 0.8
        if kind in DIGIT_SLOTS and not ch.isdigit():
            ch = TO_DIGIT.get(ch)
            fixes += 1; p *= 0.6
        elif kind in LETTER_SLOTS and not ch.isalpha():
            ch = TO_LETTER.get(ch)
            fixes += 1; p *= 0.6
        if ch is None:
            return None
        out.append(ch); conf += p
    return "".join(out), fixes, conf / len(s)


def candidates(raw: str, probs=None, rows: int | None = None) -> list[Candidate]:
    """Все допустимые интерпретации строки OCR (rows – если известно число строк номера)."""
    s = clean(raw)
    if probs is not None and len(probs) != len(s):
        probs = None
    variants = [(s, probs, 0.0)]
    # Лишний символ (рамка, флаг, повтор): пробуем удалить по одному.
    for i in range(len(s)):
        variants.append((s[:i] + s[i + 1:], None if probs is None else list(probs[:i]) + list(probs[i + 1:]), 0.15))
    if s.startswith("KZ"):
        variants.append((s[2:], None if probs is None else list(probs[2:]), 0.0))
    out: dict[str, Candidate] = {}
    for v, pv, penalty in variants:
        for lay in LAYOUTS:
            if len(v) != len(lay.mask):
                continue
            fit = _fit(v, pv, lay)
            if not fit:
                continue
            text, fixes, conf = fit
            code = "".join(c for c, k in zip(text, lay.mask) if k in "Rr")
            if "R" in lay.mask:
                name = REGIONS.get(code)
            else:
                name = (TRACTOR_LETTERS if lay.name.startswith("tractor") else OLD_1993_LETTERS).get(code)
            if not name:
                continue
            canon, formatted = lay.canonical(text)
            score = (conf - penalty - 0.05 * fixes + (0.05 if code in HOME_REGIONS else 0.0)
                     - LAYOUT_PENALTY.get(lay.name, 0.0) - (0.05 if rows and rows != lay.rows else 0.0))
            if canon not in out or out[canon].score < score:
                out[canon] = Candidate(canon, formatted, lay, code, name, score, fixes)
    return sorted(out.values(), key=lambda c: -c.score)


DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LAYOUT_PRIOR = {"moto_2012_2row": 0.7, "old_1993_legal": 0.9, "old_1993": 0.95}
SKIP_LOGP = float(np.log(0.05))  # штраф за «лишний» символ, вставленный OCR


def _region_names(layout: Layout) -> dict:
    return TRACTOR_LETTERS if layout.name.startswith("tractor") else OLD_1993_LETTERS


def _decode_part(logp: np.ndarray, alphabet: str, pad: str, mask: str, layout: Layout, skip: int | None):
    """Лучшее заполнение маски по матрице log-вероятностей (слоты × алфавит).
    Возвращает (символы, сумма log p) или None. skip – индекс слота, который считается мусором."""
    idx = {c: i for i, c in enumerate(alphabet)}
    slots = logp.shape[0]
    need = len(mask) + (skip is not None)
    if need > slots:
        return None
    chars, total, slot, i = [], 0.0, 0, 0
    while i < len(mask):
        if slot == skip:
            total += float(logp[slot].max()) + SKIP_LOGP
            slot += 1
            continue
        kind = mask[i]
        if kind == "R":
            if skip == slot + 1:
                return None
            best = max(REGIONS, key=lambda c: logp[slot, idx[c[0]]] + logp[slot + 1, idx[c[1]]])
            chars += list(best)
            total += float(logp[slot, idx[best[0]]] + logp[slot + 1, idx[best[1]]])
            slot += 2; i += 2
            continue
        pool = DIGITS if kind == "D" else LETTERS if kind == "L" else "".join(_region_names(layout))
        best = max(pool, key=lambda c: logp[slot, idx[c]])
        chars.append(best); total += float(logp[slot, idx[best]])
        slot += 1; i += 1
    if slot == skip:
        total += float(logp[slot].max()) + SKIP_LOGP
        slot += 1
    total += float(logp[slot:, idx[pad]].sum())
    return "".join(chars), total


def _candidate(text: str, layout: Layout, mean_logp: float) -> Candidate | None:
    code = "".join(c for c, k in zip(text, layout.mask) if k in "Rr")
    name = REGIONS.get(code) if "R" in layout.mask else _region_names(layout).get(code)
    if not name:
        return None
    canon, formatted = layout.canonical(text)
    score = float(np.exp(mean_logp)) * LAYOUT_PRIOR.get(layout.name, 1.0) * (1.1 if code in HOME_REGIONS else 1.0)
    return Candidate(canon, formatted, layout, code, name, min(score, 1.0), 0)


def decode_single(probs: np.ndarray, alphabet: str, pad: str) -> list[Candidate]:
    """Однострочное прочтение: матрица (слоты × алфавит) → кандидаты по всем раскладкам."""
    logp = np.log(np.clip(probs, 1e-6, 1.0))
    out: dict[str, Candidate] = {}
    for lay in LAYOUTS:
        for skip in (None, *range(len(lay.mask) + 1)):
            r = _decode_part(logp, alphabet, pad, lay.mask, lay, skip)
            if r is None:
                continue
            c = _candidate(r[0], lay, r[1] / logp.shape[0])
            if c and (c.text not in out or out[c.text].score < c.score):
                out[c.text] = c
    return sorted(out.values(), key=lambda c: -c.score)


def decode_two_rows(top: np.ndarray, bottom: np.ndarray, alphabet: str, pad: str) -> list[Candidate]:
    """Двухстрочный номер: верхняя и нижняя строки распознаны отдельно."""
    lt, lb = np.log(np.clip(top, 1e-6, 1.0)), np.log(np.clip(bottom, 1e-6, 1.0))
    out: dict[str, Candidate] = {}
    for lay in LAYOUTS:
        if lay.rows != 2:
            continue
        a = _decode_part(lt, alphabet, pad, lay.mask[:lay.split], lay, None)
        b = _decode_part(lb, alphabet, pad, lay.mask[lay.split:], lay, None)
        if a is None or b is None:
            continue
        c = _candidate(a[0] + b[0], lay, (a[1] + b[1]) / (lt.shape[0] + lb.shape[0]))
        if c and (c.text not in out or out[c.text].score < c.score):
            out[c.text] = c
    return sorted(out.values(), key=lambda c: -c.score)


def vote(reads: list[list[Candidate]], min_conf: float = 0.3, groups: list[int] | None = None) -> PlateResult:
    """Голосование по прочтениям одного номера (каждое прочтение – список кандидатов).

    Каждое прочтение голосует только за своего лучшего кандидата. groups – номер группы
    прочтения (0 – целиком, 1 – построчно): уверенность считается внутри группы, чтобы
    построчные прочтения однострочного номера не «размывали» результат."""
    groups = groups or [0] * len(reads)
    size = {g: groups.count(g) for g in set(groups)}
    total: dict[tuple[int, str], float] = {}
    best: dict[str, Candidate] = {}
    for cands, g in zip(reads, groups):
        if not cands:
            continue
        c = cands[0]
        total[(g, c.text)] = total.get((g, c.text), 0.0) + c.score
        if c.text not in best or best[c.text].score < c.score:
            best[c.text] = c
    res = PlateResult()
    if not total:
        return res
    conf = {}
    for (g, t), v in total.items():
        conf[t] = max(conf.get(t, 0.0), v / size[g])
    text = max(conf, key=conf.get)
    c = best[text]
    res.text, res.formatted, res.layout, res.layout_title = c.text, c.formatted, c.layout.name, c.layout.title
    res.region_code, res.region_name = c.region_code, c.region_name
    res.confidence = round(min(1.0, conf[text]), 3)
    res.readable = res.confidence >= min_conf
    res.raw_reads = sorted(((t, round(v, 3)) for t, v in conf.items()), key=lambda x: -x[1])[:5]
    return res


def parse(raw: str) -> PlateResult:
    """Разбор готовой строки (для других OCR и ручной проверки)."""
    return vote([candidates(raw)], min_conf=0.0)
