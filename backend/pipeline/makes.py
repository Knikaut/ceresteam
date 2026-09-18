"""Справочник производителей: страна по марке. Модель для этого не нужна."""
from __future__ import annotations

# Ключ — начало названия марки в нижнем регистре (как пишут реестр, OCR и VLM).
COUNTRY_BY_MAKE = {
    "маз": "Беларусь", "maz": "Беларусь",
    "мтз": "Беларусь", "беларус": "Беларусь", "belarus": "Беларусь",
    "белаз": "Беларусь", "belaz": "Беларусь",
    "камаз": "Россия", "kamaz": "Россия",
    "газ": "Россия", "gaz": "Россия",
    "зил": "Россия", "zil": "Россия",
    "урал": "Россия", "ural": "Россия",
    "кировец": "Россия", "kirovets": "Россия", "кировец к": "Россия",
    "ваз": "Россия", "lada": "Россия", "уаз": "Россия", "uaz": "Россия",
    "ростсельмаш": "Россия", "rostselmash": "Россия", "дон": "Россия",
    "shacman": "Китай", "shaanxi": "Китай", "howo": "Китай", "sinotruk": "Китай",
    "faw": "Китай", "dongfeng": "Китай", "foton": "Китай", "jac": "Китай", "sany": "Китай",
    "xcmg": "Китай", "sdlg": "Китай", "lonking": "Китай", "camc": "Китай", "beiben": "Китай",
    "lovol": "Китай", "ловол": "Китай",
    "scania": "Швеция", "volvo": "Швеция",
    "man": "Германия", "mercedes": "Германия", "daf": "Нидерланды", "iveco": "Италия",
    "renault": "Франция",
    "isuzu": "Япония", "hino": "Япония", "mitsubishi": "Япония", "toyota": "Япония", "nissan": "Япония",
    "hyundai": "Южная Корея", "kia": "Южная Корея", "daewoo": "Южная Корея",
    "john deere": "США", "case": "США", "new holland": "США", "caterpillar": "США", "cat": "США",
    "claas": "Германия", "fendt": "Германия", "deutz": "Германия",
}


# Одна марка в разных написаниях (кириллица/латиница, старые названия) -> общий ключ.
ALIASES = {
    "maz": "маз", "kamaz": "камаз", "gaz": "газ", "zil": "зил", "ural": "урал",
    "kirovets": "кировец", "kirovec": "кировец",
    "belarus": "мтз", "беларус": "мтз", "беларусь": "мтз", "mtz": "мтз",
    "belaz": "белаз", "uaz": "уаз", "lada": "ваз",
    "shaanxi": "shacman", "шакман": "shacman", "sinotruk": "howo", "хово": "howo",
    "мерседес": "mercedes", "mercedes-benz": "mercedes", "вольво": "volvo", "скания": "scania",
}
# Марки из нескольких слов: их нельзя резать по первому слову.
MULTI_WORD = ("john deere", "new holland", "mercedes-benz")


def canonical(name: str | None) -> str | None:
    """Нормализованный ключ марки: «МАЗ-5551», «MAZ» -> «маз»; «Кировец К-700» -> «кировец»."""
    if not name:
        return None
    key = name.strip().lower().replace("ё", "е")
    for multi in MULTI_WORD:
        if key == multi or key.startswith(multi + " "):
            return ALIASES.get(multi, multi)
    word = key.replace("(", " ").split()[0] if key.split() else ""
    word = word.split("-")[0] if word.split("-")[0] else word
    word = word.strip(".,;:\"'«»")
    return ALIASES.get(word, word) or None


def same_make(a: str | None, b: str | None) -> bool:
    """Одна ли это марка с учётом написания. Пустое значение ни с чем не совпадает."""
    ca, cb = canonical(a), canonical(b)
    return bool(ca and cb and ca == cb)


def country_for(manufacturer: str | None) -> str | None:
    if not manufacturer:
        return None
    key = manufacturer.strip().lower()
    # Сначала длинные ключи: «john deere» раньше, чем «cat» случайно совпадёт с началом слова.
    for name in sorted(COUNTRY_BY_MAKE, key=len, reverse=True):
        if key == name or key.startswith(name + " ") or key.startswith(name + "-") or key.startswith(name + "("):
            return COUNTRY_BY_MAKE[name]
    return None
