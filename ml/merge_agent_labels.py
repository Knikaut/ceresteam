"""Метки агентов -> labels.json страницы разметки и выбор проверочной выборки.

1. Переносит метки агентов в labels.json (ручные метки человека не перетирает).
2. Выбирает ~100 вырезок для проверки человеком: по одной на машину (группу), с разбивкой
   по маркам и камерам, чтобы проверка покрывала все частые марки, а не только КамАЗ.
   Эти вырезки помечаются test=true и в обучение не попадут вместе со всеми повторами машины.

    python ml/merge_agent_labels.py <папка labeling> <agent_labels.json> [--test 100]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

UNKNOWN = "не видно / не определить"

# Агенты пишут одну модель по-разному: «ВАЗ-2121 Нива», «Нива (ВАЗ-2121)» — приводим к одному виду.
MODEL_ALIASES = [("2121", "ВАЗ-2121 Нива"), ("55111", "КамАЗ-55111"), ("65115", "КамАЗ-65115"),
                 ("5511", "КамАЗ-5511"), ("3307", "ГАЗ-3307"), ("3309", "ГАЗ-3309"), ("ГАЗ-53", "ГАЗ-53"),
                 ("130", "ЗИЛ-130"), ("744", "К-744"), ("701", "К-701"), ("700", "К-700"), ("МТЗ-82", "МТЗ-82")]


def normalize_model(model: str | None) -> str | None:
    if not model:
        return None
    for key, canonical in MODEL_ALIASES:
        if key in model:
            return canonical
    return model.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("labeling")
    ap.add_argument("agent_labels")
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    lab = Path(args.labeling)
    items = {it["crop_id"]: it for it in json.loads((lab / "items.json").read_text(encoding="utf-8"))["items"]}
    labels_path = lab / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    agent = json.loads(Path(args.agent_labels).read_text(encoding="utf-8"))

    added = 0
    for l in agent:
        l["model"] = normalize_model(l.get("model"))
        cid = l["crop_id"]
        if cid not in items or labels.get(cid, {}).get("human"):
            continue
        labels[cid] = {"type": l["type"], "make": l["make"], "model": l.get("model"),
                       "confidence": l.get("confidence"), "view": l.get("view"),
                       "source": l.get("source", "agent"),
                       # исходный ответ агентов — чтобы после проверки человеком измерить и их точность
                       "agent": {"type": l["type"], "make": l["make"], "model": l.get("model")}}
        added += 1

    # Проверочная выборка: одна вырезка на группу (машину), квоты по маркам
    rng = random.Random(args.seed)
    by_make: dict[str, list[str]] = defaultdict(list)
    used_groups = set()
    candidates = [cid for cid, l in labels.items() if l.get("make") and l.get("type")]
    rng.shuffle(candidates)
    for cid in candidates:
        g = items[cid]["group"]
        if g in used_groups:
            continue
        used_groups.add(g)
        by_make[labels[cid]["make"]].append(cid)
    makes = sorted(by_make, key=lambda m: -len(by_make[m]))
    total = sum(len(v) for v in by_make.values())
    # В проверку берём не больше четверти машин каждой марки: иначе у редкой марки (8 ЗИЛов)
    # в обучении не останется примеров и проверять будет нечего. Марки, где меньше 8 машин,
    # в проверку не идут вовсе — модель их всё равно не выучит (уйдут в «другое»).
    test: list[str] = []
    for m in makes:
        n = len(by_make[m])
        if n < 8:
            continue
        share = max(2, round(args.test * n / total))
        test += by_make[m][: min(share, n // 4)]
    rest = [c for m in makes if len(by_make[m]) >= 8 for c in by_make[m][len(by_make[m]) // 4:]]
    rng.shuffle(rest)
    test += rest[: max(0, args.test - len(test))]
    test = test[: args.test]
    test_set = set(test)
    quota = "до 1/4 машин марки"
    for cid, l in labels.items():
        l["test"] = cid in test_set

    tmp = labels_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(labels_path)
    (lab / "test_ids.json").write_text(json.dumps(sorted(test_set), ensure_ascii=False, indent=1), encoding="utf-8")

    from collections import Counter
    print(f"меток от агентов добавлено: {added} | всего меток: {len(labels)}")
    print("марки (все метки):", Counter(l["make"] for l in labels.values()).most_common())
    print("типы (все метки):", Counter(l["type"] for l in labels.values()).most_common())
    print(f"проверочная выборка: {len(test)} вырезок / {len(test)} машин | квота на марку {quota}")
    print("марки в проверке:", Counter(labels[c]["make"] for c in test).most_common())
    return 0


if __name__ == "__main__":
    sys.exit(main())
