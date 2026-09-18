"""Шаг 4: обучение запасного классификатора (тип / марка / модель).

SigLIP2 заморожена, учится только логистическая регрессия поверх её эмбеддингов.
Обучение — на метках агентов (двое независимых + третий при споре).
Проверка — ТОЛЬКО на проверочных вырезках, подтверждённых человеком (test=true, verified=true).
Все повторы проверочных машин из обучения исключены: одна машина не бывает и там, и там.

    python ml/train_head.py --output /workspace/ml --labels /workspace/ml/labeling/labels.json

На выходе: head_<модель>.joblib (для пайплайна) и metrics_<модель>.json (для отчёта).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

UNKNOWN = "не видно / не определить"
OTHER = "другое"
TARGET_PRECISION = 0.9   # порог уверенности: уверенные ответы должны быть верны в ≥90% случаев
MIN_PER_CLASS = 8        # реже — класс сливается в «другое» (модель по 3 примерам не выучит марку)
FIELDS = ("type", "make", "model")


def pick_threshold(proba: np.ndarray, correct: np.ndarray) -> float:
    """Минимальный порог, при котором среди ответов выше порога доля верных ≥ TARGET_PRECISION."""
    order = np.argsort(-proba)
    hits = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.where(hits >= TARGET_PRECISION)[0]
    return float(proba[order][ok.max()]) if len(ok) else 1.01


def export_npz(heads: dict, path: Path, encoder: str, model_make: dict) -> None:
    """Веса голов в простом виде для приложения: классы, коэффициенты, порог — без scikit-learn.

    Вероятности в приложении: softmax(x @ coef.T + intercept) — ровно то, что даёт predict_proba
    у многоклассовой логистической регрессии; ниже это сверяется, прежде чем файл записать.
    """
    arrays = {"encoder": np.array(encoder), "model_make": np.array(json.dumps(model_make, ensure_ascii=False))}
    for field, h in heads.items():
        clf = h["model"]
        arrays[f"{field}_classes"] = np.array([str(c) for c in clf.classes_])
        arrays[f"{field}_coef"] = clf.coef_.astype(np.float32)
        arrays[f"{field}_intercept"] = clf.intercept_.astype(np.float32)
        arrays[f"{field}_threshold"] = np.array(h["threshold"], dtype=np.float32)
        probe = np.random.default_rng(0).normal(size=(8, clf.coef_.shape[1]))
        z = probe @ clf.coef_.T + clf.intercept_
        manual = np.exp(z - z.max(1, keepdims=True))
        manual /= manual.sum(1, keepdims=True)
        assert np.abs(manual - clf.predict_proba(probe)).max() < 1e-5, f"{field}: softmax не совпал с predict_proba"
    np.savez_compressed(path, **arrays)


def acc(pred, truth, mask=None) -> float | None:
    pred, truth = np.asarray(pred), np.asarray(truth)
    m = np.ones(len(truth), bool) if mask is None else mask
    return round(float((pred[m] == truth[m]).mean()), 3) if m.any() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model", default="google/siglip2-so400m-patch16-384")
    ap.add_argument("--no-propagate", action="store_true", help="не переносить метку на повторы той же машины")
    ap.add_argument("--final", action="store_true",
                    help="итоговая модель для приложения: учиться на всех размеченных машинах, без проверочной "
                         "выборки (точность берётся из прогона без --final)")
    args = ap.parse_args()

    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict

    out, tag = Path(args.output), args.model.split("/")[-1]
    emb = np.load(out / f"embeddings_{tag}.npy").astype(np.float32)
    ids = [x for x in (out / f"embeddings_{tag}.ids").read_text(encoding="utf-8").split("\n") if x]
    index = {cid: i for i, cid in enumerate(ids)}
    zs = {r["crop_id"]: r for r in map(json.loads, (out / f"zero_shot_{tag}.jsonl").read_text(encoding="utf-8").splitlines())}
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))

    group_of = {cid: r["group"] for cid, r in zs.items()}
    members: dict[int, list[str]] = {}
    for cid, g in group_of.items():
        members.setdefault(g, []).append(cid)

    if args.final:
        # Итоговая модель: проверочные машины тоже идут в обучение, точность меряется в обычном прогоне
        test_ids, verified, test_groups = [], [], set()
        print("итоговая модель: обучение на всех размеченных машинах, без проверочной выборки")
    else:
        test_ids = [c for c, l in labels.items() if l.get("test") and c in index]
        verified = [c for c in test_ids if labels[c].get("verified") and labels[c].get("make") and labels[c].get("type")]
        test_groups = {group_of[c] for c in test_ids}
        print(f"проверочных вырезок: {len(test_ids)} | подтверждено человеком: {len(verified)}")
        if len(verified) < 20:
            print("мало подтверждённых проверочных вырезок — нужно хотя бы 20, иначе цифре точности нельзя верить")
            return 1

    # Обучение: всё размеченное, кроме машин из проверки; метка переносится на повторы машины
    train_rows = []
    for cid, l in labels.items():
        if cid not in index or group_of[cid] in test_groups or not (l.get("make") and l.get("type")):
            continue
        if l.get("test") and not args.final:
            continue
        for m in ([cid] if args.no_propagate else members.get(group_of[cid], [cid])):
            if group_of[m] not in test_groups:
                train_rows.append((m, group_of[m], l))
    X = np.stack([emb[index[m]] for m, _, _ in train_rows])
    groups = np.array([g for _, g, _ in train_rows])
    Xt = np.stack([emb[index[c]] for c in verified]) if verified else np.zeros((0, X.shape[1]), np.float32)
    print(f"обучение: {len(train_rows)} вырезок ({len(set(groups))} машин) | проверка: {len(verified)} машин")

    heads, metrics = {}, {"train_crops": len(train_rows), "train_vehicles": int(len(set(groups))),
                          "test_verified": len(verified), "encoder": args.model}
    for field in FIELDS:
        y_raw = [l.get(field) or "" for _, _, l in train_rows]
        keep = np.array([bool(v) for v in y_raw])
        if keep.sum() < 40:
            metrics[field] = {"skipped": f"примеров {int(keep.sum())} — мало"}
            continue
        counts = Counter(v for v in y_raw if v)
        y = np.array([v if counts[v] >= MIN_PER_CLASS else OTHER for v in y_raw])[keep]
        Xf, gf = X[keep], groups[keep]
        classes = sorted(set(y))
        if len(classes) < 2:
            metrics[field] = {"skipped": "один класс"}
            continue

        clf = LogisticRegression(C=2.0, max_iter=4000, class_weight="balanced")
        folds = min(5, len(set(gf)))
        cv_proba = cross_val_predict(clf, Xf, y, groups=gf, cv=GroupKFold(folds), method="predict_proba")
        cv_pred = np.array(classes)[cv_proba.argmax(1)]
        known = y != UNKNOWN
        threshold = pick_threshold(cv_proba.max(1)[known], (cv_pred == y)[known])
        clf.fit(Xf, y)
        if args.final:
            metrics[field] = {"classes": classes, "threshold": round(threshold, 3),
                              "cv_accuracy_on_train": acc(cv_pred, y, known)}
            heads[field] = {"model": clf, "threshold": threshold}
            print(f"[{field}] классов {len(classes)} | точность на кросс-валидации {metrics[field]['cv_accuracy_on_train']} "
                  f"| порог уверенности {round(threshold, 3)}")
            continue

        # Истина, ответы агентов и zero-shot приводятся к одному списку классов модели:
        # редкое значение (КамАЗ-5410 при 3 примерах) у всех считается как «другое».
        to_space = lambda v: v if v in classes or v in ("", UNKNOWN) else OTHER  # noqa: E731
        truth_raw = [labels[c].get(field) or "" for c in verified]
        has_truth = np.array([bool(v) for v in truth_raw])
        truth = np.array([to_space(v) for v in truth_raw])
        proba = clf.predict_proba(Xt)
        pred = clf.classes_[proba.argmax(1)]
        conf = proba.max(1)
        confident = conf >= threshold
        known_t = has_truth & (truth != UNKNOWN)
        unknown_t = has_truth & (truth == UNKNOWN)
        agent_pred = np.array([to_space((labels[c].get("agent") or {}).get(field) or "") for c in verified])
        zs_pred = np.array([to_space(zs[c][field][0][0]) if field in ("type", "make") else "" for c in verified])

        report = {
            "classes": classes, "threshold": round(threshold, 3),
            "test_with_truth": int(has_truth.sum()), "test_known": int(known_t.sum()),
            "model_accuracy": acc(pred, truth, known_t),
            # вместе с «не видно» как отдельным ответом — на всех проверочных машинах с меткой
            "model_accuracy_all": acc(pred, truth, has_truth),
            # человек не смог определить, а модель уверенно назвала конкретное значение — проверить нельзя
            "test_unknown": int(unknown_t.sum()),
            "model_named_on_unknown": int((unknown_t & confident & (pred != UNKNOWN) & (pred != OTHER)).sum()),
            "model_confident_share": round(float(confident[known_t].mean()), 3) if known_t.any() else None,
            "model_confident_accuracy": acc(pred, truth, known_t & confident),
            "zero_shot_accuracy": acc(zs_pred, truth, known_t) if field != "model" else None,
            "agent_labels_accuracy": acc(agent_pred, truth, known_t),
            "cv_accuracy_on_train": acc(cv_pred, y, known),
            "confusion": Counter(f"{t} -> {p}" for t, p in zip(truth[known_t], pred[known_t]) if t != p).most_common(12),
        }
        metrics[field] = report
        heads[field] = {"model": clf, "threshold": threshold}
        print(f"[{field}] точность {report['model_accuracy']} на {report['test_known']} (с «не видно»: "
              f"{report['model_accuracy_all']}) | уверенных {report['model_confident_share']} "
              f"с точностью {report['model_confident_accuracy']} (порог {report['threshold']}) | "
              f"без обучения {report['zero_shot_accuracy']} | метки агентов {report['agent_labels_accuracy']} | "
              f"назвала там, где «не видно»: {report['model_named_on_unknown']} из {report['test_unknown']}")

    # Модель машины -> её марка (самая частая в разметке): приложение не поставит «К-744» к КамАЗу
    pairs = Counter((l.get("model"), l.get("make")) for _, _, l in train_rows if l.get("model"))
    model_make = {}
    for (model, make), _ in pairs.most_common():
        model_make.setdefault(model, make)

    suffix = "_final" if args.final else ""
    joblib.dump({"encoder": args.model, "heads": heads}, out / f"head_{tag}{suffix}.joblib")
    export_npz(heads, out / f"head_{tag}{suffix}.npz", args.model, model_make)
    (out / f"metrics_{tag}{suffix}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"сохранено: head_{tag}{suffix}.joblib / .npz, metrics_{tag}{suffix}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
