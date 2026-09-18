"""Шаг 3: подсказки без обучения (zero-shot) и подбор вырезок для ручной разметки.

1. Для каждой вырезки — топ-3 типа и марки по сходству с текстовыми описаниями (SigLIP2).
2. Почти одинаковые вырезки (одна и та же машина на соседних кадрах) склеиваются в группы.
3. Группы раскладываются по кластерам, из каждого кластера берётся несколько примеров —
   так разметка покрывает все виды техники, а не только самый частый КамАЗ.
4. На выходе папка labeling/ с уменьшенными картинками и items.json для страницы разметки.

    python ml/prepare_labeling.py --output /workspace/ml --per-cluster 8 --clusters 160
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy  # noqa: E402

DUP_COS = 0.965      # выше — считаем той же машиной на соседнем кадре
THUMB = 320


def text_embeddings(model, tokenizer, classes: dict[str, list[str]], device) -> tuple[list[str], np.ndarray]:
    names, vecs = [], []
    for name, phrases in classes.items():
        texts = [taxonomy.TEMPLATE.format(p) for p in phrases]
        tok = tokenizer(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            f = model.get_text_features(**tok)
            if not torch.is_tensor(f):
                f = f.pooler_output
        f = torch.nn.functional.normalize(f.float(), dim=-1).mean(0)
        vecs.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
        names.append(name)
    return names, np.stack(vecs)


def top3(scores: np.ndarray, names: list[str]) -> list[list]:
    # SigLIP обучена с сигмоидой, но для ранжирования достаточно softmax по логитам с её масштабом.
    idx = np.argsort(-scores, axis=1)[:, :3]
    e = np.exp((scores - scores.max(1, keepdims=True)) * 30)
    p = e / e.sum(1, keepdims=True)
    return [[[names[j], round(float(p[i, j]), 3)] for j in row] for i, row in enumerate(idx)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="google/siglip2-so400m-patch16-384")
    ap.add_argument("--clusters", type=int, default=160)
    ap.add_argument("--per-cluster", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.cluster import MiniBatchKMeans
    from transformers import AutoModel, AutoTokenizer

    out = Path(args.output)
    tag = args.model.split("/")[-1]
    emb = np.load(out / f"embeddings_{tag}.npy").astype(np.float32)
    ids = [x for x in (out / f"embeddings_{tag}.ids").read_text(encoding="utf-8").split("\n") if x]
    meta = {r["crop_id"]: r for r in map(json.loads, (out / "crops.jsonl").read_text(encoding="utf-8").splitlines())}
    print(f"вырезок с эмбеддингами: {len(ids)}", flush=True)

    # 1. Zero-shot подсказки
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    type_names, type_vecs = text_embeddings(model, tokenizer, taxonomy.TYPES, device)
    make_names, make_vecs = text_embeddings(model, tokenizer, taxonomy.MAKES, device)
    type_top, make_top = top3(emb @ type_vecs.T, type_names), top3(emb @ make_vecs.T, make_names)

    # 2. Группы почти одинаковых вырезок: жадно, внутри одной камеры, по порядку кадров
    order = sorted(range(len(ids)), key=lambda i: (meta[ids[i]]["camera"], meta[ids[i]]["image"]))
    group = [-1] * len(ids)
    reps: list[int] = []
    for i in order:
        cam = meta[ids[i]]["camera"]
        recent = [r for r in reps[-60:] if meta[ids[r]]["camera"] == cam]
        if recent:
            sims = emb[recent] @ emb[i]
            j = int(np.argmax(sims))
            if sims[j] >= DUP_COS:
                group[i] = group[recent[j]]
                continue
        group[i] = len(reps)
        reps.append(i)
    sizes = np.bincount(group)
    print(f"групп (разных машин/ракурсов): {len(reps)} из {len(ids)} вырезок", flush=True)

    # 3. Кластеры по представителям групп и выбор примеров
    rep_emb = emb[reps]
    k = min(args.clusters, len(reps))
    km = MiniBatchKMeans(n_clusters=k, random_state=args.seed, batch_size=2048, n_init=3).fit(rep_emb)
    dist = np.linalg.norm(rep_emb - km.cluster_centers_[km.labels_], axis=1)
    rng = np.random.default_rng(args.seed)
    chosen: list[int] = []
    for c in range(k):
        members = np.where(km.labels_ == c)[0]
        if not len(members):
            continue
        near = members[np.argsort(dist[members])][: max(1, args.per_cluster // 2)]
        rest = np.setdiff1d(members, near)
        far = rng.choice(rest, size=min(len(rest), args.per_cluster - len(near)), replace=False) if len(rest) else []
        chosen.extend(int(reps[m]) for m in list(near) + list(far))
    print(f"к разметке выбрано: {len(chosen)} вырезок из {k} кластеров", flush=True)

    # 4. Экспорт для страницы разметки
    lab = out / "labeling"
    (lab / "img").mkdir(parents=True, exist_ok=True)
    items = []
    for i in chosen:
        cid = ids[i]
        m = meta[cid]
        img = Image.open(out / m["crop"]).convert("RGB")
        img.thumbnail((THUMB, THUMB))
        img.save(lab / "img" / f"{cid}.jpg", quality=85)
        items.append({
            "crop_id": cid, "camera": m["camera"], "date": m["date"], "image": Path(m["image"]).name,
            "cluster": int(km.labels_[reps.index(i)]), "group": int(group[i]), "group_size": int(sizes[group[i]]),
            "yolo_class": m["yolo_class"], "type_hint": type_top[i], "make_hint": make_top[i],
        })
    items.sort(key=lambda r: (r["cluster"], r["camera"], r["date"]))
    (lab / "items.json").write_text(json.dumps({
        "types": type_names + [taxonomy.OTHER, taxonomy.UNKNOWN],
        "makes": make_names + [taxonomy.OTHER, taxonomy.UNKNOWN],
        "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")

    # Подсказки по всем вырезкам — пригодятся для сравнения «без обучения» и отчёта
    with (out / f"zero_shot_{tag}.jsonl").open("w", encoding="utf-8") as f:
        for i, cid in enumerate(ids):
            f.write(json.dumps({"crop_id": cid, "group": int(group[i]), "type": type_top[i], "make": make_top[i]},
                               ensure_ascii=False) + "\n")
    from collections import Counter
    print("типы (zero-shot, топ-1):", Counter(t[0][0] for t in type_top).most_common())
    print("марки (zero-shot, топ-1):", Counter(t[0][0] for t in make_top).most_common())
    return 0


if __name__ == "__main__":
    sys.exit(main())
