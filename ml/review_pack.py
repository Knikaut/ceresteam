"""Пакет для экрана проверки: крупная вырезка машины и весь кадр — для каждой проверочной машины.

На ноутбуке есть только превью 320 px, а полные кадры лежат на сервере в /data (только чтение).
Скрипт режет машину из исходного кадра с запасом по краям (без увеличения сверх оригинала)
и уменьшает кадр целиком — чтобы человек видел, есть ли прицеп и что вокруг машины.
Рамку машины рисует сама страница по review_meta.json, на картинках её нет.

    python ml/review_pack.py --crops /workspace/ml/crops.jsonl --labels /workspace/ml/labeling/labels.json \
        --out /workspace/ml/review
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

MARGIN = 0.12        # запас вокруг рамки детектора, доля её ширины/высоты
CROP_MAX = 1000      # длинная сторона вырезки, px (меньше — оставляем как есть)
FRAME_MAX = 1400     # длинная сторона кадра целиком, px


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True, help="crops.jsonl от extract_crops.py")
    ap.add_argument("--labels", required=True, help="labels.json: берутся вырезки с test=true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    wanted = {cid for cid, l in labels.items() if l.get("test")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = {}
    with open(args.crops, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cid = r["crop_id"]
            if cid not in wanted:
                continue
            frame = Image.open(r["image"]).convert("RGB")
            W, H = frame.size
            x1, y1, x2, y2 = r["bbox"]
            mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
            box = (max(0, int(x1 - mx)), max(0, int(y1 - my)), min(W, int(x2 + mx)), min(H, int(y2 + my)))
            crop = frame.crop(box)
            crop.thumbnail((CROP_MAX, CROP_MAX), Image.LANCZOS)
            crop.save(out / f"{cid}_crop.jpg", quality=88)
            small = frame.copy()
            small.thumbnail((FRAME_MAX, FRAME_MAX), Image.LANCZOS)
            small.save(out / f"{cid}_frame.jpg", quality=80)
            meta[cid] = {"bbox": [round(x1 / W, 4), round(y1 / H, 4), round(x2 / W, 4), round(y2 / H, 4)],
                         "frame_wh": [W, H], "conf": r.get("conf"), "yolo_class": r.get("yolo_class")}

    (out / "review_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    missing = sorted(wanted - meta.keys())
    print(f"готово: {len(meta)} из {len(wanted)} проверочных машин -> {out}")
    if missing:
        print("нет в crops.jsonl:", missing)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
