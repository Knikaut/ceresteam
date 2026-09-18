"""Шаг 1 запасного классификатора: вырезки машин из всех кадров датасета.

Детектор проходит по кадрам (обход подпапок: камера/год/месяц/день), сохраняет
вырезку каждой заметной машины и строку в crops.jsonl. Прогон возобновляется:
кадры из processed.txt повторно не считаются.

    python ml/extract_crops.py --input /data/Auto --output /workspace/ml --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.pipeline import annotate, detector  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png"}
# Машины меньше этой доли кадра — фон: по ним марку не определить ни человеку, ни модели.
MIN_AREA_FRACTION = 0.01
MIN_CONF = 0.35
CROP_LONG_SIDE = 512      # SigLIP2 всё равно смотрит на 384 px; запас под поля
CROP_MARGIN = 0.08


def camera_and_date(path: Path, root: Path) -> tuple[str, str]:
    """/data/Auto/3/2026/09/06/x.jpg -> ("3", "2026-09-06")."""
    parts = path.relative_to(root).parts
    camera = parts[0] if len(parts) > 1 else ""
    date = "-".join(parts[1:4]) if len(parts) >= 5 else ""
    return camera, date


def save_crop(img, bbox: list[int], dst: Path) -> None:
    crop = annotate.crop_with_margin(img, bbox, margin=CROP_MARGIN)
    h, w = crop.shape[:2]
    scale = CROP_LONG_SIDE / max(h, w)
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    annotate.imwrite(dst, crop, 90)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root, out = Path(args.input), Path(args.output)
    crops_dir = out / "crops"
    out.mkdir(parents=True, exist_ok=True)
    done_file, meta_file = out / "processed.txt", out / "crops.jsonl"
    done = set(done_file.read_text(encoding="utf-8").split("\n")) if done_file.exists() else set()

    frames = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXT and p.is_file())
    todo = [p for p in frames if str(p) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"кадров: {len(frames)} | уже обработано: {len(done)} | к обработке: {len(todo)} "
          f"| детектор: {detector.set_device(args.device)}", flush=True)

    t0, n_crops = time.time(), 0
    with meta_file.open("a", encoding="utf-8") as meta, done_file.open("a", encoding="utf-8") as done_f:
        for i, path in enumerate(todo, 1):
            try:
                img = annotate.imread(path)
                if img is None:
                    raise ValueError("не читается")
                h, w = img.shape[:2]
                vehicles = [d for d in detector.detect(img) if d.cls in detector.VEHICLE_CLASSES]
                vehicles.sort(key=lambda d: -d.area)
                camera, date = camera_and_date(path, root)
                for k, v in enumerate(vehicles):
                    frac = v.area / float(w * h)
                    if frac < MIN_AREA_FRACTION or v.conf < MIN_CONF:
                        continue
                    crop_id = f"{camera or 'x'}_{path.stem}_{k}"
                    dst = crops_dir / (camera or "x") / f"{crop_id}.jpg"
                    save_crop(img, v.bbox, dst)
                    meta.write(json.dumps({
                        "crop_id": crop_id, "crop": str(dst.relative_to(out)), "image": str(path),
                        "camera": camera, "date": date, "yolo_class": v.cls, "conf": round(v.conf, 3),
                        "bbox": v.bbox, "area_frac": round(frac, 4), "rank": k, "frame_wh": [w, h],
                    }, ensure_ascii=False) + "\n")
                    n_crops += 1
            except Exception as e:  # битый кадр не должен ронять прогон
                print(f"[{i}] {path}: пропущен ({type(e).__name__}: {e})", flush=True)
            done_f.write(str(path) + "\n")
            if i % 50 == 0 or i == len(todo):
                meta.flush(); done_f.flush()
                spent = time.time() - t0
                print(f"[{i}/{len(todo)}] вырезок: {n_crops} | {spent / i:.2f} c/кадр "
                      f"| осталось ~{(len(todo) - i) * spent / i / 60:.0f} мин", flush=True)
    print(f"готово: вырезок {n_crops} за {(time.time() - t0) / 60:.1f} мин", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
