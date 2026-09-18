"""Госномера для экрана проверки — прочитанные так же, как их читает приложение.

Для каждой проверочной машины из исходного кадра вырезается её рамка (как в analyze.analyze_image),
номер читается plates.read_vehicle, решение «читается / нет» — analyze.plate_verdict (те же пороги).
Результат дописывается в review/review_meta.json (поле plate), увеличенная табличка сохраняется
в review/<id>_plate.jpg — чтобы человек видел номер крупно и мог сверить или вписать его.
Потом по этим полям и по номерам, введённым человеком, считается точность чтения номеров.

    python ml/plate_suggest.py results/ml/labeling          # нужны frames/ (исходные кадры + boxes.json)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.pipeline import analyze, annotate, plates  # noqa: E402

PLATE_ZOOM_WIDTH = 420   # ширина увеличенной таблички на экране проверки, px


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("labeling", help="папка с frames/boxes.json и review/review_meta.json")
    ap.add_argument("--force", action="store_true", help="перечитать и те, что уже прочитаны")
    args = ap.parse_args()

    lab = Path(args.labeling)
    boxes = json.loads((lab / "frames" / "boxes.json").read_text(encoding="utf-8"))
    meta_path = lab / "review" / "review_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print("номера читает:", "специализированный распознаватель" if plates.use_fast_reader() else "EasyOCR")

    t0, done = time.time(), 0
    for cid, b in boxes.items():
        if cid not in meta or ("plate" in meta[cid] and not args.force):
            continue
        img = annotate.imread(lab / "frames" / b["frame"])
        if img is None:
            print("не открылся кадр", b["frame"])
            continue
        H, W = img.shape[:2]
        x1, y1, x2, y2 = b["bbox"]
        found, _ = plates.read_vehicle(img[y1:y2, x1:x2], offset=(x1, y1))
        if found:
            best = found[0]
            reliability, readable = analyze.plate_verdict(best)
            zoom = annotate.crop_with_margin(img, best.bbox, margin=0.6)
            zoom = cv2.resize(zoom, None, fx=PLATE_ZOOM_WIDTH / zoom.shape[1], fy=PLATE_ZOOM_WIDTH / zoom.shape[1],
                              interpolation=cv2.INTER_CUBIC)
            annotate.imwrite(lab / "review" / f"{cid}_plate.jpg", zoom, 92)
            px1, py1, px2, py2 = best.bbox
            meta[cid]["plate"] = {
                "found": True, "readable": bool(readable), "reliability": round(float(reliability), 3),
                # как в отчёте приложения: ненадёжное прочтение не выдаётся за номер, остаётся догадкой
                "text": best.text if readable else None,
                "formatted": best.formatted if readable else None,
                "guess_text": best.text, "guess_formatted": best.formatted,
                "alternatives": [p.formatted for p in found[1:4]],
                "bbox": [round(px1 / W, 4), round(py1 / H, 4), round(px2 / W, 4), round(py2 / H, 4)],
                "source": getattr(best, "source", ""),
            }
        else:
            meta[cid]["plate"] = {"found": False, "readable": False}
        done += 1
        p = meta[cid]["plate"]
        if p["readable"]:
            said = f"{p['formatted']}  (надёжность {p['reliability']})"
        elif p["found"]:
            said = f"ненадёжно, догадка {p['guess_formatted']}  (надёжность {p['reliability']})"
        else:
            said = "номер не найден"
        print(f"{done:3d}. {cid}: {said}", flush=True)
        save_json(meta_path, meta)   # после каждой машины: прерванный запуск продолжится с места

    have = [m["plate"] for m in meta.values() if "plate" in m]
    print(f"готово за {time.time() - t0:.0f} с: номер прочитан у {sum(p['readable'] for p in have)} из {len(have)}, "
          f"найден, но ненадёжен — {sum(p['found'] and not p['readable'] for p in have)}, "
          f"не найден — {sum(not p['found'] for p in have)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
