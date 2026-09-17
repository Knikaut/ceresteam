"""Пакетная обработка папки с кадрами (обязательная часть задания 2).

Пример:
    python run_batch.py --input /data/Auto --output results/batch

На выходе:
    <output>/detections.json      — все кадры одним файлом
    <output>/detections.csv       — то же в таблице
    <output>/<кадр>.json          — отчёт по каждому кадру
    <output>/<кадр>_annotated.jpg — кадр с рамками
    <output>/<кадр>_plate.jpg     — увеличенный номер (если найден)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import config  # noqa: E402
from backend.pipeline import analyze, export  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(config.DATA_DIR), help="папка с кадрами")
    ap.add_argument("--output", default=str(config.RESULTS_DIR / "batch"), help="папка для результатов")
    ap.add_argument("--limit", type=int, default=0, help="обработать только первые N кадров")
    ap.add_argument("--no-vlm", action="store_true", help="не вызывать VLM (марка/модель только по надписям)")
    ap.add_argument("--no-registry", action="store_true", help="не искать номер в открытых данных data.egov.kz")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    images = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXT],
                    key=lambda p: (len(p.stem), p.stem))
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"В папке {in_dir} нет изображений")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    t0 = time.time()
    for i, path in enumerate(images, 1):
        t = time.time()
        report = analyze.analyze_image(path, use_vlm=not args.no_vlm, use_registry=not args.no_registry)
        analyze.render_outputs(path, report, out_dir)
        export.write_json(report, out_dir / f"{path.stem}.json")
        reports.append(report)
        main = next((d for d in report["detections"] if d["category"] == "vehicle" and d["on_scale"]), None)
        plate = (main or {}).get("license_plate") or {}
        print(f"[{i}/{len(images)}] {path.name}: "
              f"{(main or {}).get('manufacturer') or '?'} {(main or {}).get('equipment_type') or '-'} | "
              f"номер: {plate.get('formatted') or '—'} | {report.get('timestamp_on_frame') or 'время ?'} "
              f"| {time.time() - t:.1f} c")

    export.write_json({"images": reports}, out_dir / "detections.json")
    export.write_csv(reports, out_dir / "detections.csv")
    print(f"\nГотово: {len(reports)} кадров за {time.time() - t0:.0f} c -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
