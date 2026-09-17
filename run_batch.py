"""Пакетная обработка папки с кадрами (обязательная часть задания 2).

Рассчитан на полный датасет организаторов (22 тыс. кадров, несколько часов):
возобновляется после обрыва, один битый файл прогон не роняет, память не растёт.

Пример:
    python run_batch.py --input /data/Auto --output results/batch
    python run_batch.py --input /data --recursive --device cuda

На выходе:
    <output>/<кадр>.json          — отчёт по кадру, он же отметка «этот кадр посчитан»
    <output>/<кадр>_annotated.jpg — кадр с рамками
    <output>/<кадр>_plate.jpg     — увеличенный номер (если найден)
    <output>/detections.json      — все кадры одним файлом (собирается из отчётов)
    <output>/detections.csv       — то же в таблице
    <output>/errors.jsonl         — журнал пропущенных кадров
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import config  # noqa: E402
from backend.pipeline import analyze, detector, export  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(in_dir: Path, recursive: bool) -> list[Path]:
    paths = in_dir.rglob("*") if recursive else in_dir.iterdir()
    return sorted([p for p in paths if p.suffix.lower() in IMAGE_EXT and p.is_file()],
                  key=lambda p: (p.parent.as_posix(), len(p.stem), p.stem))


def out_stem(path: Path, in_dir: Path) -> str:
    """Имя отчёта. При обходе подпапок путь уходит в имя, иначе Auto/1.jpg затрёт Сорняки/1.jpg."""
    try:
        rel = path.relative_to(in_dir)
    except ValueError:
        return path.stem
    return "__".join(rel.parts[:-1] + (rel.stem,))


def fmt_dur(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def log_error(out_dir: Path, path: Path, exc: BaseException) -> None:
    record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "image": str(path),
              "error": f"{type(exc).__name__}: {exc}"}
    with (out_dir / "errors.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_reports(out_dir: Path, stems: list[str], stats: dict) -> Iterator[dict]:
    """Готовые отчёты по одному с диска: итоговые файлы собираются без загрузки всех кадров в память."""
    for stem in stems:
        path = out_dir / f"{stem}.json"
        if not path.exists():
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            # Отодвигаем нечитаемый отчёт, чтобы кадр пересчитался при следующем запуске без --force.
            stats["broken"] += 1
            path.replace(path.with_name(path.name + ".broken"))
            print(f"  отчёт {path.name} повреждён ({type(e).__name__}: {e}) — отложен в {path.name}.broken")
            continue
        stats["merged"] += 1
        yield report


def summary_line(report: dict) -> str:
    main = next((d for d in report["detections"] if d["category"] == "vehicle" and d["on_scale"]), None)
    plate = (main or {}).get("license_plate") or {}
    return (f"{(main or {}).get('manufacturer') or '?'} {(main or {}).get('equipment_type') or '-'} | "
            f"номер: {plate.get('formatted') or '—'} | {report.get('timestamp_on_frame') or 'время ?'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(config.DATA_DIR), help="папка с кадрами")
    ap.add_argument("--output", default=str(config.RESULTS_DIR / "batch"), help="папка для результатов")
    ap.add_argument("--limit", type=int, default=0, help="обработать только первые N кадров")
    ap.add_argument("--recursive", action="store_true", help="обойти и подпапки (Auto, Сорняки, ФотоПолей)")
    ap.add_argument("--force", action="store_true", help="пересчитать кадры, по которым отчёт уже есть")
    ap.add_argument("--merge-only", action="store_true", help="только собрать detections.json/csv из готовых отчётов")
    ap.add_argument("--device", default=None, help="устройство для детектора: auto (по умолчанию), cuda, mps, cpu")
    ap.add_argument("--no-vlm", action="store_true", help="не вызывать VLM (марка/модель только по надписям)")
    ap.add_argument("--no-registry", action="store_true", help="не искать номер в открытых данных data.egov.kz")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    if not in_dir.exists():
        print(f"Папка с кадрами не найдена: {in_dir}\nУкажите её через --input или переменную DATA_DIR.")
        return 1
    if not in_dir.is_dir():
        print(f"Это файл, а не папка с кадрами: {in_dir}")
        return 1

    images = find_images(in_dir, args.recursive)
    if args.limit:
        images = images[: args.limit]
    if not images:
        hint = "" if args.recursive else " Если кадры разложены по подпапкам, добавьте --recursive."
        print(f"В папке {in_dir} нет изображений ({', '.join(sorted(IMAGE_EXT))}).{hint}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    stems = [out_stem(p, in_dir) for p in images]
    todo = [(p, s) for p, s in zip(images, stems)
            if args.force or not (out_dir / f"{s}.json").exists()]
    done_before = len(images) - len(todo)

    processed, failed = 0, 0
    interrupted = False
    if args.merge_only:
        print(f"Только сборка итоговых файлов: кадров {len(images)}, готовых отчётов {done_before}")
    else:
        device = detector.set_device(args.device)
        print(f"Кадров: {len(images)} | уже посчитано: {done_before} | к обработке: {len(todo)} | детектор: {device}")
        t0 = time.time()
        for i, (path, stem) in enumerate(todo, 1):
            t = time.time()
            try:
                report = analyze.analyze_image(path, use_vlm=not args.no_vlm, use_registry=not args.no_registry)
                analyze.render_outputs(path, report, out_dir)
                export.write_json(report, out_dir / f"{stem}.json")
            except KeyboardInterrupt:
                interrupted = True
                print("\nПрерывание. Собираю итоговые файлы из уже посчитанного — прогон можно продолжить тем же запуском.")
                break
            except Exception as e:
                # Битый или нечитаемый кадр не должен ронять прогон на 22 тыс. файлов.
                failed += 1
                log_error(out_dir, path, e)
                print(f"[{i}/{len(todo)}] {path.name}: ПРОПУЩЕН ({type(e).__name__}: {e})")
                continue
            processed += 1
            spent = time.time() - t0
            left = (len(todo) - i) * spent / i
            print(f"[{i}/{len(todo)}] {path.name}: {summary_line(report)} "
                  f"| {time.time() - t:.1f} c | прошло {fmt_dur(spent)}, осталось ~{fmt_dur(left)}")

    stats = {"merged": 0, "broken": 0}
    export.write_json_stream(iter_reports(out_dir, stems, stats), out_dir / "detections.json")
    # Второй проход по тем же отчётам: счётчики берём из первого, чтобы не удваивать.
    export.write_csv(iter_reports(out_dir, stems, {"merged": 0, "broken": 0}), out_dir / "detections.csv")

    print(f"\nВ итоговых файлах: {stats['merged']} кадров -> {out_dir}")
    if not args.merge_only:
        tail = f" (журнал: {out_dir / 'errors.jsonl'})" if failed else ""
        print(f"Посчитано за этот запуск: {processed}, пропущено из-за ошибок: {failed}{tail}")
    if stats["broken"]:
        print(f"Повреждённых отчётов: {stats['broken']} — запустите ещё раз, эти кадры пересчитаются")
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
