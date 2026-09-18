"""Сквозная обработка снимка с весовой: анализ -> файлы -> машина -> рейс -> ответ в чат."""
from __future__ import annotations

import math
import random
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from backend import config, db, scale, warehouses, waybill
from backend.pipeline import analyze, annotate, export, fingerprint, registry, vlm

CAPTURES_DIR = config.RESULTS_DIR / "captures"
REPORTS_DIR = config.RESULTS_DIR / "reports"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Минимальное сходство гистограмм, чтобы предложить «похожую» машину.
APPEARANCE_MATCH = 0.85
# На сколько лучший кандидат должен опережать второго. Значение выбрано как защита,
# а не подобрано по данным: для калибровки нужны повторные приезды одних и тех же машин.
APPEARANCE_MARGIN = 0.05
# Быстрый оборот подозрителен: за это время разгрузиться нельзя.
MIN_TURNAROUND_MIN = 3
# Рейс, висящий дольше смены, почти всегда означает пропущенный выезд или то,
# что машину при следующем приезде не узнали и завели заново.
MAX_OPEN_HOURS = 12

# Откуда взялся вес. Весы на COM-порту имитируются (backend/scale.py) — так и подписываем, чтобы
# имитацию не приняли за настоящее взвешивание. «Вручную» — вес, переданный в запросе API.
WEIGHT_MANUAL = "введён вручную"
WEIGHT_EDITED = "введён вручную при исправлении"
# Больше этого не весит даже автопоезд с прицепом: скорее всего перепутаны единицы.
MAX_WEIGHT_T = 200.0

# Откуда взято время события. Времена из разных источников сравнивать нельзя.
TIME_FRAME = "кадр"
TIME_SYSTEM = "система"


def _minutes_between(earlier: str | None, later: str | None) -> float | None:
    """Минуты между двумя отметками одного источника. None — если время не разобрать."""
    try:
        t1 = datetime.strptime(earlier, "%Y-%m-%d %H:%M:%S")
        t2 = datetime.strptime(later, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None
    return (t2 - t1).total_seconds() / 60


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def url_for(path: str | Path) -> str:
    rel = Path(path).resolve().relative_to(config.RESULTS_DIR.resolve())
    return "/results/" + rel.as_posix()


# ---------- камера (папка с кадрами) ----------

def list_frames() -> list[str]:
    if not config.DATA_DIR.exists():
        return []
    return sorted([p.name for p in config.DATA_DIR.iterdir() if p.suffix.lower() in IMAGE_EXT],
                  key=lambda n: (len(Path(n).stem), n))


def used_frames() -> list[str]:
    return db.guard_frames()


_last_random_frame: str | None = None


def capture_frame(frame: str | None = None, random_pick: bool = False) -> dict:
    """«Снимок с камеры»: берёт кадр из папки (следующий неиспользованный, указанный или случайный).

    random_pick — имитация камеры весовой: машина заехала, камера сама сделала снимок. Весовщик не
    выбирает кадр, как и в жизни; один и тот же кадр два раза подряд не выпадает.
    """
    global _last_random_frame
    frames = list_frames()
    if not frames:
        raise FileNotFoundError(f"В папке камеры нет кадров: {config.DATA_DIR}")
    if random_pick and frame is None:
        frame = random.choice([f for f in frames if f != _last_random_frame] or frames)
        _last_random_frame = frame
    elif frame is None:
        used = set(used_frames())
        frame = next((f for f in frames if f not in used), frames[0])
    if frame not in frames:
        raise FileNotFoundError(f"Кадр {frame} не найден")
    return save_capture(config.DATA_DIR / frame, frame)


def save_capture(src: Path, frame_name: str | None = None) -> dict:
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    capture_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}_{Path(src).stem}"
    dst = CAPTURES_DIR / f"{capture_id}{Path(src).suffix.lower()}"
    shutil.copyfile(src, dst)
    return {"capture_id": capture_id, "file": dst.name, "url": url_for(dst), "frame": frame_name}


def capture_path(capture_id: str) -> Path:
    for p in CAPTURES_DIR.glob(f"{capture_id}.*"):
        return p
    raise FileNotFoundError(f"Снимок {capture_id} не найден")


# ---------- обработка ----------

def _main_vehicle(report: dict) -> dict | None:
    main_id = report["summary"].get("main_vehicle_id")
    return next((d for d in report["detections"] if d["id"] == main_id), None)


def _find_by_appearance(fp: list[float], exclude_id: str | None = None,
                        predicate=None) -> tuple[dict | None, float]:
    """Самая похожая машина по отпечатку, но только если она заметно похожее остальных.

    Отпечаток — это цветовая гистограмма вырезки вместе с куском фона, поэтому две белые
    фуры на одной площадке дают близкие значения. Если второй кандидат почти так же похож,
    отличить их нельзя, и честнее не узнать машину, чем приписать рейс чужой.
    """
    best, best_sim, second_sim = None, 0.0, 0.0
    for v in db.list_vehicles():
        if v["id"] == exclude_id or not v.get("fingerprint"):
            continue
        if predicate is not None and not predicate(v):
            continue
        sim = fingerprint.similarity(fp, v["fingerprint"])
        if sim > best_sim:
            best, best_sim, second_sim = v, sim, best_sim
        elif sim > second_sim:
            second_sim = sim
    if best is not None and best_sim - second_sim < APPEARANCE_MARGIN:
        return None, best_sim
    return best, best_sim


def _fmt_t(value: float | None) -> str:
    return f"{value:.2f} т" if value is not None else "—"


def process_capture(capture_id: str, guard: dict) -> dict:
    """guard: driver, crop, warehouse_id, weight, note, plate_override, frame."""
    image = capture_path(capture_id)
    wh = warehouses.get(guard.get("warehouse_id"))
    created = now_iso()

    # Повторная отправка того же снимка (оборвалась связь, нажали дважды) не должна открывать второй рейс.
    prev = db.event_for_capture(capture_id)
    if prev and prev["status"] == "active":
        return {"guard_message": db.get_message(prev["guard_message_id"]),
                "ai_message": db.get_message(prev["ai_message_id"]), "duplicate": True}

    guard_text = ", ".join(x for x in [
        f"водитель: {guard['driver']}" if guard.get("driver") else "",
        f"культура: {guard['crop']}" if guard.get("crop") else "",
        f"вес: {guard['weight']} т" if not _weight_left_empty(guard.get("weight")) else "",
        f"номер: {guard['plate_override']}" if guard.get("plate_override") else "",
        guard.get("note") or "",
    ] if x) or "Машина на весах."
    if prev:
        # Прошлая попытка сорвалась на анализе: дописываем то же сообщение, а не плодим второе.
        guard_msg_id, event_id = prev["guard_message_id"], prev["id"]
        db.update_message(guard_msg_id, text=guard_text, payload={**guard, "capture_id": capture_id})
    else:
        guard_msg_id = db.add_message({
            "created_at": created, "role": "guard", "text": guard_text, "image": url_for(image),
            "payload": {**guard, "capture_id": capture_id}, "warehouse_id": wh["id"],
        })
        event_id = db.create_event({"created_at": created, "capture_id": capture_id,
                                    "guard_message_id": guard_msg_id, "status": "pending"})

    try:
        return _analyze_and_reply(capture_id, guard, image, wh, created, guard_msg_id, event_id)
    except Exception as e:
        # Сообщение охранника уже в чате — отвечаем об ошибке, чтобы оно не осталось без ответа.
        db.add_message({
            "created_at": now_iso(), "role": "ai",
            "text": f"Не удалось разобрать снимок: {e}. Снимок остался прикреплён — отправьте его ещё раз.",
            "payload": {"alerts": [], "error": str(e)}, "warehouse_id": wh["id"], "event_kind": "none",
        })
        raise


def _analyze_and_reply(capture_id: str, guard: dict, image: Path, wh: dict, created: str,
                       guard_msg_id: int, event_id: int) -> dict:
    # 1. Анализ кадра. Реестр здесь не вызываем: номер может исправить охранник, ищем один раз ниже.
    report = analyze.analyze_image(image, use_vlm=True, use_registry=False)
    out_dir = REPORTS_DIR / capture_id
    report["guard_input"] = {"driver_surname": guard.get("driver"), "crop": guard.get("crop"),
                             "warehouse": wh["name"], "note": guard.get("note")}
    # В отчёте — имя исходного кадра камеры, а не служебной копии снимка.
    report["source_image"] = guard.get("frame") or report["source_image"]
    report["capture_file"] = image.name
    files: dict[str, str] = {}

    def write_images() -> None:
        """Кадр с рамками пишем после всех уточнений (правка номера, реестр),
        чтобы подписи на картинке совпадали с ответом в чате."""
        files.update({k: url_for(v) for k, v in analyze.render_outputs(image, report, out_dir).items()})

    def write_data() -> None:
        """JSON и CSV — последними: в них должен попасть и вес с пометкой, откуда он взялся."""
        files["json"] = url_for(export.write_json(report, out_dir / "detections.json"))
        files["csv"] = url_for(export.write_csv([report], out_dir / "detections.csv"))

    main = _main_vehicle(report)
    frame_time = report.get("timestamp_on_frame")
    event_time = frame_time or created
    time_source = TIME_FRAME if frame_time else TIME_SYSTEM
    time_note = "время с кадра камеры" if frame_time else "время кадра не прочитано, взято системное"

    if main is None:
        write_images()
        write_data()
        text = ("На снимке не найдена техника на весовой платформе. "
                f"Найдено объектов: {report['summary']['total_objects_detected']}. Рейс не создан.")
        ai_id = db.add_message({"created_at": now_iso(), "role": "ai", "text": text, "image": files.get("annotated"),
                                "files": files, "payload": {"report_summary": report["summary"], "alerts": []},
                                "warehouse_id": wh["id"], "event_kind": "none"})
        db.update_event(event_id, kind="none", ai_message_id=ai_id, status="active", snapshot={})
        return {"guard_message": db.get_message(guard_msg_id), "ai_message": db.get_message(ai_id)}

    # 2. Номер: правка охранника важнее прочтения OCR.
    plate_info = main.get("license_plate") or {}
    plate_compact, plate_formatted, plate_source = None, None, None
    if guard.get("plate_override"):
        from backend.pipeline import plates as plates_mod
        m = plates_mod.match_plate(guard["plate_override"])
        if m:
            plate_compact, plate_formatted, plate_source = m[1], m[2], "введён охранником"
        else:
            plate_compact = plates_mod.normalize(guard["plate_override"])
            plate_formatted, plate_source = guard["plate_override"].upper(), "введён охранником (нестандартный формат)"
        # В отчёте (JSON/CSV) тоже должен быть номер, подтверждённый охранником, а не прочтение OCR.
        main["license_plate"] = {
            "text": plate_compact, "formatted": plate_formatted, "country": "Казахстан (KZ)",
            "region_code": m[3] if m else None, "format": m[0] if m else "custom",
            "readable": True, "confidence": "подтверждён охранником", "source": plate_source,
            "ocr_reading": {k: plate_info.get(k) for k in ("text", "formatted", "confidence", "readable")}
            if plate_info.get("text") else None,
        }
        # Строка номера в отчёте (рамка на кадре) тоже получает исправленный номер.
        for d in report["detections"]:
            if d["category"] == "license_plate" and d.get("vehicle_id") == main["id"]:
                old = (d.get("license_plate") or {}).get("formatted")
                d["license_plate"] = {**(d.get("license_plate") or {}), "text": plate_compact,
                                      "formatted": plate_formatted, "format": m[0] if m else "custom",
                                      "region_code": m[3] if m else None}
                d["readable"] = True
                d["notes"] = f"исправлен охранником, OCR: {old or '—'}"
    elif plate_info.get("text") and plate_info.get("readable"):
        plate_compact, plate_formatted, plate_source = plate_info["text"], plate_info["formatted"], "распознан по кадру"

    # 3. Отпечаток внешности и узнавание без номера.
    img = annotate.imread(image)
    bb = main["bounding_box_px"]
    crop = img[bb["y1"]:bb["y2"], bb["x1"]:bb["x2"]]
    fp = fingerprint.compute(crop)
    appearance = main.get("appearance") or {}
    recognized_by_appearance = None
    cand = None
    if not plate_compact:
        cand, sim = _find_by_appearance(fp)
        if cand and sim >= APPEARANCE_MATCH:
            recognized_by_appearance = {"vehicle_id": cand["id"], "similarity": round(sim, 2),
                                        "plate": cand.get("plate_formatted")}
            plate_compact, plate_formatted = cand.get("plate"), cand.get("plate_formatted")
            plate_source = f"узнана по внешности (сходство {sim:.0%}), номер из профиля"
            if cand.get("manufacturer") and not main.get("manufacturer"):
                main["manufacturer"] = cand["manufacturer"]
                analyze.set_country(main)
            if plate_compact:
                # На кадре номер не прочитан: не выдаём прочтение OCR за номер из профиля.
                main["license_plate"] = {
                    "text": plate_compact, "formatted": plate_formatted, "country": "Казахстан (KZ)",
                    "region_code": None, "format": None, "readable": False,
                    "confidence": "из профиля машины (на кадре не прочитан)", "source": plate_source,
                    "ocr_reading": {k: plate_info.get(k) for k in ("text", "formatted", "confidence", "readable")}
                    if plate_info.get("text") else None,
                }
        else:
            cand = None

    # Бортовой номер «машины без номера»: если номер прочитан впервые, а на территории стоит похожий
    # профиль без номера с открытым рейсом — это она же, номер просто не читался на заезде.
    plateless_match = None
    if plate_compact and not recognized_by_appearance and db.get_vehicle_by_plate(plate_compact) is None:
        pl_cand, pl_sim = _find_by_appearance(fp, predicate=lambda v: not v.get("plate") and db.open_trip_for(v["id"]))
        if pl_cand and pl_sim >= APPEARANCE_MATCH:
            plateless_match = pl_cand

    # Реестр data.egov.kz: один запрос по окончательному номеру (OCR, правка охранника или профиль).
    registry_status, registry_record = registry.DISABLED, None
    if plate_compact:
        registry_status, registry_record = registry.lookup_status(plate_compact)
        main["registry_status"] = registry_status
        analyze.apply_registry(main, registry_record)
    write_images()

    # Бортовой номер: у известной машины (по номеру или по внешности) свой, новой — следующий по порядку.
    existing = cand if (recognized_by_appearance and cand) else (
        db.get_vehicle_by_plate(plate_compact) or plateless_match)
    vehicle_id = existing["id"] if existing else db.next_board_no()
    temporary = not plate_compact

    # Снимок профиля ДО записи: если событие придётся отменить, машина вернётся к нему.
    vehicle_before = db.raw_vehicle(vehicle_id)

    db.upsert_vehicle({
        "id": vehicle_id, "plate": plate_compact, "plate_formatted": plate_formatted,
        "manufacturer": main.get("manufacturer"), "model": main.get("model"),
        "equipment_type": main.get("equipment_type"), "year": main.get("year"), "color": appearance.get("color"),
        "appearance": appearance or None,
        # Узнали по внешности — эталонный отпечаток не трогаем: иначе профиль «отравляется»
        # чужим кадром и следующие сопоставления идут уже от испорченного образца.
        "fingerprint": None if recognized_by_appearance else fp,
        "photo": files.get("vehicle_crop"),
        "is_temporary": temporary, "seen_at": event_time,
    })

    # 4. Рейс: есть открытый -> это выезд, иначе заезд.
    alerts: list[str] = []
    if not plate_compact:
        alerts.append("Номер не прочитан — подтвердите или введите вручную")
        if plate_info.get("text"):
            alerts[-1] += f" (предположительно {plate_info['formatted']})"
    elif recognized_by_appearance:
        alerts.append(f"Номер не прочитан; машина сопоставлена по внешности с профилем "
                      f"№{recognized_by_appearance['vehicle_id']} — подтвердите номер")
    заданный_склад = guard.get("warehouse_id")
    if not заданный_склад:
        alerts.append(f"Склад не выбран — записан {wh['name']}")
    elif заданный_склад not in warehouses.BY_ID:
        alerts.append(f"Склад «{заданный_склад}» неизвестен — записан {wh['name']}")
    load_state = appearance.get("load_state")
    has_trailer = appearance.get("has_trailer")
    manual_weight = None
    if not _weight_left_empty(guard.get("weight")):
        try:
            manual_weight = _parse_weight(guard["weight"])
        except ValueError as e:
            alerts.append(f"{e} Взят вес с весов.")

    weight_source = WEIGHT_MANUAL if manual_weight is not None else scale.source()

    open_trip = db.open_trip_for(vehicle_id)
    trip_before = db.raw_trip(open_trip["id"]) if open_trip else None
    # Путевой лист (имитация): на заезде выписан диспетчером, на выезде тот же, что в рейсе.
    # Водителя и груз весовщик больше не вводит — они из листа; поля guard остаются для API и тестов.
    wb = (open_trip or {}).get("waybill") or waybill.issue(
        vehicle_id, {**main, "plate_formatted": plate_formatted}, event_time,
        visit_no=len(db.list_trips(vehicle_id=vehicle_id)) + 1, has_trailer=has_trailer)
    driver = guard.get("driver") or wb["driver"]["name"]
    crop = guard.get("crop") or wb["task"]["cargo"]
    if open_trip:
        event_kind = "exit"
        weight = manual_weight if manual_weight is not None else scale.read(
            "exit", vehicle_id, main.get("manufacturer"), main.get("equipment_type"), load_state)
        entry_w = open_trip.get("entry_weight")
        net = round(entry_w - weight, 2) if entry_w is not None else None
        if entry_w is None:
            alerts.append("Брутто на заезде неизвестно — нетто не посчитано")
        elif weight >= entry_w:
            alerts.append(f"Выезд тяжелее заезда ({_fmt_t(weight)} ≥ {_fmt_t(entry_w)}) — груз не разгружен?")
        if load_state == "гружёный":
            alerts.append("На выезде кузов выглядит гружёным — разгрузка не подтверждена")
        if open_trip.get("entry_has_trailer") and has_trailer is False:
            alerts.append("На заезде был прицеп, на выезде прицепа нет")
        if open_trip["warehouse_id"] != wh["id"]:
            alerts.append(f"Заезд был через {warehouses.get(open_trip['warehouse_id'])['name']}, выезд через {wh['name']}")
        # Времена из разных источников сравнивать нельзя: на части кадров надпись камеры
        # показывает 2019 год, и смесь «заезд с кадра, выезд системный» давала тревогу
        # почти на каждом выезде. Оба с кадров — считаем по ним, иначе по системным часам.
        same_source = open_trip.get("entry_time_source") == time_source
        both_from_frame = same_source and time_source == TIME_FRAME
        if same_source:
            minutes = _minutes_between(open_trip.get("entry_time"), event_time)
        else:
            # Одно время с надписи камеры, другое системное: разница между ними бессмысленна,
            # поэтому оборот не проверяем вовсе, а не выдумываем тревогу.
            minutes = None
        if minutes is not None and 0 <= minutes < MIN_TURNAROUND_MIN:
            alerts.append(f"Слишком быстрый оборот: {minutes:.0f} мин между заездом и выездом")
        if minutes is not None and minutes > MAX_OPEN_HOURS * 60:
            alerts.append(f"Рейс был открыт {minutes / 60:.0f} ч — проверьте, не пропущен ли выезд")
        # Сбитые часы камеры видно, только если оба времени сняты с кадров.
        if both_from_frame and minutes is not None and minutes < 0:
            alerts.append("Время выезда на кадре раньше времени заезда — проверьте часы камеры")
        db.close_trip(open_trip["id"], {
            "exit_time": event_time, "exit_weight": weight, "exit_message_id": guard_msg_id,
            "exit_load_state": load_state, "exit_has_trailer": has_trailer, "net_weight": net,
            "exit_time_source": time_source,
            "alerts": (open_trip.get("alerts") or []) + alerts, "exit_weight_source": weight_source,
            "driver": driver, "crop": crop,
        })
        trip_id = open_trip["id"]
        summary = (f"Выезд. Тара {_fmt_t(weight)} ({weight_source}), брутто на заезде {_fmt_t(entry_w)}, "
                   f"нетто {_fmt_t(net)}. Рейс №{trip_id} закрыт.")
    else:
        event_kind = "entry"
        weight = manual_weight if manual_weight is not None else scale.read(
            "entry", vehicle_id, main.get("manufacturer"), main.get("equipment_type"), load_state)
        if load_state == "пустой":
            alerts.append("Машина заезжает пустой — возможно, это отгрузка, а не приёмка")
        trip_id = db.create_trip({
            "vehicle_id": vehicle_id, "warehouse_id": wh["id"], "driver": driver,
            "crop": crop, "entry_time": event_time, "entry_weight": weight, "waybill": wb,
            "entry_message_id": guard_msg_id, "entry_load_state": load_state, "entry_has_trailer": has_trailer,
            "alerts": alerts, "entry_weight_source": weight_source, "entry_time_source": time_source,
        })
        summary = (f"Заезд. Брутто {_fmt_t(weight)} ({weight_source}). "
                   f"Открыт рейс №{trip_id}; ждём выезда пустой машины.")

    # Вес попадает и в выгрузку: в JSON отдельным блоком, в CSV — примечанием к машине.
    report["weighing"] = {"event": event_kind, "trip_id": trip_id, "vehicle_id": vehicle_id,
                          "weight_t": weight, "weight_source": weight_source,
                          "measured": manual_weight is not None}
    main["notes"] = (main.get("notes") or "") + f"Вес {weight:.2f} т — {weight_source}. "
    write_data()

    vehicle = db.get_vehicle(vehicle_id)
    trip = db.get_trip(trip_id)

    # 5. Ответ в чат.
    model_txt = main.get("model") if main.get("model") not in (None, "", "не определена") else None
    lines = [
        f"**{vehicle['label']}** — {main.get('equipment_type') or 'техника'}; "
        f"производитель: {main.get('manufacturer') or 'не определён'}"
        + (f" ({main['manufacturer_country']})" if main.get("manufacturer_country") else "")
        + f"; модель: {model_txt or 'не определена'}"
        + (f"; год выпуска: {main['year']}" if main.get("year") else "") + ".",
        f"Номер: {plate_formatted or 'не прочитан'}"
        + (f" ({plate_source})" if plate_source else "")
        + (f"; OCR предполагает {plate_info['formatted']}" if plate_info.get("text") and not plate_info.get("readable")
           and not guard.get("plate_override") else "") + ".",
    ]
    if registry_status == registry.FOUND and registry_record:
        lines.append(f"Реестр data.egov.kz: {registry_record.get('marka_raw') or ''} "
                     f"{registry_record.get('year') or ''} г., тип: {registry_record.get('type') or '—'}.")
    elif registry_status == registry.NOT_FOUND:
        lines.append("В открытых данных data.egov.kz номер не найден (набор неполный) — марка/модель по фото и надписям.")
    elif registry_status == registry.UNAVAILABLE:
        lines.append("Реестр data.egov.kz недоступен (нет сети или таймаут) — номер не проверялся, марка/модель по фото и надписям.")
    # Сырые вероятности — в JSON (блок fallback); в чате — словами: ответ выдаётся, только если он выше
    # порога, при котором на проверке модель права в ≥90% случаев. «35%» среди 12 моделей в чате пугало бы зря.
    make_by_model = analyze.FALLBACK_SOURCE in (main.get("manufacturer_confidence") or "")
    model_by_model = analyze.FALLBACK_SOURCE in (main.get("model_confidence") or "")
    if make_by_model or model_by_model:
        what = "Марку и модель" if make_by_model and model_by_model else "Марку" if make_by_model else "Модель"
        names = ", ".join(main[f] for f, used in (("manufacturer", make_by_model), ("model", model_by_model)) if used)
        lines.append(f"{what} ({names}) определила по виду машины обученная модель.")
    if appearance.get("description"):
        lines.append(f"Внешность: {appearance['description']}")
    if recognized_by_appearance:
        lines.append(f"Машина узнана по внешности: профиль №{recognized_by_appearance['vehicle_id']} "
                     f"(сходство {recognized_by_appearance['similarity']:.0%}).")
    if plateless_match:
        lines.append(f"Профиль №{plateless_match['id']} (без номера) получил номер {plate_formatted}.")
    lines.append(summary)
    lines.append(f"Склад: {wh['name']}. Время: {event_time} ({time_note}).")
    lines.append(f"Путевой лист №{wb['number']}: водитель {driver}, груз {crop}, {wb['task']['from']}.")
    lines.extend(f"⚠ {a}" for a in alerts)   # каждое предупреждение — своей строкой
    objects = report["summary"]
    # Ссылки на файлы (кадр, JSON, CSV) интерфейс показывает строкой под ответом — в тексте не повторяем
    lines.append(f"На кадре: техники {objects['vehicles_or_equipment']}, людей {objects['people']}, "
                 f"номеров прочитано {objects['plates_read']}.")

    payload = {
        "vehicle": vehicle, "trip": trip, "event_kind": event_kind, "weight": weight, "alerts": alerts,
        "weight_source": weight_source, "weight_measured": manual_weight is not None,
        "plate": {"formatted": plate_formatted, "source": plate_source, "ocr": plate_info},
        "registry": {"status": registry_status, "record": registry_record},
        "recognized_by_appearance": recognized_by_appearance,
        "report_summary": objects, "processing_seconds": report.get("processing_seconds"),
        "vlm_used": bool(appearance.get("source")),
        # Для таблицы в чате и у руководителя: то, чего нет в профиле машины и рейсе.
        # Номер, вес, водителя и культуру таблица берёт из vehicle/trip — их обновляет правка события.
        "details": {
            "equipment_type": main.get("equipment_type"), "manufacturer": main.get("manufacturer"),
            "country": main.get("manufacturer_country"), "model": model_txt, "year": main.get("year"),
            "identified_by": _identified_by(main, registry_status, registry_record, bool(appearance.get("source"))),
            "appearance": appearance.get("description"),
            "registry": _registry_text(registry_status, registry_record),
            "plate_guess": plate_info.get("formatted") if plate_info.get("text") and not plate_info.get("readable")
            and not guard.get("plate_override") else None,
            "recognized": (f"узнана по внешности: профиль №{recognized_by_appearance['vehicle_id']} "
                           f"(сходство {recognized_by_appearance['similarity']:.0%})") if recognized_by_appearance else None,
            "plateless_match": f"профиль №{plateless_match['id']} без номера получил этот номер" if plateless_match else None,
            "event_time": event_time, "time_note": time_note, "warehouse": wh["name"],
            "waybill": wb, "driver_source": "весовщик" if guard.get("driver") else f"путевой лист №{wb['number']}",
            "note": guard.get("note") or None,
        },
    }
    ai_id = db.add_message({
        "created_at": now_iso(), "role": "ai", "text": "\n".join(lines), "image": files.get("annotated"),
        "files": files, "payload": payload, "vehicle_id": vehicle_id, "warehouse_id": wh["id"],
        "trip_id": trip_id, "event_kind": event_kind,
    })
    db.update_message(guard_msg_id, vehicle_id=vehicle_id, trip_id=trip_id, event_kind=event_kind)
    db.update_event(event_id, kind=event_kind, vehicle_id=vehicle_id, trip_id=trip_id, ai_message_id=ai_id,
                    status="active", snapshot={"vehicle": vehicle_before, "trip": trip_before})
    return {"guard_message": db.get_message(guard_msg_id), "ai_message": db.get_message(ai_id),
            "event_id": event_id}


# ---------- исправление ошибки весовщицы ----------

def _registry_text(status: str, record: dict | None) -> str | None:
    """Строка «Реестр» для таблицы ответа — те же слова, что и в тексте."""
    if status == registry.FOUND and record:
        return (f"{record.get('marka_raw') or ''} {record.get('year') or ''} г., "
                f"тип: {record.get('type') or '—'}").strip()
    if status == registry.NOT_FOUND:
        return "номер не найден (открытый набор неполный)"
    if status == registry.UNAVAILABLE:
        return "недоступен (нет сети или таймаут) — номер не проверялся"
    return None


def _identified_by(main: dict, registry_status: str, registry_record: dict | None, vlm_used: bool) -> str | None:
    """Откуда марка и модель: реестр по номеру, надпись на технике, обученная модель или VLM."""
    def source(conf: str | None, value) -> str | None:
        conf = conf or ""
        if not value:
            return None
        if "реестр" in conf:
            return "реестр data.egov.kz по номеру"
        if analyze.FALLBACK_SOURCE in conf:
            return analyze.FALLBACK_SOURCE
        if "надпись" in conf:
            return "надпись на технике"
        return "VLM по фото" if vlm_used else None

    model = main.get("model") if main.get("model") not in (None, "", "не определена") else None
    make_src = source(main.get("manufacturer_confidence"), main.get("manufacturer"))
    model_src = source(main.get("model_confidence"), model)
    if make_src and model_src and make_src != model_src:
        return f"марка — {make_src}; модель — {model_src}"
    return make_src or model_src


def _kind_ru(kind: str) -> str:
    return "заезд" if kind == "entry" else "выезд"


def last_event() -> dict | None:
    """Последнее оформленное событие с данными для формы правки."""
    ev = db.last_active_event()
    if not ev:
        return None
    trip = db.get_trip(ev["trip_id"]) if ev.get("trip_id") else None
    vehicle = db.get_vehicle(ev["vehicle_id"]) if ev.get("vehicle_id") else None
    weight = (trip or {}).get("entry_weight" if ev["kind"] == "entry" else "exit_weight")
    source = (trip or {}).get("entry_weight_source" if ev["kind"] == "entry" else "exit_weight_source")
    return {"event": ev, "trip": trip, "vehicle": vehicle, "kind_ru": _kind_ru(ev["kind"]),
            "weight": weight, "weight_source": source, "ai_message_id": ev.get("ai_message_id")}


def _mark_message(message_id: int | None, extra: dict) -> None:
    """Помечаем сообщение в чате, но не стираем: в базе должно остаться, что событие было."""
    if not message_id:
        return
    m = db.get_message(message_id)
    if m:
        db.update_message(message_id, payload={**(m.get("payload") or {}), **extra})


def undo_last_event() -> dict:
    """Откат последнего заезда/выезда: рейс и профиль машины возвращаются в прежнее состояние."""
    ev = db.last_active_event()
    if not ev:
        raise LookupError("Отменять нечего: ни одного заезда или выезда ещё не оформлено.")
    trip = db.get_trip(ev["trip_id"]) if ev.get("trip_id") else None
    vehicle = db.get_vehicle(ev["vehicle_id"]) if ev.get("vehicle_id") else None
    label = (vehicle or {}).get("label") or f"№{ev.get('vehicle_id')}"
    snapshot = ev.get("snapshot") or {}
    vehicle_removed = ev.get("vehicle_id") is not None and not snapshot.get("vehicle")

    db.restore_state(snapshot, ev.get("vehicle_id"), ev.get("trip_id"))
    at = now_iso()
    db.mark_event_undone(ev["id"], at)
    _mark_message(ev.get("guard_message_id"), {"undone": True, "undone_at": at})
    _mark_message(ev.get("ai_message_id"), {"undone": True, "undone_at": at})

    if ev["kind"] == "entry":
        lines = [f"Заезд отменён. Рейс №{ev['trip_id']} удалён, машина {label} больше не числится на территории."]
        if vehicle_removed:
            lines.append(f"Профиль {label} завёлся этим заездом — он тоже убран.")
    else:
        entry_w = (snapshot.get("trip") or {}).get("entry_weight")
        lines = [f"Выезд отменён. Рейс №{ev['trip_id']} снова открыт: брутто на заезде {_fmt_t(entry_w)}, "
                 f"машина {label} снова на территории."]
    lines.append(f"Оформлено было {ev['created_at']}, отменено {at}. Запись об отмене осталась в базе. "
                 f"Снимок можно отправить заново.")
    ai_id = db.add_message({
        "created_at": at, "role": "ai", "text": "\n".join(lines),
        "payload": {"undo_of": ev["id"], "event_kind": "undo", "kind_undone": ev["kind"],
                    "trip_id": ev.get("trip_id"), "alerts": []},
        "vehicle_id": ev.get("vehicle_id"), "warehouse_id": (trip or {}).get("warehouse_id"),
        "event_kind": "undo",
    })
    return {"ok": True, "undone": ev["kind"], "trip_id": ev.get("trip_id"),
            "ai_message": db.get_message(ai_id)}


# Раньше подсказка в поле веса была «пусто = весы», и весовщица так и писала «пусто».
EMPTY_WEIGHT_WORDS = {"", "пусто", "пустой", "нет", "-", "—", "–"}


def _weight_left_empty(raw) -> bool:
    """Вес не введён: поле пустое или в нём слово вроде «пусто» — берём вес с весов, без предупреждения."""
    return raw is None or str(raw).strip().lower() in EMPTY_WEIGHT_WORDS


def _parse_weight(raw) -> float:
    """Вес из строки. float() принимает «nan» и «inf», а nan бесшумно ломает все сравнения."""
    try:
        value = float(str(raw).replace(",", ".").replace(" ", ""))
    except ValueError:
        raise ValueError(f"Вес «{raw}» не похож на число. Напишите, например, 12.5")
    if not math.isfinite(value):
        raise ValueError(f"Вес «{raw}» — не число. Напишите, например, 12.5")
    if value <= 0:
        raise ValueError("Вес должен быть больше нуля.")
    if value > MAX_WEIGHT_T:
        raise ValueError(f"Вес {value:.0f} т слишком велик — проверьте единицы, ожидаются тонны.")
    return round(value, 2)


def edit_last_event(changes: dict) -> dict:
    """Правка последнего события: номер, водитель, культура, вес. Прежние значения остаются в журнале."""
    ev = db.last_active_event()
    if not ev:
        raise LookupError("Исправлять нечего: ни одного заезда или выезда ещё не оформлено.")
    trip = db.get_trip(ev["trip_id"]) if ev.get("trip_id") else None
    vehicle = db.get_vehicle(ev["vehicle_id"]) if ev.get("vehicle_id") else None
    if trip is None:
        raise LookupError("Рейс этого события уже не найден — обновите страницу.")
    entry = ev["kind"] == "entry"
    done: list[dict] = []

    plate_raw = (changes.get("plate") or "").strip()
    if plate_raw and vehicle:
        from backend.pipeline import plates as plates_mod
        m = plates_mod.match_plate(plate_raw)
        compact = m[1] if m else plates_mod.normalize(plate_raw)
        formatted = m[2] if m else plate_raw.upper()
        if not compact:
            raise ValueError("Номер не разобрать. Введите его как на табличке, например 041 AHF 10.")
        if compact != vehicle.get("plate"):
            done.append({"field": "номер", "was": vehicle.get("plate_formatted"), "now": formatted})
            db.update_vehicle(ev["vehicle_id"], plate=compact, plate_formatted=formatted, is_temporary=0)

    fields: dict = {}
    for key, name in (("driver", "водитель"), ("crop", "культура")):
        value = (changes.get(key) or "").strip()
        if value and value != (trip.get(key) or ""):
            done.append({"field": name, "was": trip.get(key), "now": value})
            fields[key] = value

    if changes.get("weight") not in (None, ""):
        value = _parse_weight(changes["weight"])
        was = trip.get("entry_weight") if entry else trip.get("exit_weight")
        if value != was:
            done.append({"field": "вес на заезде" if entry else "вес на выезде", "was": was, "now": value})
            fields["entry_weight" if entry else "exit_weight"] = value
            fields["entry_weight_source" if entry else "exit_weight_source"] = WEIGHT_EDITED
            other = trip.get("exit_weight") if entry else trip.get("entry_weight")
            if other is not None:
                fields["net_weight"] = round((value - other) if entry else (other - value), 2)

    if not done:
        raise ValueError("Ничего не изменилось: впишите новое значение хотя бы в одно поле.")
    if fields:
        db.update_trip(trip["id"], **fields)
    at = now_iso()
    db.append_event_edit(ev["id"], {"at": at, "changes": done})

    trip = db.get_trip(trip["id"])
    vehicle = db.get_vehicle(ev["vehicle_id"]) if ev.get("vehicle_id") else None
    # Ответ ИИ в чате обновляем, иначе он продолжит показывать старый номер и вес.
    _mark_message(ev.get("ai_message_id"), {
        "edited": True, "vehicle": vehicle, "trip": trip,
        "weight": trip.get("entry_weight") if entry else trip.get("exit_weight"),
        "weight_source": trip.get("entry_weight_source") if entry else trip.get("exit_weight_source"),
        "plate": {"formatted": (vehicle or {}).get("plate_formatted"), "source": "исправлен весовщиком"},
    })
    listed = "; ".join(f"{c['field']}: {c['was'] if c['was'] not in (None, '') else '—'} → {c['now']}" for c in done)
    ai_id = db.add_message({
        "created_at": at, "role": "ai",
        "text": (f"Исправлено весовщиком ({_kind_ru(ev['kind'])}, рейс №{trip['id']}, "
                 f"{(vehicle or {}).get('label') or ''}): {listed}.\n"
                 f"Прежние значения остались в журнале правок рейса."),
        "payload": {"edit_of": ev["id"], "event_kind": "edit", "changes": done, "trip": trip,
                    "vehicle": vehicle, "alerts": []},
        "vehicle_id": ev.get("vehicle_id"), "warehouse_id": trip.get("warehouse_id"), "trip_id": trip["id"],
        "event_kind": "edit",
    })
    return {"ok": True, "changes": done, "trip": trip, "vehicle": vehicle,
            "ai_message": db.get_message(ai_id)}


# ---------- сводки для панелей ----------

# Тревоги рейсов по видам — руководителю важнее «вес введён с ошибкой ×3», чем три одинаковых строки.
ALERT_KINDS = [
    ("номер не прочитан", ("Номер не прочитан",)),
    ("вес введён с ошибкой", ("не похож на число", "— не число", "Вес должен", "слишком велик")),
    ("выезд тяжелее заезда", ("тяжелее заезда",)),
    ("груз не подтверждён", ("заезжает пустой", "выглядит гружёным")),
    ("пропал прицеп", ("прицепа нет",)),
    ("склад не выбран", ("Склад не выбран", "неизвестен — записан")),
    ("слишком быстрый оборот", ("быстрый оборот",)),
    ("рейс долго открыт", ("Рейс был открыт",)),
    ("часы камеры сбиты", ("раньше времени заезда",)),
    ("заезд и выезд через разные склады", ("выезд через",)),
    ("нетто не посчитано", ("нетто не посчитано",)),
]


def _alert_kind(text: str) -> str:
    return next((kind for kind, keys in ALERT_KINDS if any(k in text for k in keys)), "прочее")


def _crop_name(crop: str | None) -> str:
    """«пшеница» и «Пшеница» — одна культура; опечатки не исправляем, показываем как ввели."""
    crop = (crop or "").strip()
    return crop[:1].upper() + crop[1:].lower() if crop else "культура не указана"


def warehouse_summary(wh: dict) -> dict:
    trips = db.list_trips(warehouse_id=wh["id"])
    open_trips = [t for t in trips if t["status"] == "open"]
    closed = [t for t in trips if t["status"] == "closed"]
    last_entry = max(trips, key=lambda t: t["entry_time"] or "") if trips else None
    last_exit = max(closed, key=lambda t: t["exit_time"] or "") if closed else None
    # Нетто = брутто на заезде − тара на выезде. Плюс — груз оставили на складе (приёмка),
    # минус — машина уехала тяжелее, то есть груз взяли со склада (отгрузка). Складывать их нельзя:
    # раньше «принято» показывало −1 т, когда была одна отгрузка.
    nets = [t for t in closed if t.get("net_weight") is not None]
    received = round(sum(t["net_weight"] for t in nets if t["net_weight"] > 0), 2)
    shipped = round(sum(-t["net_weight"] for t in nets if t["net_weight"] < 0), 2)
    by_crop: dict[str, dict] = {}
    for t in nets:
        c = by_crop.setdefault(_crop_name(t.get("crop")), {"received_t": 0.0, "shipped_t": 0.0, "trips": 0})
        c["received_t" if t["net_weight"] > 0 else "shipped_t"] += abs(t["net_weight"])
        c["trips"] += 1
    for c in by_crop.values():
        c["received_t"], c["shipped_t"] = round(c["received_t"], 2), round(c["shipped_t"], 2)
    alert_kinds: dict[str, int] = {}
    for t in trips:
        for a in t.get("alerts") or []:
            alert_kinds[_alert_kind(a)] = alert_kinds.get(_alert_kind(a), 0) + 1
    # Время на весовой считаем, только если заезд и выезд взяты из одного источника (оба с кадра или оба
    # системные) и рейс уложился в смену: дольше — это сбитые часы камеры (2019 год на надписи) или
    # пропущенный выезд, и одно такое значение превращало среднее в «67 100 ч».
    turnaround = []
    for t in closed:
        if t.get("entry_time_source") == t.get("exit_time_source"):
            minutes = _minutes_between(t.get("entry_time"), t.get("exit_time"))
            if minutes is not None and 0 <= minutes <= MAX_OPEN_HOURS * 60:
                turnaround.append(minutes)
    now = []
    for t in sorted(open_trips, key=lambda t: t.get("entry_time") or ""):
        alerts = t.get("alerts") or []
        now.append({"vehicle": db.get_vehicle(t["vehicle_id"]), "trip_id": t["id"], "entry_time": t.get("entry_time"),
                    "entry_weight": t.get("entry_weight"), "driver": t.get("driver"), "crop": t.get("crop"),
                    "alerts": alerts,
                    "status": "есть замечания" if alerts else "на складе, ждём выезда"})
    return {
        **wh,
        "count_now": len(open_trips),
        "vehicles_now": [n["vehicle"] for n in now],
        "on_site": now,
        "last_entry": {**last_entry, "vehicle": db.get_vehicle(last_entry["vehicle_id"])} if last_entry else None,
        "last_exit": {**last_exit, "vehicle": db.get_vehicle(last_exit["vehicle_id"])} if last_exit else None,
        "received_t": received,
        "shipped_t": shipped,
        "awaiting_gross_t": round(sum(t.get("entry_weight") or 0 for t in open_trips), 2),
        "by_crop": dict(sorted(by_crop.items(), key=lambda kv: -(kv[1]["received_t"] + kv[1]["shipped_t"]))),
        "trips_open": len(open_trips), "trips_closed": len(closed),
        "alert_kinds": dict(sorted(alert_kinds.items(), key=lambda kv: -kv[1])),
        "avg_turnaround_min": round(sum(turnaround) / len(turnaround)) if turnaround else None,
        "turnaround_trips": len(turnaround),
        # load_t — принятое на склад (для полоски заполнения на карте), только положительное нетто
        "load_t": received,
        "load_pct": min(100, round(100 * received / wh["capacity_t"])) if wh.get("capacity_t") else 0,
        "trips_total": len(trips),
        "alerts": sum(len(t.get("alerts") or []) for t in trips),
    }


def state() -> dict:
    vehicles = db.list_vehicles()
    open_by_vehicle = {t["vehicle_id"]: t for t in db.list_trips() if t["status"] == "open"}
    for v in vehicles:
        t = open_by_vehicle.get(v["id"])
        v["on_territory"] = t is not None
        v["current_warehouse"] = warehouses.get(t["warehouse_id"])["name"] if t else None
    frames = list_frames()
    used = used_frames()
    vlm_status = vlm.status()
    return {
        "vehicles": vehicles,
        "warehouses": [warehouse_summary(w) for w in warehouses.WAREHOUSES],
        "on_territory": len(open_by_vehicle),
        "camera": {"frames": frames, "used": used, "next": next((f for f in frames if f not in set(used)), None),
                   "folder": str(config.DATA_DIR)},
        "vlm": vlm_status,
        "scale": scale.status(),
        # Старые ключи — для совместимости с интерфейсом.
        "vlm_enabled": vlm_status["enabled"],
        "vlm_model": vlm_status["model"] if vlm_status["enabled"] else None,
        "server_time": now_iso(),
    }


def vehicle_details(vehicle_id: str) -> dict | None:
    v = db.get_vehicle(vehicle_id)
    if not v:
        return None
    trips = db.list_trips(vehicle_id=vehicle_id)
    for t in trips:
        t["warehouse"] = warehouses.get(t["warehouse_id"])["name"]
    msgs = [m for m in db.list_messages(limit=1000) if m.get("vehicle_id") == vehicle_id]
    v["on_territory"] = any(t["status"] == "open" for t in trips)
    return {"vehicle": v, "trips": trips, "messages": msgs}


def warehouse_details(warehouse_id: str) -> dict | None:
    wh = warehouses.BY_ID.get(warehouse_id)
    if not wh:
        return None
    data = warehouse_summary(wh)
    trips = db.list_trips(warehouse_id=warehouse_id)
    for t in trips:
        t["vehicle"] = db.get_vehicle(t["vehicle_id"])
    data["trips"] = trips
    data["messages"] = [m for m in db.list_messages(limit=1000)
                        if m.get("warehouse_id") == warehouse_id and m["role"] == "ai"]
    return data
