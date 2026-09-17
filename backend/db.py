"""SQLite-хранилище: машины, рейсы (заезд/выезд), сообщения чата.

«Память» системы — это эта база, а не дообучение модели.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from backend import config
from backend.pipeline.makes import same_make
from backend.vehicle_id import display_name

SCHEMA = """
CREATE TABLE IF NOT EXISTS vehicles (
    id TEXT PRIMARY KEY,
    plate TEXT,
    plate_formatted TEXT,
    manufacturer TEXT,
    model TEXT,
    equipment_type TEXT,
    year TEXT,
    color TEXT,
    appearance TEXT,          -- JSON: описание внешности от VLM
    fingerprint TEXT,         -- JSON: цветовая гистограмма вырезки
    photo TEXT,               -- путь к вырезке машины
    is_temporary INTEGER DEFAULT 0,
    first_seen TEXT,
    last_seen TEXT,
    visits INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS trips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id TEXT NOT NULL,
    warehouse_id TEXT NOT NULL,
    driver TEXT,
    crop TEXT,
    entry_time TEXT,
    entry_weight REAL,
    entry_message_id INTEGER,
    entry_load_state TEXT,
    entry_has_trailer INTEGER,
    exit_time TEXT,
    exit_weight REAL,
    exit_message_id INTEGER,
    exit_load_state TEXT,
    exit_has_trailer INTEGER,
    net_weight REAL,
    status TEXT DEFAULT 'open',   -- open / closed
    alerts TEXT DEFAULT '[]'      -- JSON-список предупреждений
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    role TEXT,                -- guard / ai
    text TEXT,
    image TEXT,               -- исходный кадр (URL)
    files TEXT,               -- JSON: {annotated, plate_zoom, json, csv}
    payload TEXT,             -- JSON: полный отчёт / данные охранника
    vehicle_id TEXT,
    warehouse_id TEXT,
    trip_id INTEGER,
    event_kind TEXT           -- entry / exit / none
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    kind TEXT,                -- entry / exit / none
    capture_id TEXT,          -- по нему ловим повторную отправку того же снимка
    vehicle_id TEXT,
    trip_id INTEGER,
    guard_message_id INTEGER,
    ai_message_id INTEGER,
    snapshot TEXT,            -- JSON: строки vehicles/trips ДО события (для отмены)
    status TEXT DEFAULT 'pending',  -- pending (обработка) / active / undone
    undone_at TEXT,
    edits TEXT DEFAULT '[]'   -- JSON: журнал правок весовщицы
);
"""

# Поиск всегда идёт по номеру, по рейсам машины/склада и по событию снимка.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_vehicles_plate ON vehicles(plate);
CREATE INDEX IF NOT EXISTS idx_trips_vehicle ON trips(vehicle_id, status);
CREATE INDEX IF NOT EXISTS idx_trips_warehouse ON trips(warehouse_id);
CREATE INDEX IF NOT EXISTS idx_messages_vehicle ON messages(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_messages_trip ON messages(trip_id);
CREATE INDEX IF NOT EXISTS idx_events_capture ON events(capture_id, status);
CREATE INDEX IF NOT EXISTS idx_events_status ON events(status, id);
"""


def _connect() -> sqlite3.Connection:
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


_con: sqlite3.Connection | None = None


# Колонки, добавленные после первых запусков: старой базе дописываем их на ходу.
MIGRATIONS = [
    ("vehicles", "year", "TEXT"),
    # Откуда взялся вес: введён весовщицей или придуман генератором (весов в демо нет).
    ("trips", "entry_weight_source", "TEXT"),
    ("trips", "exit_weight_source", "TEXT"),
    # Откуда взято время события: с надписи камеры или системное. Сравнивать между собой
    # их нельзя — на части кадров часы камеры показывают 2019 год.
    ("trips", "entry_time_source", "TEXT"),
    ("trips", "exit_time_source", "TEXT"),
]


def init() -> None:
    global _con
    _con = _connect()
    _con.executescript(SCHEMA)
    for table, column, ctype in MIGRATIONS:
        existing = {r[1] for r in _con.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            _con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ctype}")
    _con.executescript(INDEXES)
    _con.commit()


@contextmanager
def cursor():
    if _con is None:
        init()
    cur = _con.cursor()
    try:
        yield cur
        _con.commit()
    except Exception:
        # Иначе половина правки (например, рейс без машины) останется в базе.
        _con.rollback()
        raise
    finally:
        cur.close()


def _row(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for k in ("appearance", "fingerprint", "alerts", "files", "payload", "snapshot", "edits"):
        if k in d and isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except json.JSONDecodeError:
                pass
    return d


# ---------- vehicles ----------

def _vehicle(r: sqlite3.Row | None) -> dict | None:
    v = _row(r)
    if v is not None:
        v["label"] = display_name(v)
    return v


def get_vehicle(vehicle_id: str) -> dict | None:
    with cursor() as cur:
        return _vehicle(cur.execute("SELECT * FROM vehicles WHERE id=?", (str(vehicle_id),)).fetchone())


def get_vehicle_by_plate(plate: str | None) -> dict | None:
    if not plate:
        return None
    with cursor() as cur:
        return _vehicle(cur.execute(
            "SELECT * FROM vehicles WHERE plate=? ORDER BY last_seen DESC LIMIT 1", (plate,)).fetchone())


def next_board_no() -> str:
    """Бортовой номер: 1, 2, 3… — следующий за максимальным в базе."""
    with cursor() as cur:
        row = cur.execute("SELECT MAX(CAST(id AS INTEGER)) FROM vehicles WHERE id GLOB '[0-9]*'").fetchone()
    return str((row[0] or 0) + 1)


def list_vehicles() -> list[dict]:
    with cursor() as cur:
        return [_vehicle(r) for r in cur.execute(
            "SELECT * FROM vehicles ORDER BY CAST(id AS INTEGER) ASC, last_seen DESC").fetchall()]


def upsert_vehicle(v: dict) -> None:
    existing = get_vehicle(v["id"])
    with cursor() as cur:
        if existing is None:
            cur.execute(
                "INSERT INTO vehicles (id, plate, plate_formatted, manufacturer, model, equipment_type, year, color, "
                "appearance, fingerprint, photo, is_temporary, first_seen, last_seen, visits) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                (v["id"], v.get("plate"), v.get("plate_formatted"), v.get("manufacturer"), v.get("model"),
                 v.get("equipment_type"), v.get("year"), v.get("color"),
                 json.dumps(v.get("appearance"), ensure_ascii=False), json.dumps(v.get("fingerprint")),
                 v.get("photo"), int(v.get("is_temporary", False)), v["seen_at"], v["seen_at"]),
            )
        else:
            # Не затираем известные поля пустыми значениями.
            merged = {k: (v.get(k) if v.get(k) not in (None, "", "не определена") else existing.get(k))
                      for k in ("plate", "plate_formatted", "manufacturer", "model", "equipment_type", "year",
                                "color", "photo")}
            # Марка сменилась (например, реестр поправил надпись) — старые модель и год к ней не относятся.
            if v.get("manufacturer") and not same_make(v["manufacturer"], existing.get("manufacturer")):
                merged["model"], merged["year"] = v.get("model"), v.get("year")
            appearance = v.get("appearance") or existing.get("appearance")
            fingerprint = v.get("fingerprint") or existing.get("fingerprint")
            cur.execute(
                "UPDATE vehicles SET plate=?, plate_formatted=?, manufacturer=?, model=?, equipment_type=?, year=?, "
                "color=?, appearance=?, fingerprint=?, photo=?, is_temporary=MIN(is_temporary, ?), last_seen=?, "
                "visits=visits+1 WHERE id=?",
                (merged["plate"], merged["plate_formatted"], merged["manufacturer"], merged["model"],
                 merged["equipment_type"], merged["year"], merged["color"], json.dumps(appearance, ensure_ascii=False),
                 json.dumps(fingerprint), merged["photo"], int(v.get("is_temporary", False)), v["seen_at"], v["id"]),
            )


# ---------- trips ----------

def open_trip_for(vehicle_id: str) -> dict | None:
    with cursor() as cur:
        return _row(cur.execute(
            "SELECT * FROM trips WHERE vehicle_id=? AND status='open' ORDER BY id DESC", (vehicle_id,)).fetchone())


def create_trip(t: dict) -> int:
    with cursor() as cur:
        cur.execute(
            "INSERT INTO trips (vehicle_id, warehouse_id, driver, crop, entry_time, entry_weight, entry_message_id, "
            "entry_load_state, entry_has_trailer, alerts, entry_weight_source, entry_time_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (t["vehicle_id"], t["warehouse_id"], t.get("driver"), t.get("crop"), t["entry_time"],
             t.get("entry_weight"), t.get("entry_message_id"), t.get("entry_load_state"),
             t.get("entry_has_trailer"), json.dumps(t.get("alerts", []), ensure_ascii=False),
             t.get("entry_weight_source"), t.get("entry_time_source")),
        )
        return cur.lastrowid


def close_trip(trip_id: int, t: dict) -> None:
    with cursor() as cur:
        cur.execute(
            "UPDATE trips SET exit_time=?, exit_weight=?, exit_message_id=?, exit_load_state=?, exit_has_trailer=?, "
            "net_weight=?, status='closed', alerts=?, exit_weight_source=?, exit_time_source=?, "
            "driver=COALESCE(?, driver), crop=COALESCE(?, crop) WHERE id=?",
            (t["exit_time"], t.get("exit_weight"), t.get("exit_message_id"), t.get("exit_load_state"),
             t.get("exit_has_trailer"), t.get("net_weight"), json.dumps(t.get("alerts", []), ensure_ascii=False),
             t.get("exit_weight_source"), t.get("exit_time_source"),
             t.get("driver"), t.get("crop"), trip_id),
        )


# Что весовщица может поправить у уже оформленного события.
TRIP_EDITABLE = {"driver", "crop", "entry_weight", "exit_weight", "net_weight",
                 "entry_weight_source", "exit_weight_source"}
VEHICLE_EDITABLE = {"plate", "plate_formatted", "is_temporary"}


def _update(table: str, allowed: set[str], key: str, key_value, fields: dict) -> None:
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    with cursor() as cur:
        cur.execute(f"UPDATE {table} SET {sets} WHERE {key}=?", (*fields.values(), key_value))


def update_trip(trip_id: int, **fields) -> None:
    _update("trips", TRIP_EDITABLE, "id", trip_id, fields)


def update_vehicle(vehicle_id: str, **fields) -> None:
    _update("vehicles", VEHICLE_EDITABLE, "id", str(vehicle_id), fields)


def get_trip(trip_id: int) -> dict | None:
    with cursor() as cur:
        return _row(cur.execute("SELECT * FROM trips WHERE id=?", (trip_id,)).fetchone())


def list_trips(vehicle_id: str | None = None, warehouse_id: str | None = None) -> list[dict]:
    q, args = "SELECT * FROM trips", []
    conds = []
    if vehicle_id:
        conds.append("vehicle_id=?"); args.append(vehicle_id)
    if warehouse_id:
        conds.append("warehouse_id=?"); args.append(warehouse_id)
    if conds:
        q += " WHERE " + " AND ".join(conds)
    q += " ORDER BY id DESC"
    with cursor() as cur:
        return [_row(r) for r in cur.execute(q, args).fetchall()]


# ---------- messages ----------

def add_message(m: dict) -> int:
    with cursor() as cur:
        cur.execute(
            "INSERT INTO messages (created_at, role, text, image, files, payload, vehicle_id, warehouse_id, trip_id, "
            "event_kind) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (m["created_at"], m["role"], m.get("text"), m.get("image"),
             json.dumps(m.get("files") or {}, ensure_ascii=False), json.dumps(m.get("payload"), ensure_ascii=False),
             m.get("vehicle_id"), m.get("warehouse_id"), m.get("trip_id"), m.get("event_kind")),
        )
        return cur.lastrowid


def update_message(message_id: int, **fields) -> None:
    if not fields:
        return
    for k in ("files", "payload"):
        if k in fields:
            fields[k] = json.dumps(fields[k], ensure_ascii=False)
    sets = ", ".join(f"{k}=?" for k in fields)
    with cursor() as cur:
        cur.execute(f"UPDATE messages SET {sets} WHERE id=?", (*fields.values(), message_id))


def list_messages(limit: int = 200) -> list[dict]:
    with cursor() as cur:
        rows = cur.execute("SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in reversed(rows)]


def guard_frames() -> list[str]:
    """Кадры, уже отправленные весовщицей. Отдельный запрос, а не обрезанная лента чата:
    после тысячи сообщений очередь кадров начинала выдавать уже обработанные."""
    with cursor() as cur:
        rows = cur.execute("SELECT payload FROM messages WHERE role='guard' AND payload IS NOT NULL").fetchall()
    frames = []
    for r in rows:
        try:
            p = json.loads(r["payload"])
        except (json.JSONDecodeError, TypeError):
            continue
        # Отменённый заезд возвращает кадр в очередь: весовщица переснимет ту же машину.
        if isinstance(p, dict) and p.get("frame") and not p.get("undone"):
            frames.append(p["frame"])
    return frames


def get_message(message_id: int) -> dict | None:
    with cursor() as cur:
        return _row(cur.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone())


# ---------- события: отмена и правка ----------

def create_event(e: dict) -> int:
    with cursor() as cur:
        cur.execute(
            "INSERT INTO events (created_at, kind, capture_id, vehicle_id, trip_id, guard_message_id, "
            "ai_message_id, snapshot, status, edits) VALUES (?,?,?,?,?,?,?,?,?,'[]')",
            (e["created_at"], e.get("kind"), e.get("capture_id"), e.get("vehicle_id"), e.get("trip_id"),
             e.get("guard_message_id"), e.get("ai_message_id"),
             json.dumps(e.get("snapshot"), ensure_ascii=False), e.get("status", "pending")),
        )
        return cur.lastrowid


EVENT_FIELDS = {"kind", "vehicle_id", "trip_id", "guard_message_id", "ai_message_id", "snapshot", "status"}


def update_event(event_id: int, **fields) -> None:
    fields = {k: v for k, v in fields.items() if k in EVENT_FIELDS}
    if not fields:
        return
    if "snapshot" in fields:
        fields["snapshot"] = json.dumps(fields["snapshot"], ensure_ascii=False)
    sets = ", ".join(f"{k}=?" for k in fields)
    with cursor() as cur:
        cur.execute(f"UPDATE events SET {sets} WHERE id=?", (*fields.values(), event_id))


def event_for_capture(capture_id: str) -> dict | None:
    """Событие этого снимка, если он уже отправлялся и не был отменён."""
    with cursor() as cur:
        return _row(cur.execute(
            "SELECT * FROM events WHERE capture_id=? AND status IN ('pending','active') "
            "ORDER BY id DESC LIMIT 1", (capture_id,)).fetchone())


def last_active_event() -> dict | None:
    """Последнее оформленное событие: только заезд или выезд — отменять и править имеет смысл их."""
    with cursor() as cur:
        return _row(cur.execute(
            "SELECT * FROM events WHERE status='active' AND kind IN ('entry','exit') "
            "ORDER BY id DESC LIMIT 1").fetchone())


def mark_event_undone(event_id: int, at: str) -> None:
    with cursor() as cur:
        cur.execute("UPDATE events SET status='undone', undone_at=? WHERE id=?", (at, event_id))


def append_event_edit(event_id: int, edit: dict) -> None:
    """Журнал правок остаётся в базе: видно, что и когда исправили."""
    with cursor() as cur:
        row = cur.execute("SELECT edits FROM events WHERE id=?", (event_id,)).fetchone()
        log = json.loads((row["edits"] if row else None) or "[]")
        log.append(edit)
        cur.execute("UPDATE events SET edits=? WHERE id=?", (json.dumps(log, ensure_ascii=False), event_id))


def raw_vehicle(vehicle_id: str) -> dict | None:
    """Строка как есть, без разбора JSON: годится, чтобы записать её обратно при отмене."""
    with cursor() as cur:
        r = cur.execute("SELECT * FROM vehicles WHERE id=?", (str(vehicle_id),)).fetchone()
    return dict(r) if r else None


def raw_trip(trip_id: int) -> dict | None:
    with cursor() as cur:
        r = cur.execute("SELECT * FROM trips WHERE id=?", (trip_id,)).fetchone()
    return dict(r) if r else None


def _write_row(cur: sqlite3.Cursor, table: str, row: dict) -> None:
    # Колонки сверяем с таблицей: снимок лежит в базе как JSON, лишним ключам там не место.
    cols = [c[1] for c in cur.execute(f"PRAGMA table_info({table})") if c[1] in row]
    cur.execute(f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [row[c] for c in cols])


def restore_state(snapshot: dict, vehicle_id: str | None, trip_id: int | None) -> None:
    """Возврат к состоянию до события. Чего в снимке нет — событие это и создало, удаляем."""
    snapshot = snapshot or {}
    with cursor() as cur:
        if trip_id is not None:
            trip = snapshot.get("trip")
            if trip:
                _write_row(cur, "trips", trip)
            else:
                cur.execute("DELETE FROM trips WHERE id=?", (trip_id,))
        if vehicle_id is not None:
            vehicle = snapshot.get("vehicle")
            if vehicle:
                _write_row(cur, "vehicles", vehicle)
            else:
                cur.execute("DELETE FROM vehicles WHERE id=?", (str(vehicle_id),))


def reset() -> None:
    with cursor() as cur:
        cur.execute("DELETE FROM messages")
        cur.execute("DELETE FROM events")
        cur.execute("DELETE FROM trips")
        cur.execute("DELETE FROM vehicles")
        # Нумерация рейсов и сообщений снова с 1 (таблица есть, только если был AUTOINCREMENT-insert).
        if cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'").fetchone():
            cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('trips', 'messages', 'events')")
