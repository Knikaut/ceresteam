"""Страница ручной разметки вырезок (запускается на ноутбуке, без интернета).

    python ml/label_tool/server.py <папка labeling с items.json и img/> [порт]

Открыть http://127.0.0.1:8010 (разметка) или http://127.0.0.1:8010/review (проверка по одной машине).
Метки сохраняются в <папка>/labels.json после каждого клика.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8010
LABELS = DATA / "labels.json"
PAGES = {"/": "index.html", "/index.html": "index.html", "/review": "review.html", "/review.html": "review.html"}
# Запросы обслуживаются в потоках: без замка два быстрых сохранения читают один и тот же файл,
# и второе затирает первое (пропадает отметка «проверено»).
lock = threading.Lock()


def load_labels() -> dict:
    try:
        return json.loads(LABELS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_labels(labels: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=DATA, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)
    os.replace(tmp, LABELS)  # целиком или никак: файл не бьётся при сбое


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def translate_path(self, path):
        path = path.split("?")[0]
        if path in PAGES:
            return str(HERE / PAGES[path])
        return str(DATA / path.lstrip("/"))

    def end_headers(self):
        # Страницы и метки меняются на ходу — браузер не должен показывать старую копию
        if not self.path.split("?")[0].endswith(".jpg"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.startswith("/labels.json"):
            with lock:
                return self._json(load_labels())
        return super().do_GET()

    def do_POST(self):
        """/save — заменить метку целиком (null — удалить); /update — поменять только присланные поля."""
        if self.path not in ("/save", "/update"):
            return self.send_error(404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        with lock:
            labels = load_labels()
            for cid, value in body.items():
                if value is None:
                    labels.pop(cid, None)
                elif self.path == "/update":
                    labels.setdefault(cid, {}).update(value)
                else:
                    labels[cid] = value
            save_labels(labels)
            result = {cid: labels.get(cid) for cid in body}
        self._json({"ok": True, "count": len(labels), "labels": result})

    def _json(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print(f"Разметка: {DATA}\nОткройте http://127.0.0.1:{PORT}  (Ctrl+C — выход, метки уже сохранены)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
