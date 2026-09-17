"""HTTP API и раздача интерфейса. Запуск: python -m backend.main"""
from __future__ import annotations

import asyncio
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import config, db, service

FRONTEND_DIR = config.ROOT / "frontend"


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    service.CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Весовая · контроль техники", version="0.1", lifespan=lifespan)

# Один кадр обрабатывается за раз: модели на CPU, параллелить смысла нет.
_lock = asyncio.Lock()


class CaptureRequest(BaseModel):
    frame: str | None = None


class ReportRequest(BaseModel):
    capture_id: str
    driver: str | None = None
    crop: str | None = None
    warehouse_id: str | None = None
    weight: str | float | None = None
    note: str | None = None
    plate_override: str | None = None
    frame: str | None = None


@app.get("/api/state")
def get_state():
    return service.state()


@app.get("/api/messages")
def get_messages():
    return db.list_messages()


@app.get("/api/vehicles/{vehicle_id}")
def get_vehicle(vehicle_id: str):
    data = service.vehicle_details(vehicle_id)
    if not data:
        raise HTTPException(404, "Машина не найдена")
    return data


@app.get("/api/warehouses/{warehouse_id}")
def get_warehouse(warehouse_id: str):
    data = service.warehouse_details(warehouse_id)
    if not data:
        raise HTTPException(404, "Склад не найден")
    return data


@app.post("/api/camera/capture")
def camera_capture(req: CaptureRequest):
    try:
        return service.capture_frame(req.frame)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    if suffix not in service.IMAGE_EXT:
        raise HTTPException(400, "Нужно изображение (jpg/png)")
    tmp = service.CAPTURES_DIR / f"upload_{Path(file.filename).stem}{suffix}"
    service.CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    with tmp.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    result = service.save_capture(tmp, None)
    tmp.unlink(missing_ok=True)
    return result


@app.post("/api/report")
async def create_report(req: ReportRequest):
    async with _lock:
        try:
            return await asyncio.to_thread(service.process_capture, req.capture_id, req.model_dump())
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            raise HTTPException(500, f"Ошибка обработки: {e}")


@app.post("/api/reset")
def reset():
    db.reset()
    return {"ok": True}


@app.get("/frames/{name}")
def frame(name: str):
    path = config.DATA_DIR / name
    if not path.exists() or path.suffix.lower() not in service.IMAGE_EXT:
        raise HTTPException(404, "Кадр не найден")
    return FileResponse(path)


config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=config.RESULTS_DIR), name="results")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=config.HOST, port=config.PORT, reload=False)
