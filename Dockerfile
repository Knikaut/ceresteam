# Контейнер для проверки организаторами: данные ожидаются в /data (например /data/Auto).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data/Auto RESULTS_DIR=/app/results HOST=0.0.0.0 PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY run_batch.py .

# Веса YOLO и модели EasyOCR скачиваются при первом запуске; для офлайн-контейнера
# положите yolov8s.pt в /app и модели EasyOCR в ~/.EasyOCR/model.
EXPOSE 8000

# По умолчанию — пакетная обработка /data/Auto -> /app/results/batch.
# Веб-интерфейс: docker run ... python -m backend.main
CMD ["python", "run_batch.py", "--input", "/data/Auto", "--output", "/app/results/batch"]
