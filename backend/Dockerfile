FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY yolov8n.pt* ./
RUN mkdir -p /app/data /app/processed /app/models
ENV DATABASE_PATH=/app/data/crowdguard.db \
    PROCESSED_DIR=/app/processed \
    MODEL_PATH=/app/models/crowd_risk_model.joblib
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
