FROM python:3.11-slim

# OpenCV and DeepFace system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY backend/ ./backend/

ENV PYTHONPATH=/app
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0

# Pre-download DeepFace emotion weights into the image layer so the first
# request doesn't block on a ~7 MB download at runtime.
RUN python -c "\
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; \
import numpy as np; \
from deepface import DeepFace; \
DeepFace.analyze(np.zeros((100,100,3), dtype='uint8'), \
  actions=['emotion'], enforce_detection=False, silent=True)" \
  || echo 'DeepFace pre-warm skipped (weights will download on first request)'

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
