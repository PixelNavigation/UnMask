# ---------- Stage 1: Build Next.js / React frontend (adjust if using CRA) ----------
FROM node:18 AS frontend-builder
WORKDIR /frontend
COPY frontend/ .
RUN npm install && npm run build

# ---------- Stage 2: Backend (Flask + Gunicorn) ----------
FROM python:3.10-slim

# System deps (add more if needed: libgl1 etc. for opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Writable cache locations to avoid PermissionError: '/.cache'
ENV HF_HOME=/tmp/.cache/huggingface \
    TORCH_HOME=/tmp/.cache/torch \
    TIMM_CACHE_DIR=/tmp/.cache/timm \
    XDG_CACHE_HOME=/tmp/.cache \
    HF_HUB_CACHE=/tmp/.cache/huggingface \
    TRANSFORMERS_CACHE=/tmp/.cache/huggingface \
    NO_ALBUMENTATIONS_UPDATE=1

RUN mkdir -p /tmp/.cache/huggingface /tmp/.cache/torch /tmp/.cache/timm

# Copy backend requirements first for better layer caching
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend/ ./backend

# Copy built frontend output into backend static (adjust path if build folder differs)
COPY --from=frontend-builder /frontend/dist ./backend/static

WORKDIR /app/backend

# Expose Space port (Hugging Face uses 7860)
EXPOSE 7860

# Start with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "180"]
