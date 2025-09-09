# ---------- Stage 1: Build Next.js frontend as static export ----------
FROM node:18 AS frontend-builder
WORKDIR /frontend
COPY frontend/ .
# Install and build
RUN npm install && npm run build

# ---------- Stage 2: Python backend with Flask + Gunicorn ----------
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend and requirements
COPY backend/ ./backend
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy exported frontend (Next.js 'out' folder) into Flask static dir
# The Next.js export outputs to frontend/out by default
COPY --from=frontend-builder /frontend/out ./backend/static

# Expose port used by Hugging Face Spaces (Default is 7860)
EXPOSE 7860

# Environment for Flask
ENV PORT=7860 \
    PYTHONPATH=/app

# Start with gunicorn for production
WORKDIR /app/backend
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "180"]
