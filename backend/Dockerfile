# Backend image. EXTRAS selects dependency groups:
#   api/analytics:  docker build --build-arg EXTRAS=ml .
#   camera worker:  docker build --build-arg EXTRAS=cv,ml .   (use a CUDA base image for GPU)
FROM python:3.12-slim

ARG EXTRAS=""
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TI_CONFIG_DIR=/app/configs TI_DATA_DIR=/app/data TI_MODELS_DIR=/app/models

WORKDIR /app/backend
COPY backend/pyproject.toml ./
COPY backend/README.md ./README.md
COPY backend/traffic_intel ./traffic_intel
RUN pip install --no-cache-dir ".${EXTRAS:+[$EXTRAS]}"
COPY backend/alembic.ini ./
COPY configs /app/configs

EXPOSE 8000
CMD ["uvicorn", "traffic_intel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
