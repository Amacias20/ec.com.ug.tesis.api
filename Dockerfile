# Dockerfile — FastAPI + PyTorch Backend
# Railway detecta este archivo y lo usa para construir el contenedor

FROM python:3.11-slim

# Evita que Python genere archivos .pyc y activa logs sin buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema necesarias para psycopg y otros paquetes
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias Python primero (aprovecha cache de Docker)
COPY requirements.txt .

# Instalar PyTorch CPU-only ANTES del resto (evita descargar la versión CUDA de ~2.5GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Instalar el resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente completo (incluye app/ y artifacts/)
COPY . .

# Railway inyecta $PORT automáticamente — uvicorn debe escuchar en ese puerto
# El host 0.0.0.0 es obligatorio para que Railway enrute correctamente
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
