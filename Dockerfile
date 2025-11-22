# ============================================================
# Stage 1: Builder - Compilar dependencias
# ============================================================
FROM python:3.8-slim AS builder

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Instalar herramientas de compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    pkg-config \
    libhdf5-dev \
    libhdf5-serial-dev \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Crear virtual environment e instalar dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================================
# Stage 2: Runtime - Imagen final limpia
# ============================================================
FROM python:3.8-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000 \
    PYTHONPATH=/install/lib/python3.8/site-packages

# Instalar solo dependencias runtime 
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1-mesa-glx \
    libhdf5-103 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copiar dependencias instaladas desde builder
COPY --from=builder /install /install

# Copiar código de la aplicación
COPY app/ ./app/
COPY model/ ./model/

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Comando de inicio
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
