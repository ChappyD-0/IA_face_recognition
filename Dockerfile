# ============================================================
# IAFace API - Dockerfile optimizado para Render
# ============================================================

FROM python:3.8-slim

WORKDIR /app

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libglib2.0-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1-mesa-glx \
    curl \
    gcc \
    g++ \
    make \
    pkg-config \
    libhdf5-dev \
    libhdf5-serial-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN echo "Installing Python dependencies..." && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip && \
    echo "Dependencies installed"

# Verificar instalación de uvicorn
RUN python -c "import uvicorn; print(f' uvicorn {uvicorn.__version__} installed')"

# Copiar código de la aplicación
COPY app/ ./app/
COPY model/ ./model/

# Crear usuario no-root 
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app

USER apiuser

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8000}/health || exit 1

# Comando de inicio
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
