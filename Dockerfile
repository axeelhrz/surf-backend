FROM python:3.11

WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV, TensorFlow y DeepFace
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip y setuptools
RUN pip install --upgrade pip setuptools wheel

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código del backend
COPY . .

# Exponer puerto (Railway usa la variable PORT)
EXPOSE 8000

# Comando para iniciar la aplicación
# Railway inyecta la variable PORT automáticamente
CMD python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}