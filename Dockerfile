FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV y InsightFace
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip y setuptools
RUN pip install --upgrade pip setuptools wheel

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código
COPY . .

# Copiar y dar permisos al script de inicio
COPY start.sh .
RUN chmod +x start.sh

# Exponer puerto (Railway usa la variable PORT)
EXPOSE 8000

# Comando para iniciar la aplicación usando el script
# El script usa la variable PORT de Railway
CMD ["./start.sh"]