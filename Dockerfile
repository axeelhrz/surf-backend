FROM python:3.13

WORKDIR /app

# Instalar todas las dependencias del sistema necesarias para OpenCV y DeepFace
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libxkbcommon0 \
    libdbus-1-3 \
    libxcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkb1 \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip y setuptools
RUN pip install --upgrade pip setuptools wheel

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY main.py .

# Exponer puerto
EXPOSE 8000

# Comando para iniciar la aplicación
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]