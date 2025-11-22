# 🎥 Backend - Reconocimiento Facial con FaceNet

API FastAPI para reconocimiento facial usando DeepFace y FaceNet.

## 🚀 Inicio Rápido

### Instalación Local

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

La API estará disponible en `http://127.0.0.1:8000`

### Documentación Interactiva

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## 📝 Endpoints

### GET `/health`
Verifica el estado del servidor.

**Respuesta:**
```json
{
  "status": "ok",
  "message": "Servidor de reconocimiento facial activo",
  "model": "Facenet",
  "threshold": 0.6
}
```

### POST `/compare-faces`
Compara un selfie con múltiples fotos.

**Parámetros:**
- `selfie` (file): Imagen del rostro a comparar
- `photos` (files): Lista de imágenes para comparar

**Respuesta:**
```json
{
  "status": "success",
  "selfie": "selfie.jpg",
  "matches": [
    {
      "file": "foto1.jpg",
      "similarity": 85.5
    }
  ],
  "non_matches": [
    {
      "file": "foto2.jpg",
      "similarity": 45.2,
      "reason": "Similitud por debajo del umbral"
    }
  ],
  "statistics": {
    "total_photos": 5,
    "matches_count": 1,
    "non_matches_count": 4,
    "errors_count": 0,
    "match_percentage": 20.0,
    "threshold_used": 60.0
  }
}
```

## 🔧 Configuración

### Variables de Entorno

Copia `.env.example` a `.env` y personaliza:

```bash
cp .env.example .env
```

Variables disponibles:
- `PORT`: Puerto del servidor (default: 8000)
- `HOST`: Host del servidor (default: 0.0.0.0)
- `SIMILARITY_THRESHOLD`: Umbral de similitud (0-1, default: 0.60)
- `MODEL_NAME`: Modelo a usar (default: Facenet)
- `LOG_LEVEL`: Nivel de logging (default: INFO)
- `MAX_FILE_SIZE`: Tamaño máximo de archivo en bytes (default: 5MB)

## 🐳 Docker

### Construir imagen

```bash
docker build -t surf-backend .
```

### Ejecutar contenedor

```bash
docker run -p 8000:8000 surf-backend
```

## 🚂 Despliegue en Railway

### Requisitos

- Cuenta en Railway
- Repositorio en GitHub

### Pasos

1. Conecta tu repositorio a Railway
2. Railway detectará automáticamente el `Procfile`
3. Configura variables de entorno si es necesario
4. Despliega

La aplicación estará disponible en la URL proporcionada por Railway.

## 📊 Estructura del Código

```
backend/
├── main.py              # Aplicación principal
├── requirements.txt     # Dependencias
├── Procfile            # Configuración para Railway
├── runtime.txt         # Versión de Python
├── .env.example        # Variables de entorno de ejemplo
└── README.md           # Este archivo
```

## 🔍 Funciones Principales

### `validate_image_file(file)`
Valida que el archivo sea una imagen válida.

### `read_image_to_array(file)`
Lee un archivo de imagen y lo convierte a array numpy.

### `detect_face(image)`
Detecta si hay un rostro en la imagen usando OpenCV.

### `extract_face_embedding(image)`
Extrae el embedding facial usando DeepFace + FaceNet.

### `calculate_similarity(embedding1, embedding2)`
Calcula la similitud entre dos embeddings usando cosine similarity.

## 🐛 Solución de Problemas

### Error: "Cannot import 'setuptools.build_meta'"
Asegúrate de que `setuptools` esté en `requirements.txt`.

### Error: "No se detectó rostro"
- La imagen debe ser clara
- El rostro debe estar visible y bien iluminado
- Intenta con una imagen diferente

### Error: "Archivo demasiado grande"
- Máximo permitido: 5MB
- Comprime la imagen
- Usa formatos: JPG, PNG, GIF, BMP

## 📚 Dependencias

- **FastAPI**: Framework web moderno
- **Uvicorn**: Servidor ASGI
- **DeepFace**: Librería de reconocimiento facial
- **OpenCV**: Procesamiento de imágenes
- **NumPy**: Cálculos numéricos
- **SciPy**: Funciones científicas
- **Pillow**: Procesamiento de imágenes

## 🔐 Seguridad

- CORS habilitado para todas las URLs (configurable)
- Validación de entrada en todos los endpoints
- Límite de tamaño de archivo
- Manejo de errores robusto

## 📈 Performance

- Caché de modelos para evitar recargas
- Procesamiento eficiente de imágenes
- Embeddings de baja dimensionalidad (128D)

## 📞 Soporte

Para reportar bugs o sugerencias, abre un issue en GitHub.

---

**Desarrollado con ❤️ usando FastAPI y FaceNet**