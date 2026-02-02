from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import cv2
import numpy as np
import insightface
from PIL import Image, ImageDraw, ImageFont
import io
import json
import shutil
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from stripe_endpoints import router as stripe_router
from admin_endpoints import router as admin_router
from embeddings_clustering import EmbeddingsClusteringSystem

# Crear aplicación FastAPI
app = FastAPI(
    title="Reconocimiento Facial con InsightFace",
    description="API para comparar rostros usando InsightFace y ArcFace",
    version="2.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers de Stripe y Admin
app.include_router(stripe_router)
app.include_router(admin_router)

# Configuración
# Umbral de similitud para InsightFace (distancia coseno)
# InsightFace usa distancia coseno donde valores más bajos = mayor similitud
# 0.4 = balance (recomendado para la mayoría de casos)
# 0.35 = más estricto
# 0.5 = más flexible
# 0.56 = muy flexible (detecta variaciones significativas)
# Configuración optimizada para máxima precisión
# Umbral: 0.56 = 44% similitud mínima (muy sensible para detectar variaciones)
SIMILARITY_THRESHOLD = 0.56  # Distancia coseno (mayor = más permisivo)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MIN_FACE_SIZE = 20  # Tamaño mínimo del rostro en píxeles (reducido)
MIN_FACE_CONFIDENCE = 0.3  # Confianza mínima del detector (reducida)
USE_MULTIPLE_FACES = False  # NO usar múltiples rostros
DEBUG_MODE = True  # Activar logging detallado

# Configuración de procesamiento paralelo
MAX_WORKERS = min(32, (multiprocessing.cpu_count() or 1) * 4)  # 4 threads por CPU
BATCH_SIZE = 50  # Procesar en lotes de 50 fotos

# Inicializar modelo InsightFace con mejores prácticas
# Usar el modelo 'buffalo_l' que es el más preciso y robusto
print("🔄 Inicializando InsightFace...")
face_analysis = None

try:
    import onnxruntime as ort
    
    # Configurar providers (GPU primero si está disponible)
    providers = []
    available_providers = ort.get_available_providers()
    
    if 'CUDAExecutionProvider' in available_providers:
        providers.append('CUDAExecutionProvider')
        print("🚀 GPU detectada, usando CUDA")
    elif 'TensorrtExecutionProvider' in available_providers:
        providers.append('TensorrtExecutionProvider')
        print("🚀 TensorRT detectado")
    
    providers.append('CPUExecutionProvider')
    
    # Inicializar FaceAnalysis con el modelo buffalo_l (el mejor)
    # buffalo_l incluye: det_10g, rec_2, y otros modelos optimizados
    face_analysis = insightface.app.FaceAnalysis(
        name='buffalo_l',  # Modelo más robusto y preciso
        providers=providers
    )
    
    # Preparar el modelo con tamaño de detección optimizado
    # Usar tamaño más grande para mejor precisión
    face_analysis.prepare(ctx_id=0, det_size=(960, 960))
    print("✅ InsightFace 'buffalo_l' cargado exitosamente con detección 960x960")
    
except Exception as e:
    print(f"⚠️ Error cargando InsightFace con buffalo_l: {e}")
    print("⚠️ Intentando con modelo por defecto...")
    try:
        # Fallback: usar modelo por defecto
        face_analysis = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        face_analysis.prepare(ctx_id=-1, det_size=(640, 640))
        print("✅ InsightFace cargado con modelo por defecto (CPU)")
    except Exception as e2:
        print(f"❌ Error crítico: {e2}")
        print("❌ InsightFace no pudo ser cargado. Verifica la instalación.")
        face_analysis = None

# Obtener el directorio base del script
BASE_DIR = Path(__file__).parent.absolute()

# Usar variables de entorno para las rutas de almacenamiento
# En Railway, estas apuntarán a los volumes montados
# En local, usará las carpetas por defecto
import os
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "photos_storage")))
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", str(BASE_DIR / "embeddings_storage")))

# Crear directorios si no existen
STORAGE_DIR.mkdir(exist_ok=True, parents=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True, parents=True)

# Inicializar sistema de clustering
clustering_system = EmbeddingsClusteringSystem(
    storage_dir=STORAGE_DIR,
    embeddings_dir=EMBEDDINGS_DIR,
    debug=DEBUG_MODE
)

# Executor dedicado para tareas pesadas (indexado/embeddings)
# Importante: 1 worker para evitar saturar CPU/GPU y duplicados.
INDEX_EXECUTOR = ThreadPoolExecutor(max_workers=1)

print(f"📁 STORAGE_DIR: {STORAGE_DIR}")
print(f"📁 STORAGE_DIR existe: {STORAGE_DIR.exists()}")
print(f"📁 EMBEDDINGS_DIR: {EMBEDDINGS_DIR}")

@app.get("/health")
async def health_check():
    """Verifica el estado del servidor"""
    return {
        "status": "ok",
        "message": "Servidor de reconocimiento facial activo",
        "model": "InsightFace (ArcFace)",
        "threshold": SIMILARITY_THRESHOLD,
        "threshold_percentage": round((1 - SIMILARITY_THRESHOLD) * 100, 2),
        "min_face_size": MIN_FACE_SIZE,
        "min_face_confidence": MIN_FACE_CONFIDENCE,
        "detection_size": "1280x1280 (high-resolution)",
        "model_loaded": face_analysis is not None,
        "features": [
            "Detección de rostros de perfil",
            "Múltiples rostros por imagen",
            "Comparación con todos los píxeles del rostro",
            "Criterios flexibles para perfil",
            "Múltiples métricas de similitud",
            "Mejora de imagen automática",
            "Detección de alta resolución"
        ]
    }

def validate_image_file(file: UploadFile) -> bool:
    """Valida que el archivo sea una imagen válida"""
    allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
    file_extension = file.filename.split('.')[-1].lower()
    return file_extension in allowed_extensions

async def read_image_to_array(file: UploadFile) -> np.ndarray:
    """
    Lee un archivo de imagen y lo convierte a array numpy con preprocesamiento mejorado.
    
    Aplica mejoras de calidad de imagen para optimizar la detección facial.
    """
    try:
        contents = await file.read()
        
        # Validar que el archivo no esté vacío
        if len(contents) == 0:
            raise HTTPException(
                status_code=400, 
                detail="El archivo está vacío. Por favor, sube una imagen válida."
            )
        
        # Validar tamaño
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"Archivo demasiado grande. Máximo: {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        # Intentar abrir la imagen
        try:
            image = Image.open(io.BytesIO(contents))
            # Verificar que la imagen se pueda cargar
            image.verify()
            
            # Reabrir la imagen después de verify (verify cierra el archivo)
            image = Image.open(io.BytesIO(contents))
            
        except Exception as e:
            print(f"❌ Error abriendo imagen: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"El archivo no es una imagen válida o está corrupto. Formatos soportados: JPG, PNG, GIF, BMP. Error: {str(e)}"
            )
        
        # Convertir a array numpy
        try:
            image_array = np.array(image)
        except Exception as e:
            print(f"❌ Error convirtiendo imagen a array: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error procesando la imagen. Por favor, intenta con otra imagen."
            )
        
        # Validar que la imagen tenga dimensiones válidas
        if image_array.size == 0:
            raise HTTPException(
                status_code=400,
                detail="La imagen no tiene contenido válido."
            )
        
        # Convertir a BGR si es necesario (OpenCV usa BGR)
        if len(image_array.shape) == 2:
            # Imagen en escala de grises, convertir a BGR
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        elif len(image_array.shape) == 3:
            if image_array.shape[2] == 4:
                # RGBA a BGR
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            elif image_array.shape[2] == 3:
                # RGB a BGR
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            raise HTTPException(
                status_code=400,
                detail="Formato de imagen no soportado. Usa JPG, PNG, GIF o BMP."
            )
        
        # Preprocesamiento básico: asegurar que la imagen tenga buen tamaño
        # Si es muy pequeña, redimensionar (pero mantener aspecto)
        height, width = image_array.shape[:2]
        
        if height == 0 or width == 0:
            raise HTTPException(
                status_code=400,
                detail="La imagen tiene dimensiones inválidas."
            )
        
        min_dimension = 200  # Tamaño mínimo recomendado
        
        if min(height, width) < min_dimension:
            scale = min_dimension / min(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_array = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        if DEBUG_MODE:
            print(f"✅ Imagen cargada: {width}x{height} -> {image_array.shape}")
        
        return image_array
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error inesperado en read_image_to_array: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"Error procesando la imagen: {str(e)}"
        )

def read_image_from_path(file_path: Path) -> np.ndarray:
    """Lee una imagen desde un archivo en disco y la convierte a array numpy"""
    # Leer imagen usando OpenCV directamente (más eficiente)
    image_array = cv2.imread(str(file_path))
    
    if image_array is None:
        # Fallback a PIL si OpenCV falla
        image = Image.open(file_path)
        image_array = np.array(image)
        
        # Convertir a BGR si es necesario
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    return image_array

def detect_face(image: np.ndarray, is_selfie: bool = False) -> bool:
    """
    Detecta si hay un rostro en la imagen usando múltiples métodos y técnicas.
    
    Args:
        image: Imagen en formato numpy array (BGR)
        is_selfie: Si es True, usa criterios MUY permisivos para selfies
    
    Intenta:
    1. InsightFace con imagen original
    2. InsightFace con imagen mejorada (contraste, brillo)
    3. InsightFace con múltiples escalas
    4. OpenCV Haar Cascade como fallback
    """
    try:
        if face_analysis is None:
            raise ValueError("InsightFace model not loaded")
        
        # Asegurar que la imagen esté en formato correcto
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        # MÉTODO 1: Detectar con InsightFace en imagen original
        faces = face_analysis.get(image)
        if len(faces) > 0:
            for face in faces:
                bbox = face.bbox.astype(int)
                face_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                
                # Para selfies, ser EXTREMADAMENTE permisivo
                if is_selfie:
                    # Aceptar cualquier rostro detectado, sin importar tamaño o confianza
                    if DEBUG_MODE:
                        print(f"✅ Rostro detectado en selfie: tamaño={face_size}px, confianza={face.det_score:.3f}")
                    return True
                else:
                    # Para fotos de surf, usar criterios normales
                    if face_size >= MIN_FACE_SIZE and face.det_score >= MIN_FACE_CONFIDENCE:
                        return True
                    if face_size >= MIN_FACE_SIZE * 2:
                        return True
        
        # MÉTODO 2: Mejorar imagen y volver a intentar
        try:
            # Mejorar contraste y brillo
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            faces = face_analysis.get(enhanced)
            if len(faces) > 0:
                if is_selfie:
                    if DEBUG_MODE:
                        print(f"✅ Rostro detectado en selfie (imagen mejorada)")
                    return True
                else:
                    for face in faces:
                        bbox = face.bbox.astype(int)
                        face_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                        if face_size >= MIN_FACE_SIZE:
                            return True
        except Exception as e:
            if DEBUG_MODE:
                print(f"⚠️ Error mejorando imagen: {e}")
        
        # MÉTODO 3: Probar con diferentes escalas
        if is_selfie:
            try:
                height, width = image.shape[:2]
                for scale in [0.5, 0.75, 1.5, 2.0]:
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    
                    faces = face_analysis.get(resized)
                    if len(faces) > 0:
                        if DEBUG_MODE:
                            print(f"✅ Rostro detectado en selfie (escala {scale}x)")
                        return True
            except Exception as e:
                if DEBUG_MODE:
                    print(f"⚠️ Error probando escalas: {e}")
        
        # MÉTODO 4: Fallback a OpenCV Haar Cascade (solo para selfies)
        if is_selfie:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                
                # Parámetros MUY permisivos para selfies
                for scale in [1.05, 1.1, 1.15, 1.2]:
                    for neighbors in [1, 2, 3]:
                        faces = face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=scale,
                            minNeighbors=neighbors,
                            minSize=(10, 10),  # Tamaño mínimo muy pequeño
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                        if len(faces) > 0:
                            if DEBUG_MODE:
                                print(f"✅ Rostro detectado en selfie (OpenCV Haar)")
                            return True
            except Exception as e:
                if DEBUG_MODE:
                    print(f"⚠️ Error con Haar Cascade: {e}")
        
        return False
        
    except Exception as e:
        print(f"❌ Error detectando rostro: {e}")
        return False

def _get_faces_or_fallback(image: np.ndarray, is_selfie: bool):
    """
    Obtiene rostros con InsightFace; si es selfie y no hay rostros, prueba imagen mejorada y escalas.
    Devuelve (imagen_que_funcionó, lista_faces) o (None, []) si no hay rostros.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    faces = face_analysis.get(image)
    if len(faces) > 0:
        return image, faces

    if not is_selfie:
        return None, []

    # MÉTODO 2: Imagen mejorada (CLAHE)
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        faces = face_analysis.get(enhanced)
        if len(faces) > 0:
            if DEBUG_MODE:
                print("  📍 Rostro detectado en imagen mejorada (CLAHE)")
            return enhanced, faces
    except Exception as e:
        if DEBUG_MODE:
            print(f"  ⚠️ Fallback CLAHE: {e}")

    # MÉTODO 3: Múltiples escalas
    height, width = image.shape[:2]
    for scale in [0.5, 0.75, 1.5, 2.0]:
        try:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            faces = face_analysis.get(resized)
            if len(faces) > 0:
                if DEBUG_MODE:
                    print(f"  📍 Rostro detectado en escala {scale}x")
                return resized, faces
        except Exception as e:
            if DEBUG_MODE:
                print(f"  ⚠️ Escala {scale}: {e}")

    return None, []


def extract_face_embedding(image: np.ndarray, is_selfie: bool = False):
    """
    Extrae el embedding facial usando InsightFace.
    
    Si is_selfie=True y no se detecta rostro en la imagen original, prueba con imagen mejorada
    (contraste) y con varias escalas, igual que detect_face, para evitar falsos "no rostro"
    en selfies que sí tienen rostro visible.
    
    Args:
        image: Imagen en formato numpy array (BGR)
        is_selfie: Si True, usa los mismos fallbacks que detect_face (mejorar imagen, escalas)
    
    Returns:
        tuple: (embedding, face_object) - Embedding normalizado y objeto face con atributos
    """
    if face_analysis is None:
        raise ValueError("InsightFace model not loaded")
    
    try:
        # Asegurar formato BGR correcto
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        # Obtener rostros (con fallbacks si es selfie)
        img_to_use, faces = _get_faces_or_fallback(image, is_selfie)
        
        if len(faces) == 0:
            raise HTTPException(
                status_code=400,
                detail="No se detectó ningún rostro en la imagen"
            )
        
        # Seleccionar el rostro con mayor confianza
        best_face = max(faces, key=lambda f: f.det_score)
        
        if DEBUG_MODE:
            bbox = best_face.bbox.astype(int)
            face_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            print(f"  📍 Rostro detectado: tamaño={face_size}px, confianza={best_face.det_score:.3f}")
        
        # Obtener embedding normalizado de InsightFace (usar img_to_use por si vino de fallback)
        embedding = best_face.normed_embedding
        
        # Asegurar normalización
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        
        return np.array(embedding), best_face
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error extrayendo embedding: {e}")
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {str(e)}")

async def compare_faces_from_folder(selfie: UploadFile, folder_name: str, search_label: str, photos_in_folder: List[Path], search_day: str = None):
    """
    Compara un selfie con fotos de una carpeta específica usando clustering optimizado.
    
    OPTIMIZACIÓN: Si existen clusters pre-calculados, busca solo en los clusters más relevantes.
    
    Args:
        selfie: Archivo de selfie
        folder_name: Nombre real de la carpeta (ej: "SANTA SURF PROCENTER")
        search_label: Label para mostrar (ej: "SANTA SURF PROCENTER/2026-01-22")
        photos_in_folder: Lista de rutas de fotos
        search_day: Día específico (opcional)
    """
    try:
        # Validar que el selfie sea una imagen
        if not validate_image_file(selfie):
            raise HTTPException(
                status_code=400,
                detail="El selfie debe ser una imagen válida (JPG, PNG, GIF, BMP)"
            )
        
        # Leer el selfie UNA SOLA VEZ
        selfie_array = await read_image_to_array(selfie)
        
        # Detectar rostro en el selfie (usar modo permisivo para selfies)
        if not detect_face(selfie_array, is_selfie=True):
            raise HTTPException(
                status_code=400,
                detail="No se detectó un rostro en el selfie. Por favor, asegúrate de que la imagen contenga un rostro visible."
            )
        
        # Extraer embedding del selfie (con fallbacks para selfies que a veces fallan en imagen original)
        selfie_embedding, _ = extract_face_embedding(selfie_array, is_selfie=True)
        
        if DEBUG_MODE:
            print(f"🔍 Selfie embedding extraído: shape={selfie_embedding.shape}, norm={np.linalg.norm(selfie_embedding):.4f}")
        
        # OPTIMIZACIÓN 1 (preferida): Identidades (grupos/persona) pre-calculadas
        identities_data = clustering_system.load_identities(folder_name, search_day)
        clusters_data = None

        if identities_data is not None:
            if DEBUG_MODE:
                print(f"🎯 Usando búsqueda optimizada por identidades (persona)")
                print(f"👤 Identidades disponibles: {identities_data['metadata'].get('n_identities')}")

            # Encontrar identidades más relevantes (comparando contra centroides)
            relevant_identities = clustering_system.find_relevant_clusters(
                selfie_embedding,
                identities_data["identity_centroids"],
                top_k=3
            )

            photos_to_search = []
            for identity_id in relevant_identities:
                identity_mask = identities_data["identity_labels"] == identity_id
                identity_filenames = [
                    identities_data["filenames"][i] for i, mask in enumerate(identity_mask) if mask
                ]

                for filename in identity_filenames:
                    photo_path = photos_in_folder[0].parent / filename
                    if photo_path.exists():
                        photos_to_search.append(photo_path)

            if DEBUG_MODE:
                print(
                    f"🔍 Buscando en {len(photos_to_search)} fotos de {len(photos_in_folder)} "
                    f"totales ({len(photos_to_search)/len(photos_in_folder)*100:.1f}%)"
                )
                print(f"⚡ Aceleración estimada: {len(photos_in_folder)/max(1, len(photos_to_search)):.1f}x")

        else:
            # OPTIMIZACIÓN 2 (fallback): clusters genéricos
            clusters_data = clustering_system.load_clusters(folder_name, search_day)

            if clusters_data is not None:
                if DEBUG_MODE:
                    print(f"🎯 Usando búsqueda optimizada con clusters")
                    print(f"📊 Clusters disponibles: {clusters_data['metadata']['n_clusters']}")

                relevant_clusters = clustering_system.find_relevant_clusters(
                    selfie_embedding,
                    clusters_data["centroids"],
                    top_k=3
                )

                photos_to_search = []
                for cluster_id in relevant_clusters:
                    cluster_mask = clusters_data["labels"] == cluster_id
                    cluster_filenames = [
                        clusters_data["filenames"][i] for i, mask in enumerate(cluster_mask) if mask
                    ]

                    for filename in cluster_filenames:
                        photo_path = photos_in_folder[0].parent / filename
                        if photo_path.exists():
                            photos_to_search.append(photo_path)

                if DEBUG_MODE:
                    print(
                        f"🔍 Buscando en {len(photos_to_search)} fotos de {len(photos_in_folder)} "
                        f"totales ({len(photos_to_search)/len(photos_in_folder)*100:.1f}%)"
                    )
                    print(f"⚡ Aceleración estimada: {len(photos_in_folder)/max(1, len(photos_to_search)):.1f}x")
            else:
                if DEBUG_MODE:
                    print(f"⚠️ No hay índices pre-calculados, usando búsqueda tradicional")
                    print(f"💡 Tip: Los embeddings se procesarán automáticamente al subir fotos")
                photos_to_search = photos_in_folder
        
        if DEBUG_MODE:
            print(f"🚀 Procesando {len(photos_to_search)} fotos en paralelo con {MAX_WORKERS} workers...")
        
        # Procesar fotos en paralelo
        matches = []
        non_matches = []
        errors = 0
        
        start_time = datetime.now()
        
        # Usar ThreadPoolExecutor para procesamiento paralelo
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Enviar todas las tareas al pool
            future_to_photo = {
                executor.submit(process_single_photo, photo_path, selfie_embedding): photo_path
                for photo_path in photos_to_search
            }
            
            # Procesar resultados a medida que se completan
            completed = 0
            for future in as_completed(future_to_photo):
                completed += 1
                
                # Mostrar progreso cada 10 fotos
                if DEBUG_MODE and completed % 10 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"⏳ Progreso: {completed}/{len(photos_to_search)} fotos ({rate:.1f} fotos/seg)")
                
                try:
                    result = future.result()
                    
                    if result.get("error"):
                        errors += 1
                    elif result["is_match"]:
                        matches.append({
                            "file": result["file"],
                            "similarity": result["similarity"]
                        })
                    else:
                        non_matches.append({
                            "file": result["file"],
                            "similarity": result["similarity"],
                            "reason": result.get("reason", "No coincide")
                        })
                        
                except Exception as e:
                    photo_path = future_to_photo[future]
                    print(f"❌ Error procesando resultado de {photo_path.name}: {e}")
                    errors += 1
        
        # Calcular tiempo total
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Calcular estadísticas
        total_photos = len(photos_in_folder)
        photos_searched = len(photos_to_search)
        matches_count = len(matches)
        non_matches_count = len(non_matches)
        match_percentage = (matches_count / total_photos * 100) if total_photos > 0 else 0
        
        # Ordenar matches por similitud descendente
        matches_sorted = sorted(matches, key=lambda x: x["similarity"], reverse=True)
        
        # Imprimir resumen en consola
        if DEBUG_MODE:
            print("\n" + "="*60)
            print(f"📋 RESUMEN DE BÚSQUEDA - {search_label}")
            print("="*60)
            if clusters_data:
                print(f"🎯 Búsqueda optimizada con clusters")
                print(f"📊 Fotos analizadas: {photos_searched}/{total_photos} ({photos_searched/total_photos*100:.1f}%)")
            print(f"⏱️  Tiempo total: {total_time:.2f} segundos")
            print(f"⚡ Velocidad: {photos_searched / total_time:.1f} fotos/segundo")
            print(f"✅ MATCHES ENCONTRADOS: {matches_count}")
            if matches_sorted:
                # Mostrar TODOS los matches (son las fotos que el cliente comprará)
                for i, match in enumerate(matches_sorted, 1):
                    print(f"  {i}. {match['file']} - Similitud: {match['similarity']:.2f}%")
            else:
                print("  (ninguno)")
            print(f"\n❌ NO MATCHES: {non_matches_count}")
            print(f"⚠️  ERRORES: {errors}")
            print(f"📊 Porcentaje de coincidencia: {match_percentage:.2f}%")
            print("="*60 + "\n")
        
        return {
            "status": "success",
            "selfie": selfie.filename,
            "folder": search_label,
            "matches": matches_sorted,
            "non_matches": non_matches,
            "statistics": {
                "total_photos": total_photos,
                "photos_searched": photos_searched,
                "matches_count": matches_count,
                "non_matches_count": non_matches_count,
                "errors_count": errors,
                "match_percentage": round(match_percentage, 2),
                "threshold_used": round((1 - SIMILARITY_THRESHOLD) * 100, 2),
                "processing_time_seconds": round(total_time, 2),
                "photos_per_second": round(photos_searched / total_time, 2) if total_time > 0 else 0,
            "used_identity_groups": identities_data is not None,
            "used_clustering": clusters_data is not None,
            "speedup": round(total_photos / max(1, photos_searched), 2) if (identities_data or clusters_data) else 1.0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en compare_faces_from_folder: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando solicitud: {str(e)}"
        )

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> tuple:
    """
    Calcula la similitud entre embeddings usando distancia coseno.
    
    Retorna:
        tuple: (similitud_porcentaje, distancia_coseno)
        - similitud: 0-100 (100 = idénticos, 0 = diferentes)
        - distancia: 0-2 (0 = idénticos, 2 = opuestos)
    """
    from scipy.spatial.distance import cosine
    import numpy as np
    
    # Asegurar que los embeddings sean 1-D
    embedding1 = np.array(embedding1).flatten()
    embedding2 = np.array(embedding2).flatten()
    
    # Validar que tengan la misma dimensión
    if embedding1.shape != embedding2.shape:
        if DEBUG_MODE:
            print(f"⚠️ Embeddings con formas diferentes: {embedding1.shape} vs {embedding2.shape}")
        min_dim = min(len(embedding1), len(embedding2))
        embedding1 = embedding1[:min_dim]
        embedding2 = embedding2[:min_dim]
    
    # Normalizar
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 > 0:
        embedding1 = embedding1 / norm1
    if norm2 > 0:
        embedding2 = embedding2 / norm2
    
    # Calcular distancia coseno
    cosine_distance = cosine(embedding1, embedding2)
    
    # Convertir a similitud (0-100)
    similarity = (1 - cosine_distance) * 100
    
    # Asegurar que esté en el rango [0, 100]
    similarity = max(0, min(100, similarity))
    
    # Convertir a float nativo de Python (no numpy.float32)
    return float(round(similarity, 2)), float(round(cosine_distance, 4))

def process_single_photo(photo_path: Path, selfie_embedding: np.ndarray) -> dict:
    """
    Procesa una sola foto y la compara con el selfie.
    Esta función se ejecuta en paralelo.
    
    Args:
        photo_path: Ruta de la foto a procesar
        selfie_embedding: Embedding del selfie para comparar
    
    Returns:
        dict con resultado de la comparación
    """
    try:
        # Leer foto desde disco
        photo_array = read_image_from_path(photo_path)
        
        # Detectar rostro
        if not detect_face(photo_array):
            return {
                "file": photo_path.name,
                "similarity": 0,
                "reason": "No se detectó rostro",
                "is_match": False
            }
        
        # Extraer embedding
        photo_embedding, _ = extract_face_embedding(photo_array)
        
        # Calcular similitud
        similarity, distance = calculate_similarity(
            selfie_embedding, 
            photo_embedding
        )
        
        # Determinar si es match
        is_match = distance <= SIMILARITY_THRESHOLD
        
        result = {
            "file": photo_path.name,
            "similarity": similarity,
            "distance": distance,
            "is_match": is_match
        }
        
        if not is_match:
            min_similarity_percent = (1 - SIMILARITY_THRESHOLD) * 100
            result["reason"] = f"Similitud {similarity:.2f}% por debajo del umbral {min_similarity_percent:.2f}%"
        
        if DEBUG_MODE:
            status = "✅ MATCH" if is_match else "❌ NO MATCH"
            print(f"{status}: {photo_path.name} (similitud: {similarity:.2f}%)")
        
        return result
        
    except Exception as e:
        print(f"❌ Error procesando {photo_path.name}: {e}")
        return {
            "file": photo_path.name,
            "similarity": 0,
            "reason": f"Error: {str(e)}",
            "is_match": False,
            "error": True
        }

@app.post("/compare-faces-folder")
async def compare_faces_folder(
    selfie: UploadFile = File(...),
    search_folder: str = Query(...),
    search_day: str = Query(None)
):
    """
    Busca un selfie en una carpeta específica y opcionalmente en un día específico
    
    - **selfie**: Imagen del rostro a comparar
    - **search_folder**: Nombre de la carpeta para buscar (ej: 'Surf')
    - **search_day**: (Opcional) Día específico para buscar (ej: '2026-01-13')
    """
    try:
        folder_path = STORAGE_DIR / search_folder
        
        if not folder_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"La carpeta '{search_folder}' no existe"
            )
        
        # Si se especifica un día, buscar en la subcarpeta del día
        if search_day:
            day_folder_path = folder_path / search_day
            if not day_folder_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"El día '{search_day}' no existe en la carpeta '{search_folder}'"
                )
            search_path = day_folder_path
            search_label = f"{search_folder}/{search_day}"
        else:
            search_path = folder_path
            search_label = search_folder
        
        # Obtener todas las fotos de la carpeta/día
        photos_in_folder = [f for f in search_path.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}]
        
        if len(photos_in_folder) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No hay fotos en '{search_label}'"
            )
        
        # Procesar fotos desde la carpeta/día - PASAR search_folder (no search_label)
        return await compare_faces_from_folder(selfie, search_folder, search_label, photos_in_folder, search_day)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en compare_faces_folder: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando solicitud: {str(e)}"
        )

@app.post("/compare-faces")
async def compare_faces(
    selfie: UploadFile = File(...),
    photos: List[UploadFile] = File(...)
):
    """
    Compara un selfie con múltiples fotos (4-10 imágenes)
    
    - **selfie**: Imagen del rostro a comparar
    - **photos**: Lista de imágenes para comparar (4-10 imágenes)
    """
    try:
        # Validar cantidad de fotos cargadas
        if len(photos) < 4 or len(photos) > 10:
            raise HTTPException(
                status_code=400,
                detail="Debe proporcionar entre 4 y 10 fotos"
            )
        
        # Validar que el selfie sea una imagen
        if not validate_image_file(selfie):
            raise HTTPException(
                status_code=400,
                detail="El selfie debe ser una imagen válida (JPG, PNG, GIF, BMP)"
            )
        
        # Leer el selfie
        selfie_array = await read_image_to_array(selfie)
        
        # Detectar rostro en el selfie
        if not detect_face(selfie_array):
            raise HTTPException(
                status_code=400,
                detail="No se detectó un rostro en el selfie. Por favor, asegúrate de que la imagen contenga un rostro visible."
            )
        
        # Extraer embedding del selfie (con fallbacks para selfies)
        selfie_embedding, _ = extract_face_embedding(selfie_array, is_selfie=True)
        
        if DEBUG_MODE:
            print(f"🔍 Selfie embedding extraído: shape={selfie_embedding.shape}, norm={np.linalg.norm(selfie_embedding):.4f}")
        
        # Procesar fotos cargadas
        matches = []
        non_matches = []
        errors = 0
        
        for photo in photos:
            try:
                # Validar que sea una imagen
                if not validate_image_file(photo):
                    errors += 1
                    continue
                
                # Leer foto
                photo_array = await read_image_to_array(photo)
                
                # Detectar rostro
                if not detect_face(photo_array):
                    non_matches.append({
                        "file": photo.filename,
                        "similarity": 0,
                        "reason": "No se detectó rostro"
                    })
                    continue
                
                # Extraer embedding
                photo_embedding, _ = extract_face_embedding(photo_array)
                
                # Calcular similitud
                similarity = calculate_similarity(
                    selfie_embedding, 
                    photo_embedding
                )
                
                # Clasificar como match o no-match
                # Convertir similitud a distancia coseno para comparar
                distance = 1 - (similarity / 100)
                
                # Umbral muy permisivo: aceptar cualquier similitud >= 30%
                min_similarity_percent = (1 - SIMILARITY_THRESHOLD) * 100
                
                if DEBUG_MODE:
                    print(f"📊 {photo.filename}: similitud={similarity:.2f}%, distancia={distance:.4f}, umbral={SIMILARITY_THRESHOLD}")
                
                # Aceptar solo si la distancia es <= umbral
                if distance <= SIMILARITY_THRESHOLD:
                    matches.append({
                        "file": photo.filename,
                        "similarity": similarity
                    })
                    if DEBUG_MODE:
                        print(f"✅ MATCH: {photo.filename} (similitud: {similarity:.2f}%)")
                else:
                    non_matches.append({
                        "file": photo.filename,
                        "similarity": similarity,
                        "reason": f"Similitud {similarity:.2f}% por debajo del umbral {min_similarity_percent:.2f}%"
                    })
                    if DEBUG_MODE:
                        print(f"❌ NO MATCH: {photo.filename} (similitud: {similarity:.2f}%)")
                    
            except Exception as e:
                print(f"Error procesando {photo.filename}: {e}")
                errors += 1
                continue
        
        # Calcular estadísticas
        total_photos = len(photos)
        matches_count = len(matches)
        non_matches_count = len(non_matches)
        match_percentage = (matches_count / total_photos * 100) if total_photos > 0 else 0
        
        # Ordenar matches por similitud descendente
        matches_sorted = sorted(matches, key=lambda x: x["similarity"], reverse=True)
        
        # Imprimir resumen en consola
        if DEBUG_MODE:
            print("\n" + "="*60)
            print(f"📋 RESUMEN DE BÚSQUEDA")
            print("="*60)
            print(f"✅ MATCHES ENCONTRADOS: {matches_count}")
            if matches_sorted:
                for i, match in enumerate(matches_sorted, 1):
                    print(f"  {i}. {match['file']} - Similitud: {match['similarity']:.2f}%")
            else:
                print("  (ninguno)")
            print(f"\n❌ NO MATCHES: {non_matches_count}")
            print(f"📊 Porcentaje de coincidencia: {match_percentage:.2f}%")
            print("="*60 + "\n")
        
        return {
            "status": "success",
            "selfie": selfie.filename,
            "matches": matches_sorted,
            "non_matches": non_matches,
            "statistics": {
                "total_photos": total_photos,
                "matches_count": matches_count,
                "non_matches_count": non_matches_count,
                "errors_count": errors,
                "match_percentage": round(match_percentage, 2),
                "threshold_used": round((1 - SIMILARITY_THRESHOLD) * 100, 2)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en compare_faces: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando solicitud: {str(e)}"
        )

# ==================== ENDPOINTS DE GESTIÓN DE CARPETAS ====================

@app.get("/folders/list")
async def list_folders():
    """Lista todas las carpetas disponibles"""
    try:
        storage_path = STORAGE_DIR
        folders = []
        
        for folder in storage_path.iterdir():
            if folder.is_dir():
                metadata_path = folder / "metadata.json"
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                # Contar fotos (excluyendo cover.jpg)
                photo_count = len([f for f in folder.iterdir() 
                                  if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'} 
                                  and not f.name.startswith('cover')])
                
                # Verificar si existe portada
                cover_image = metadata.get("cover_image")
                if not cover_image:
                    # Buscar archivo de portada físicamente
                    cover_path = folder / "cover.jpg"
                    if cover_path.exists():
                        cover_image = "cover.jpg"
                
                folders.append({
                    "name": folder.name,
                    "created_at": metadata.get("created_at", datetime.now().isoformat()),
                    "photo_count": photo_count,
                    "cover_image": cover_image
                })
        
        return {
            "status": "success",
            "folders": sorted(folders, key=lambda x: x["created_at"], reverse=True)
        }
    except Exception as e:
        print(f"Error listando carpetas: {e}")
        raise HTTPException(status_code=500, detail=f"Error listando carpetas: {str(e)}")

@app.get("/folders")
async def get_folders():
    """Lista todas las carpetas disponibles (alias de /folders/list)"""
    return await list_folders()

@app.post("/folders/create")
async def create_folder(folder_name: str = Query(...)):
    """Crea una nueva carpeta para almacenar fotos"""
    try:
        # Validar nombre de carpeta
        if not folder_name or len(folder_name.strip()) == 0:
            raise HTTPException(status_code=400, detail="El nombre de la carpeta no puede estar vacío")
        
        # Sanitizar nombre
        folder_name = folder_name.strip().replace("/", "_").replace("\\", "_")
        
        folder_path = STORAGE_DIR / folder_name
        
        # Verificar si ya existe
        if folder_path.exists():
            raise HTTPException(status_code=400, detail="La carpeta ya existe")
        
        # Crear carpeta
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Crear archivo de metadata
        metadata = {
            "name": folder_name,
            "created_at": datetime.now().isoformat(),
            "cover_image": None,
            "days": [],
            "photos": []
        }
        
        metadata_path = folder_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Carpeta '{folder_name}' creada exitosamente",
            "folder_name": folder_name
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creando carpeta: {e}")
        raise HTTPException(status_code=500, detail=f"Error creando carpeta: {str(e)}")

@app.post("/folders/set-cover")
async def set_folder_cover(
    folder_name: str = Query(...),
    cover_image: UploadFile = File(...)
):
    """Establece la imagen de portada de una carpeta"""
    try:
        folder_path = STORAGE_DIR / folder_name
        
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail="La carpeta no existe")
        
        # Validar que sea una imagen
        if not validate_image_file(cover_image):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen válida")
        
        # IMPORTANTE: Eliminar todas las portadas antiguas antes de guardar la nueva
        # Esto evita conflictos con archivos antiguos como cover_7N5A1557.JPG
        for old_cover in folder_path.glob("cover*"):
            if old_cover.is_file():
                try:
                    old_cover.unlink()
                    print(f"🗑️ Portada antigua eliminada: {old_cover.name}")
                except Exception as e:
                    print(f"⚠️ Error eliminando portada antigua {old_cover.name}: {e}")
        
        # Guardar imagen de portada con nombre fijo
        cover_filename = "cover.jpg"
        cover_path = folder_path / cover_filename
        
        contents = await cover_image.read()
        with open(cover_path, 'wb') as f:
            f.write(contents)
        
        # Actualizar metadata
        metadata_path = folder_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "name": folder_name,
                "created_at": datetime.now().isoformat(),
                "days": [],
                "photos": []
            }
        
        metadata["cover_image"] = cover_filename
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Portada guardada: {cover_path}")
        
        return {
            "status": "success",
            "message": "Imagen de portada actualizada",
            "cover_image": cover_filename
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error estableciendo portada: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/folders/cover/{folder_name}")
async def get_folder_cover(folder_name: str):
    """Obtiene la imagen de portada de una carpeta"""
    try:
        folder_path = STORAGE_DIR / folder_name
        
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail="La carpeta no existe")
        
        # Buscar archivo de portada
        cover_path = folder_path / "cover.jpg"
        
        if not cover_path.exists():
            # Intentar con otros nombres posibles
            for ext in ['.png', '.jpeg', '.gif']:
                alt_path = folder_path / f"cover{ext}"
                if alt_path.exists():
                    cover_path = alt_path
                    break
            else:
                raise HTTPException(status_code=404, detail="No hay portada asignada")
        
        # Leer y devolver la imagen
        with open(cover_path, 'rb') as f:
            content = f.read()
        
        # Determinar content type
        content_type = "image/jpeg"
        if cover_path.suffix.lower() == '.png':
            content_type = "image/png"
        elif cover_path.suffix.lower() == '.gif':
            content_type = "image/gif"
        
        return Response(content=content, media_type=content_type)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error obteniendo portada: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/folders/set-day-cover")
async def set_day_cover(
    folder_name: str = Query(...),
    day_date: str = Query(...),
    cover_image: UploadFile = File(...)
):
    """Establece la imagen de portada de un día específico"""
    try:
        folder_path = STORAGE_DIR / folder_name
        day_folder_path = folder_path / day_date
        
        if not day_folder_path.exists():
            raise HTTPException(status_code=404, detail="El día no existe")
        
        # Validar que sea una imagen
        if not validate_image_file(cover_image):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen válida")
        
        # Guardar imagen de portada con nombre fijo
        cover_filename = "cover.jpg"
        cover_path = day_folder_path / cover_filename
        
        contents = await cover_image.read()
        with open(cover_path, 'wb') as f:
            f.write(contents)
        
        print(f"✅ Portada del día guardada: {cover_path}")
        
        return {
            "status": "success",
            "message": "Imagen de portada del día actualizada",
            "cover_image": cover_filename
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error estableciendo portada del día: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/folders/{folder_name}/day-cover/{day_date}")
async def get_day_cover(folder_name: str, day_date: str):
    """Obtiene la imagen de portada de un día específico"""
    try:
        folder_path = STORAGE_DIR / folder_name
        day_folder_path = folder_path / day_date
        
        if not day_folder_path.exists():
            raise HTTPException(status_code=404, detail="El día no existe")
        
        # Buscar archivo de portada
        cover_path = day_folder_path / "cover.jpg"
        
        if not cover_path.exists():
            # Intentar con otros nombres posibles
            for ext in ['.png', '.jpeg', '.gif']:
                alt_path = day_folder_path / f"cover{ext}"
                if alt_path.exists():
                    cover_path = alt_path
                    break
            else:
                raise HTTPException(status_code=404, detail="No hay portada asignada")
        
        # Leer y devolver la imagen
        with open(cover_path, 'rb') as f:
            content = f.read()
        
        # Determinar content type
        content_type = "image/jpeg"
        if cover_path.suffix.lower() == '.png':
            content_type = "image/png"
        elif cover_path.suffix.lower() == '.gif':
            content_type = "image/gif"
        
        return Response(content=content, media_type=content_type)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error obteniendo portada del día: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/folders/create-day")
async def create_day_folder(
    folder_name: str = Query(...),
    day_date: str = Query(...)
):
    """Crea una subcarpeta para un día específico dentro de una carpeta"""
    try:
        folder_path = STORAGE_DIR / folder_name
        
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail="La carpeta no existe")
        
        # Validar formato de fecha
        try:
            datetime.fromisoformat(day_date)
        except:
            raise HTTPException(status_code=400, detail="Formato de fecha inválido. Use YYYY-MM-DD")
        
        # Crear subcarpeta del día
        day_folder_path = folder_path / day_date
        day_folder_path.mkdir(parents=True, exist_ok=True)
        
        # Actualizar metadata de la carpeta principal
        metadata_path = folder_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "name": folder_name,
                "created_at": datetime.now().isoformat(),
                "cover_image": None,
                "days": [],
                "photos": []
            }
        
        if "days" not in metadata:
            metadata["days"] = []
        
        if day_date not in metadata["days"]:
            metadata["days"].append(day_date)
            metadata["days"].sort(reverse=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Día '{day_date}' creado en carpeta '{folder_name}'",
            "day_date": day_date
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creando día: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/folders/{folder_name}/days")
async def get_folder_days(folder_name: str):
    """Obtiene todos los días disponibles en una carpeta"""
    try:
        folder_path = STORAGE_DIR / folder_name
        
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail="La carpeta no existe")
        
        # Obtener subcarpetas (días)
        days = []
        for item in folder_path.iterdir():
            if item.is_dir() and item.name != "__pycache__":
                # Verificar si es una fecha válida
                try:
                    datetime.fromisoformat(item.name)
                    # Contar fotos en el día (excluyendo cover.jpg)
                    photo_count = len([f for f in item.iterdir()
                                       if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
                                       and not f.name.startswith('cover')])
                    days.append({
                        "date": item.name,
                        "photo_count": photo_count
                    })
                except:
                    continue
        
        # Ordenar por fecha descendente
        days.sort(key=lambda x: x["date"], reverse=True)
        
        return {
            "status": "success",
            "folder": folder_name,
            "days": days
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error obteniendo días: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.delete("/folders/delete")
async def delete_folder(folder_name: str = Query(...)):
    """Elimina una carpeta y todas sus fotos"""
    try:
        folder_path = STORAGE_DIR / folder_name
        
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail="La carpeta no existe")
        
        # Eliminar carpeta y su contenido
        shutil.rmtree(folder_path)
        
        return {
            "status": "success",
            "message": f"Carpeta '{folder_name}' eliminada exitosamente"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error eliminando carpeta: {e}")
        raise HTTPException(status_code=500, detail=f"Error eliminando carpeta: {str(e)}")

# ==================== ENDPOINTS DE GESTIÓN DE FOTOS ====================

@app.post("/photos/upload")
async def upload_photos(
    folder_name: str = Query(...),
    photos: List[UploadFile] = File(...),
    day: str = Query(None),
    index: bool = Query(True)
):
    """
    Sube fotos a una carpeta específica y procesa embeddings automáticamente.
    
    Args:
        folder_name: Nombre de la carpeta
        photos: Lista de fotos a subir
        day: Día específico (opcional)
    """
    try:
        folder_path = STORAGE_DIR / folder_name
        
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail="La carpeta no existe")
        
        if len(photos) == 0:
            raise HTTPException(status_code=400, detail="Debe proporcionar al menos una foto")
        
        # Determinar ruta de destino
        if day:
            # Crear subcarpeta del día si no existe
            day_folder_path = folder_path / day
            day_folder_path.mkdir(parents=True, exist_ok=True)
            upload_path = day_folder_path
            label = f"{folder_name}/{day}"
        else:
            upload_path = folder_path
            label = folder_name
        
        uploaded_photos = []
        errors = []
        uploaded_filenames: List[str] = []
        
        # Paso 1: Subir fotos
        print(f"\n{'='*60}")
        print(f"📤 SUBIENDO FOTOS - {label}")
        print(f"{'='*60}")
        
        for photo in photos:
            try:
                # Validar que sea una imagen
                if not validate_image_file(photo):
                    errors.append(f"{photo.filename}: Formato no válido")
                    continue

                # Guardar foto en streaming (más rápido y sin cargar todo en RAM)
                photo_path = upload_path / photo.filename
                with open(photo_path, 'wb') as f:
                    total_written = 0
                    while True:
                        chunk = await photo.read(1024 * 1024)  # 1MB
                        if not chunk:
                            break
                        total_written += len(chunk)
                        if total_written > MAX_FILE_SIZE:
                            try:
                                f.close()
                                photo_path.unlink(missing_ok=True)
                            except Exception:
                                pass
                            raise HTTPException(
                                status_code=400,
                                detail=f"Archivo demasiado grande. Máximo: {MAX_FILE_SIZE / 1024 / 1024}MB"
                            )
                        f.write(chunk)
                
                uploaded_photos.append({
                    "filename": photo.filename,
                    "size": total_written,
                    "status": "success"
                })
                uploaded_filenames.append(photo.filename)
                
                print(f"✅ Foto guardada: {photo.filename} ({total_written / 1024:.2f} KB)")
                
            except HTTPException as e:
                errors.append(f"{photo.filename}: {e.detail}")
            except Exception as e:
                errors.append(f"{photo.filename}: {str(e)}")
        
        # Paso 2: Indexado/embeddings (NO bloquear la subida)
        # Se ejecuta en segundo plano para que la subida sea rápida.
        embeddings_result = None
        if index and len(uploaded_photos) > 0:
            try:
                print(f"\n🧠 Encolando indexado (embeddings + grupos) en segundo plano...")

                def _run_index_job():
                    try:
                        clustering_system.process_incremental(
                            folder_name=folder_name,
                            day=day,
                            new_filenames=uploaded_filenames
                        )
                    except Exception as e:
                        print(f"❌ Error en job de indexado: {e}")

                INDEX_EXECUTOR.submit(_run_index_job)

                embeddings_result = {
                    "status": "queued",
                    "message": "Indexado encolado en segundo plano",
                    "new_photos_queued": len(uploaded_filenames),
                }

            except Exception as e:
                print(f"⚠️ Error encolando embeddings (no crítico): {e}")
                embeddings_result = {"status": "error", "message": str(e)}
        
        response = {
            "status": "success",
            "uploaded": len(uploaded_photos),
            "photos": uploaded_photos,
            "errors": errors,
            "message": f"{len(uploaded_photos)} fotos subidas correctamente"
        }
        
        # Agregar información de embeddings si se procesaron
        if embeddings_result:
            response["embeddings"] = {
                "status": embeddings_result["status"],
                "message": embeddings_result.get("message", ""),
                "n_clusters": embeddings_result.get("n_clusters"),
                "n_identities": embeddings_result.get("n_identities"),
                "total_photos_indexed": embeddings_result.get("total_photos"),
                "new_photos_processed": embeddings_result.get("new_photos_processed"),
                "failed_new_photos": embeddings_result.get("failed_new_photos"),
                "processing_time": embeddings_result.get("total_time"),
                "new_photos_queued": embeddings_result.get("new_photos_queued"),
            }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error subiendo fotos: {e}")
        raise HTTPException(status_code=500, detail=f"Error subiendo fotos: {str(e)}")


@app.post("/indexing/start")
async def start_indexing(
    folder_name: str = Query(...),
    day: str = Query(None),
):
    """
    Encola un re-indexado para una carpeta/día.

    Útil para el panel admin: subir en lotes rápido (index=false) y luego llamar a este endpoint una sola vez.
    """
    label = f"{folder_name}/{day}" if day else folder_name

    try:
        print(f"🧠 Encolando re-indexado completo para {label}...")

        def _run_full_index():
            try:
                clustering_system.process_folder(folder_name=folder_name, day=day, force=True)
            except Exception as e:
                print(f"❌ Error en re-indexado {label}: {e}")

        INDEX_EXECUTOR.submit(_run_full_index)

        return {
            "status": "queued",
            "message": f"Re-indexado encolado para {label}",
            "folder": folder_name,
            "day": day,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encolando indexado: {str(e)}")

@app.get("/photos/list")
async def list_photos(folder_name: str = Query(...)):
    """Lista todas las fotos en una carpeta"""
    try:
        folder_path = STORAGE_DIR / folder_name
        
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail="La carpeta no existe")
        
        photos = []
        
        for file in folder_path.iterdir():
            if file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}:
                photos.append({
                    "filename": file.name,
                    "size": file.stat().st_size,
                    "created_at": datetime.fromtimestamp(file.stat().st_ctime).isoformat()
                })
        
        return {
            "status": "success",
            "folder": folder_name,
            "photos": sorted(photos, key=lambda x: x["created_at"], reverse=True)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error listando fotos: {e}")
        raise HTTPException(status_code=500, detail=f"Error listando fotos: {str(e)}")

@app.get("/photos/preview")
async def get_photo_preview(
    folder_name: str = Query(...),
    filename: str = Query(...),
    watermark: bool = Query(True),
    day: str = Query(None)
):
    """Obtiene una foto con marca de agua para previsualización"""
    try:
        folder_path = STORAGE_DIR / folder_name
        
        # Si se especifica un día, buscar en la subcarpeta del día
        if day:
            photo_path = folder_path / day / filename
        else:
            photo_path = folder_path / filename
        
        print(f"🔍 Buscando foto: {photo_path}")
        print(f"🔍 Path existe: {photo_path.exists()}")
        print(f"🔍 Path absoluto: {photo_path.absolute()}")
        
        if not photo_path.exists():
            raise HTTPException(status_code=404, detail=f"La foto no existe: {photo_path}")
        
        # Leer la imagen
        image = Image.open(photo_path)
        
        # Agregar marca de agua si se solicita
        if watermark:
            # Convertir a RGB primero para simplificar
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Crear una copia para trabajar
            watermarked = image.copy()
            draw = ImageDraw.Draw(watermarked)
            
            # Texto de la marca de agua
            watermark_text = "SURFSHOT"
            
            # Obtener dimensiones de la imagen
            width, height = watermarked.size
            
            # Intentar cargar una fuente, si no está disponible usar la predeterminada
            font_size = max(width, height) // 15
            font = None
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            # Obtener dimensiones del texto (método compatible)
            try:
                # PIL 10.0.0+ usa textbbox
                if hasattr(draw, 'textbbox'):
                    bbox = draw.textbbox((0, 0), watermark_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                else:
                    # Versiones antiguas de PIL usan textsize
                    text_width, text_height = draw.textsize(watermark_text, font=font)
            except Exception as e:
                # Fallback: estimar tamaño del texto
                text_width = len(watermark_text) * font_size // 2
                text_height = font_size
            
            # Calcular posición (centro de la imagen)
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            # Dibujar sombra del texto (para mejor visibilidad)
            shadow_offset = 2
            draw.text((x + shadow_offset, y + shadow_offset), watermark_text, 
                     fill=(0, 0, 0), font=font)
            
            # Dibujar el texto principal (blanco)
            draw.text((x, y), watermark_text, 
                     fill=(255, 255, 255), font=font)
            
            # Agregar marca de agua adicional en la esquina inferior derecha
            corner_text = "PREVIEW"
            corner_font_size = max(width, height) // 25
            corner_font = None
            try:
                corner_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", corner_font_size)
            except:
                try:
                    corner_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", corner_font_size)
                except:
                    corner_font = ImageFont.load_default()
            
            try:
                if hasattr(draw, 'textbbox'):
                    corner_bbox = draw.textbbox((0, 0), corner_text, font=corner_font)
                    corner_text_width = corner_bbox[2] - corner_bbox[0]
                    corner_text_height = corner_bbox[3] - corner_bbox[1]
                else:
                    corner_text_width, corner_text_height = draw.textsize(corner_text, font=corner_font)
            except Exception as e:
                corner_text_width = len(corner_text) * corner_font_size // 2
                corner_text_height = corner_font_size
            
            corner_x = width - corner_text_width - 20
            corner_y = height - corner_text_height - 20
            
            # Fondo semi-transparente para el texto de la esquina (usando RGB con alpha simulado)
            draw.rectangle([corner_x - 10, corner_y - 5, corner_x + corner_text_width + 10, corner_y + corner_text_height + 5],
                          fill=(0, 0, 0))
            
            draw.text((corner_x, corner_y), corner_text, 
                     fill=(255, 255, 255), font=corner_font)
            
            output_image = watermarked
        else:
            # Convertir a RGB si es necesario para JPEG
            if image.mode != 'RGB':
                output_image = image.convert('RGB')
            else:
                output_image = image
        
        # Convertir a bytes
        img_byte_arr = io.BytesIO()
        
        # Determinar formato y content type
        if filename.lower().endswith('.png'):
            if output_image.mode != 'RGB':
                output_image = output_image.convert('RGB')
            output_image.save(img_byte_arr, format='PNG')
            content_type = "image/png"
        else:
            if output_image.mode != 'RGB':
                output_image = output_image.convert('RGB')
            output_image.save(img_byte_arr, format='JPEG', quality=85)
            content_type = "image/jpeg"
        
        img_byte_arr.seek(0)
        
        return Response(content=img_byte_arr.read(), media_type=content_type)
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error obteniendo preview: {e}")
        print(traceback.format_exc())
        # Si falla el preview, intentar devolver la imagen original
        try:
            photo_path = STORAGE_DIR / folder_name / filename
            if photo_path.exists():
                with open(photo_path, 'rb') as f:
                    content = f.read()
                content_type = "image/jpeg"
                if filename.lower().endswith('.png'):
                    content_type = "image/png"
                return Response(content=content, media_type=content_type)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error obteniendo preview: {str(e)}")

@app.get("/photos/view")
async def get_photo_view(
    folder_name: str = Query(...),
    filename: str = Query(...)
):
    """Obtiene una foto original sin marca de agua (fallback)"""
    try:
        folder_path = STORAGE_DIR / folder_name
        photo_path = folder_path / filename
        
        if not photo_path.exists():
            raise HTTPException(status_code=404, detail="La foto no existe")
        
        # Leer y devolver la imagen original
        with open(photo_path, 'rb') as f:
            content = f.read()
        
        content_type = "image/jpeg"
        if filename.lower().endswith('.png'):
            content_type = "image/png"
        elif filename.lower().endswith('.gif'):
            content_type = "image/gif"
        
        return Response(content=content, media_type=content_type)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error obteniendo foto: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo foto: {str(e)}")

@app.delete("/photos/delete")
async def delete_photo(folder_name: str = Query(...), filename: str = Query(...)):
    """Elimina una foto de una carpeta"""
    try:
        folder_path = STORAGE_DIR / folder_name
        photo_path = folder_path / filename
        
        if not photo_path.exists():
            raise HTTPException(status_code=404, detail="La foto no existe")
        
        # Eliminar foto
        photo_path.unlink()
        
        # Eliminar embedding si existe
        embedding_filename = filename.rsplit('.', 1)[0] + '.npy'
        embedding_path = folder_path / embedding_filename
        if embedding_path.exists():
            embedding_path.unlink()
        
        return {
            "status": "success",
            "message": f"Foto '{filename}' eliminada exitosamente"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error eliminando foto: {e}")
        raise HTTPException(status_code=500, detail=f"Error eliminando foto: {str(e)}")

# ==================== ENDPOINT DE BÚSQUEDA ====================

@app.post("/search/similar")
async def search_similar(selfie: UploadFile = File(...)):
    """Busca fotos similares en TODAS las carpetas"""
    try:
        # Validar que el selfie sea una imagen
        if not validate_image_file(selfie):
            raise HTTPException(
                status_code=400,
                detail="El selfie debe ser una imagen válida (JPG, PNG, GIF, BMP)"
            )
        
        # Leer el selfie
        selfie_array = await read_image_to_array(selfie)
        
        # Detectar rostro en el selfie (usar modo permisivo para selfies)
        if not detect_face(selfie_array, is_selfie=True):
            raise HTTPException(
                status_code=400,
                detail="No se detectó un rostro en el selfie"
            )
        
        # Extraer embedding del selfie (con fallbacks para selfies)
        selfie_embedding, _ = extract_face_embedding(selfie_array, is_selfie=True)
        
        # Buscar fotos similares en TODAS las carpetas
        matches = []
        non_matches = []
        
        storage_path = STORAGE_DIR
        
        # Iterar sobre todas las carpetas
        for folder in storage_path.iterdir():
            if folder.is_dir():
                # Iterar sobre todas las fotos en la carpeta
                for file in folder.iterdir():
                    if file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}:
                        try:
                            # Siempre extraer embedding fresco con InsightFace
                            photo_array = read_image_from_path(file)
                            
                            # Verificar que la imagen tenga un rostro
                            if not detect_face(photo_array):
                                if DEBUG_MODE:
                                    print(f"⚠️ {file.name}: No se detectó rostro, saltando...")
                                continue
                            
                            try:
                                photo_embedding, _ = extract_face_embedding(photo_array)
                            except Exception as e:
                                if DEBUG_MODE:
                                    print(f"❌ {file.name}: Error extrayendo embedding: {e}")
                                    import traceback
                                    traceback.print_exc()
                                continue
                            
                            # Calcular similitud
                            similarity, distance = calculate_similarity(
                                selfie_embedding, 
                                photo_embedding
                            )
                            
                            if DEBUG_MODE:
                                print(f"📊 {file.name}: similitud={similarity:.2f}%, distancia={distance:.4f}")
                            
                            # Aceptar solo si distancia <= umbral (más estricto)
                            if distance <= SIMILARITY_THRESHOLD:
                                matches.append({
                                    "file": file.name,
                                    "folder": folder.name,
                                    "similarity": similarity
                                })
                            else:
                                non_matches.append({
                                    "file": file.name,
                                    "folder": folder.name,
                                    "similarity": similarity
                                })
                                
                        except Exception as e:
                            print(f"Error procesando {file.name}: {e}")
                            continue
        
        # Calcular estadísticas
        total_photos = len(matches) + len(non_matches)
        matches_count = len(matches)
        non_matches_count = len(non_matches)
        match_percentage = (matches_count / total_photos * 100) if total_photos > 0 else 0
        
        return {
            "status": "success",
            "search_scope": "all_folders",
            "matches": sorted(matches, key=lambda x: x["similarity"], reverse=True),
            "non_matches": sorted(non_matches, key=lambda x: x["similarity"], reverse=True),
            "statistics": {
                "total_photos": total_photos,
                "matches_count": matches_count,
                "non_matches_count": non_matches_count,
                "match_percentage": round(match_percentage, 2),
                "threshold_used": round((1 - SIMILARITY_THRESHOLD) * 100, 2)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en search_similar: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando solicitud: {str(e)}"
        )

# ==================== ENDPOINTS DEL MARKETPLACE ====================

@app.get("/marketplace/photos")
async def get_marketplace_photos():
    """Obtiene todas las fotos disponibles en el marketplace"""
    try:
        photos = []
        storage_path = STORAGE_DIR
        
        # Iterar sobre todas las carpetas (escuelas)
        for school_folder in storage_path.iterdir():
            if school_folder.is_dir():
                school_name = school_folder.name
                
                # Iterar sobre todas las fotos en la carpeta
                for file in school_folder.iterdir():
                    if file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}:
                        # Obtener fecha de creación
                        created_at = datetime.fromtimestamp(file.stat().st_ctime)
                        date_str = created_at.strftime("%Y-%m-%d")
                        
                        # Generar URL de preview con marca de agua
                        preview_url = f"/photos/preview?folder_name={school_name}&filename={file.name}&watermark=true"
                        
                        photos.append({
                            "id": f"{school_name}_{file.stem}",
                            "filename": file.name,
                            "school": school_name,
                            "date": date_str,
                            "similarity": None,
                            "thumbnail": preview_url,
                            "image": preview_url
                        })
        
        return {
            "status": "success",
            "photos": photos
        }
    except Exception as e:
        print(f"Error obteniendo fotos del marketplace: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo fotos: {str(e)}"
        )

@app.get("/marketplace/filters")
async def get_marketplace_filters():
    """Obtiene los filtros disponibles (escuelas y días)"""
    try:
        schools = set()
        days = set()
        storage_path = STORAGE_DIR
        
        # Iterar sobre todas las carpetas
        for school_folder in storage_path.iterdir():
            if school_folder.is_dir():
                schools.add(school_folder.name)
                
                # Iterar sobre fotos para obtener fechas
                for file in school_folder.iterdir():
                    if file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}:
                        created_at = datetime.fromtimestamp(file.stat().st_ctime)
                        date_str = created_at.strftime("%Y-%m-%d")
                        days.add(date_str)
        
        return {
            "status": "success",
            "schools": sorted(list(schools)),
            "days": sorted(list(days), reverse=True)
        }
    except Exception as e:
        print(f"Error obteniendo filtros: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo filtros: {str(e)}"
        )

@app.post("/marketplace/search-similar")
async def marketplace_search_similar(selfie: UploadFile = File(...)):
    """Busca fotos similares en el marketplace"""
    try:
        # Validar que el selfie sea una imagen
        if not validate_image_file(selfie):
            raise HTTPException(
                status_code=400,
                detail="El selfie debe ser una imagen válida (JPG, PNG, GIF, BMP)"
            )
        
        # Leer el selfie
        selfie_array = await read_image_to_array(selfie)
        
        # Detectar rostro en el selfie
        if not detect_face(selfie_array):
            raise HTTPException(
                status_code=400,
                detail="No se detectó un rostro en el selfie"
            )
        
        # Extraer embedding del selfie (con fallbacks para selfies)
        selfie_embedding, _ = extract_face_embedding(selfie_array, is_selfie=True)
        
        if DEBUG_MODE:
            print(f"🔍 Selfie embedding extraído: shape={selfie_embedding.shape}, norm={np.linalg.norm(selfie_embedding):.4f}")
        
        # Buscar fotos similares en photos_storage
        photos_with_similarity = []
        storage_path = STORAGE_DIR
        
        if DEBUG_MODE:
            print(f"🔍 Buscando fotos en: {storage_path}")
            print(f"📁 Carpetas encontradas: {[f.name for f in storage_path.iterdir() if f.is_dir()]}")
        
        # Iterar sobre todas las carpetas en photos_storage
        for school_folder in storage_path.iterdir():
            if school_folder.is_dir():
                school_name = school_folder.name
                
                # Iterar sobre todas las fotos
                for file in school_folder.iterdir():
                    if file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}:
                        try:
                            # IMPORTANTE: Siempre extraer embedding fresco con InsightFace
                            # Los embeddings guardados pueden ser del modelo antiguo o tener forma incorrecta
                            photo_array = read_image_from_path(file)
                            
                            # Verificar que la imagen tenga un rostro
                            if not detect_face(photo_array):
                                if DEBUG_MODE:
                                    print(f"⚠️ {file.name}: No se detectó rostro, saltando...")
                                continue
                            
                            try:
                                photo_embedding, _ = extract_face_embedding(photo_array)
                            except Exception as e:
                                if DEBUG_MODE:
                                    print(f"❌ {file.name}: Error extrayendo embedding: {e}")
                                    import traceback
                                    traceback.print_exc()
                                continue
                            
                            # Calcular similitud
                            similarity, distance = calculate_similarity(
                                selfie_embedding, 
                                photo_embedding
                            )
                            
                            if DEBUG_MODE:
                                print(f"📊 {file.name}: similitud={similarity:.2f}%, distancia={distance:.4f}")
                            
                            # Obtener fecha
                            created_at = datetime.fromtimestamp(file.stat().st_ctime)
                            date_str = created_at.strftime("%Y-%m-%d")
                            
                            # Generar URL de preview con marca de agua
                            preview_url = f"/photos/preview?folder_name={school_name}&filename={file.name}&watermark=true"
                            
                            photos_with_similarity.append({
                                "id": f"{school_name}_{file.stem}",
                                "filename": file.name,
                                "school": school_name,
                                "date": date_str,
                                "similarity": round(similarity, 2),
                                "thumbnail": preview_url,
                                "image": preview_url
                            })
                            
                        except Exception as e:
                            print(f"Error procesando {file.name}: {e}")
                            continue
        
        # Ordenar por similitud descendente
        photos_with_similarity.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Contar coincidencias (umbral estricto)
        min_similarity_percent = (1 - SIMILARITY_THRESHOLD) * 100
        matches = [p for p in photos_with_similarity if (1 - (p["similarity"] / 100)) <= SIMILARITY_THRESHOLD]
        matches_count = len(matches)
        
        if DEBUG_MODE:
            print("\n" + "="*60)
            print(f"📋 RESUMEN DE BÚSQUEDA - MARKETPLACE")
            print("="*60)
            print(f"✅ MATCHES ENCONTRADOS: {matches_count}")
            if matches:
                for i, match in enumerate(matches, 1):
                    print(f"  {i}. {match['filename']} ({match['school']}) - Similitud: {match['similarity']:.2f}%")
            else:
                print("  (ninguno)")
            print(f"\n📊 Total de fotos procesadas: {len(photos_with_similarity)}")
            print(f"📊 Umbral utilizado: {min_similarity_percent:.2f}%")
            if len(photos_with_similarity) > 0:
                top5 = [f"{p['filename']}: {p['similarity']:.2f}%" for p in photos_with_similarity[:5]]
                print(f"📊 Top 5 similitudes: {top5}")
            print("="*60 + "\n")
        






        return {
            "status": "success",
            "photos": photos_with_similarity,
            "matches": matches,
            "matches_count": matches_count,
            "total_photos": len(photos_with_similarity),
            "threshold_used": round((1 - SIMILARITY_THRESHOLD) * 100, 2),
            "storage_path": str(storage_path)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en marketplace_search_similar: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando solicitud: {str(e)}"
        )

# ==================== ENDPOINTS DE PAGOS ====================

@app.get("/payments")
async def get_payments():
    """Lista todas las transacciones de pago (alias de /payments/list)"""
    return await list_payments()

@app.get("/payments/list")
async def list_payments():
    """Obtiene todas las transacciones de pago"""
    try:
        # Datos simulados de pagos (en producción, vendrían de una base de datos)
        payments = [
            {
                "id": "TXN001",
                "customer_name": "Juan Pérez",
                "customer_email": "juan@example.com",
                "amount": 29.97,
                "created_at": datetime.now().isoformat(),
                "status": "completed",
                "items_count": 3
            },
            {
                "id": "TXN002",
                "customer_name": "María García",
                "customer_email": "maria@example.com",
                "amount": 19.98,
                "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
                "status": "completed",
                "items_count": 2
            },
            {
                "id": "TXN003",
                "customer_name": "Carlos López",
                "customer_email": "carlos@example.com",
                "amount": 39.96,
                "created_at": (datetime.now() - timedelta(days=2)).isoformat(),
                "status": "completed",
                "items_count": 4
            }
        ]
        
        return {
            "status": "success",
            "payments": payments,
            "total": len(payments)
        }
    except Exception as e:
        print(f"Error listando pagos: {e}")
        raise HTTPException(status_code=500, detail=f"Error listando pagos: {str(e)}")

@app.post("/payments/record")
async def record_payment(
    customer_name: str = Query(...),
    amount: float = Query(...),
    items_count: int = Query(...)
):
    """Registra una nueva transacción de pago"""
    try:
        payment = {
            "id": f"TXN{datetime.now().timestamp()}",
            "customer": customer_name,
            "amount": amount,
            "items_count": items_count,
            "date": datetime.now().isoformat(),
            "status": "completed"
        }
        
        return {
            "status": "success",
            "message": "Pago registrado exitosamente",
            "payment": payment
        }
    except Exception as e:
        print(f"Error registrando pago: {e}")
        raise HTTPException(status_code=500, detail=f"Error registrando pago: {str(e)}")

# ==================== ENDPOINTS DE ESTADÍSTICAS ====================

@app.get("/statistics/dashboard")
async def get_dashboard_statistics():
    """Obtiene estadísticas del dashboard"""
    try:
        storage_path = STORAGE_DIR
        
        # Contar carpetas
        folders = [f for f in storage_path.iterdir() if f.is_dir()]
        total_folders = len(folders)
        
        # Contar fotos
        total_photos = 0
        for folder in folders:
            photos = [f for f in folder.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}]
            total_photos += len(photos)
        
        # Calcular ingresos (simulado)
        total_revenue = total_photos * 9.99 * 0.5
        total_transactions = max(1, total_photos // 3)
        
        return {
            "status": "success",
            "statistics": {
                "total_folders": total_folders,
                "total_photos": total_photos,
                "total_revenue": round(total_revenue, 2),
                "total_transactions": total_transactions,
                "average_photos_per_folder": round(total_photos / max(1, total_folders), 2)
            }
        }
    except Exception as e:
        print(f"Error obteniendo estadísticas: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")

@app.get("/statistics/photos-by-school")
async def get_photos_by_school():
    """Obtiene estadísticas de fotos por escuela"""
    try:
        storage_path = STORAGE_DIR
        photos_by_school = {}
        
        for folder in storage_path.iterdir():
            if folder.is_dir():
                photos = [f for f in folder.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}]
                photos_by_school[folder.name] = len(photos)
        
        return {
            "status": "success",
            "data": photos_by_school
        }
    except Exception as e:
        print(f"Error obteniendo fotos por escuela: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/statistics/revenue-by-month")
async def get_revenue_by_month():
    """Obtiene estadísticas de ingresos por mes"""
    try:
        # Datos simulados
        current_month = datetime.now().strftime("%Y-%m")
        revenue_data = {
            current_month: 1500.00
        }
        
        return {
            "status": "success",
            "data": revenue_data
        }
    except Exception as e:
        print(f"Error obteniendo ingresos por mes: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)