"""
DEMO TÉCNICA DE RECONOCIMIENTO FACIAL
FastAPI + DeepFace + FaceNet + OpenCV
Comparación biométrica de rostros usando embeddings faciales
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(title="Face Recognition Demo", version="1.0.0")

# Configurar CORS para permitir solicitudes desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración
SIMILARITY_THRESHOLD = 0.40  # Umbral de similitud (0-1) - Detecta coincidencias desde 40%
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp"}
MODEL_NAME = "Facenet"  # Modelo de reconocimiento facial


def validate_image_file(file: UploadFile) -> bool:
    """
    Valida que el archivo sea una imagen válida.
    
    Args:
        file: Archivo subido
        
    Returns:
        bool: True si es válido, False en caso contrario
    """
    # Validar extensión
    if not file.filename:
        return False
    
    extension = file.filename.split(".")[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return False
    
    return True


def read_image_to_array(file: UploadFile) -> np.ndarray:
    """
    Lee un archivo de imagen y lo convierte a array numpy.
    
    Args:
        file: Archivo de imagen subido
        
    Returns:
        np.ndarray: Imagen como array numpy
        
    Raises:
        ValueError: Si la imagen no puede ser leída
    """
    try:
        # Leer contenido del archivo
        contents = file.file.read()
        
        # Validar tamaño
        if len(contents) > MAX_FILE_SIZE:
            raise ValueError(f"Archivo demasiado grande. Máximo: {MAX_FILE_SIZE / 1024 / 1024}MB")
        
        # Convertir a array numpy
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decodificar imagen
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen")
        
        return img
    
    except Exception as e:
        logger.error(f"Error al leer imagen: {str(e)}")
        raise ValueError(f"Error al procesar imagen: {str(e)}")


def detect_face(image: np.ndarray) -> bool:
    """
    Detecta si hay un rostro en la imagen usando DeepFace.
    DeepFace es más flexible que OpenCV Cascade Classifier.
    
    Args:
        image: Imagen como array numpy
        
    Returns:
        bool: True si se detecta rostro, False en caso contrario
    """
    try:
        # Usar DeepFace para detectar rostros (más flexible que OpenCV)
        # Si enforce_detection=False, intentará extraer embedding incluso sin detección clara
        embedding_objs = DeepFace.represent(
            img_path=image,
            model_name=MODEL_NAME,
            enforce_detection=False  # Más flexible
        )
        
        return len(embedding_objs) > 0
    
    except Exception as e:
        logger.error(f"Error en detección de rostro: {str(e)}")
        return False


def extract_face_embedding(image: np.ndarray) -> np.ndarray:
    """
    Extrae el embedding facial usando DeepFace + FaceNet.
    
    Args:
        image: Imagen como array numpy
        
    Returns:
        np.ndarray: Vector de embedding facial (128 dimensiones)
        
    Raises:
        ValueError: Si no se puede extraer el embedding
    """
    try:
        # Extraer embedding usando DeepFace directamente con el array numpy
        embedding_objs = DeepFace.represent(
            img_path=image,
            model_name=MODEL_NAME,
            enforce_detection=True
        )
        
        if not embedding_objs or len(embedding_objs) == 0:
            raise ValueError("No se detectó rostro en la imagen")
        
        # Obtener el primer embedding (rostro principal)
        embedding_obj = embedding_objs[0]
        
        logger.info(f"DEBUG - Tipo de embedding_obj: {type(embedding_obj)}")
        logger.info(f"DEBUG - Contenido embedding_obj: {embedding_obj}")
        
        # DeepFace retorna un diccionario con la clave "embedding"
        if isinstance(embedding_obj, dict):
            # Extraer el embedding del diccionario
            embedding_list = embedding_obj.get("embedding")
            if embedding_list is None:
                logger.error(f"DEBUG - Claves disponibles: {embedding_obj.keys()}")
                raise ValueError("No se encontró 'embedding' en la respuesta")
            
            logger.info(f"DEBUG - Tipo de embedding_list: {type(embedding_list)}")
            logger.info(f"DEBUG - Longitud de embedding_list: {len(embedding_list) if isinstance(embedding_list, (list, tuple)) else 'N/A'}")
            
            embedding = np.array(embedding_list, dtype=np.float32)
        else:
            # Si no es diccionario, convertir directamente
            embedding = np.array(embedding_obj, dtype=np.float32)
        
        # Validar que el embedding sea 1D
        if embedding.ndim != 1:
            embedding = embedding.flatten()
        
        # Validar que tenga al menos 128 dimensiones
        if embedding.shape[0] < 128:
            logger.warning(f"Embedding tiene solo {embedding.shape[0]} dimensiones, esperaba 128")
        
        logger.info(f"Embedding extraído: dimensiones {embedding.shape}, tipo {type(embedding)}")
        return embedding
    
    except Exception as e:
        logger.error(f"Error al extraer embedding: {str(e)}")
        raise ValueError(f"Error al extraer características faciales: {str(e)}")


def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calcula la similitud entre dos embeddings usando cosine similarity.
    
    Args:
        embedding1: Primer vector de embedding
        embedding2: Segundo vector de embedding
        
    Returns:
        float: Similitud normalizada (0-100)
    """
    try:
        # Calcular distancia coseno
        distance = cosine(embedding1, embedding2)
        
        # Convertir a similitud (1 - distancia)
        similarity = 1 - distance
        
        # Normalizar a porcentaje (0-100)
        similarity_percentage = similarity * 100
        
        return round(similarity_percentage, 2)
    
    except Exception as e:
        logger.error(f"Error al calcular similitud: {str(e)}")
        return 0.0


@app.get("/health")
async def health_check():
    """
    Endpoint de verificación de salud del servidor.
    """
    return {
        "status": "ok",
        "message": "Servidor de reconocimiento facial activo",
        "model": MODEL_NAME,
        "threshold": SIMILARITY_THRESHOLD
    }


@app.post("/compare-faces")
async def compare_faces(
    selfie: UploadFile = File(...),
    photos: list[UploadFile] = File(...)
):
    """
    Endpoint principal para comparar rostros.
    
    Recibe:
        - selfie: Imagen base del rostro a comparar
        - photos: Lista de imágenes adicionales
        
    Retorna:
        JSON con:
        - matches: Imágenes que coinciden con el selfie
        - non_matches: Imágenes que no coinciden
        - statistics: Estadísticas generales
    """
    
    try:
        logger.info("=== INICIANDO COMPARACIÓN DE ROSTROS ===")
        
        # Validar selfie
        if not validate_image_file(selfie):
            raise HTTPException(
                status_code=400,
                detail="Archivo selfie inválido. Formatos permitidos: jpg, jpeg, png, gif, bmp"
            )
        
        logger.info(f"Selfie recibido: {selfie.filename}")
        
        # Leer selfie
        selfie_image = read_image_to_array(selfie)
        
        # Detectar rostro en selfie
        if not detect_face(selfie_image):
            raise HTTPException(
                status_code=400,
                detail="No se detectó rostro en la imagen selfie"
            )
        
        logger.info("Rostro detectado en selfie")
        
        # Extraer embedding del selfie
        selfie_embedding = extract_face_embedding(selfie_image)
        logger.info("Embedding del selfie extraído exitosamente")
        
        # Procesar fotos adicionales
        matches = []
        non_matches = []
        errors = []
        
        logger.info(f"Procesando {len(photos)} imágenes adicionales...")
        
        for idx, photo in enumerate(photos, 1):
            try:
                logger.info(f"[{idx}/{len(photos)}] Procesando: {photo.filename}")
                
                # Validar foto
                if not validate_image_file(photo):
                    errors.append({
                        "file": photo.filename,
                        "error": "Formato de archivo no permitido"
                    })
                    continue
                
                # Leer foto
                photo_image = read_image_to_array(photo)
                
                # Detectar rostro
                if not detect_face(photo_image):
                    non_matches.append({
                        "file": photo.filename,
                        "similarity": 0.0,
                        "reason": "No se detectó rostro"
                    })
                    logger.info(f"  ✗ No se detectó rostro")
                    continue
                
                # Extraer embedding
                photo_embedding = extract_face_embedding(photo_image)
                
                # Calcular similitud
                similarity = calculate_similarity(selfie_embedding, photo_embedding)
                
                # Convertir threshold a porcentaje para comparación
                threshold_percentage = SIMILARITY_THRESHOLD * 100
                
                # DEBUG: Log de similitud
                logger.info(f"  DEBUG: similarity={similarity}, threshold={threshold_percentage}, match={similarity >= threshold_percentage}")
                
                # Clasificar como coincidencia o no
                if similarity >= threshold_percentage:
                    matches.append({
                        "file": photo.filename,
                        "similarity": similarity
                    })
                    logger.info(f"  ✓ COINCIDENCIA: {similarity}%")
                else:
                    non_matches.append({
                        "file": photo.filename,
                        "similarity": similarity,
                        "reason": "Similitud por debajo del umbral"
                    })
                    logger.info(f"  ✗ No coincide: {similarity}%")
            
            except Exception as e:
                error_msg = str(e)
                errors.append({
                    "file": photo.filename,
                    "error": error_msg
                })
                logger.error(f"  ✗ Error: {error_msg}")
        
        # Construir respuesta
        response = {
            "status": "success",
            "selfie": selfie.filename,
            "matches": matches,
            "non_matches": non_matches,
            "errors": errors,
            "statistics": {
                "total_photos": len(photos),
                "matches_count": len(matches),
                "non_matches_count": len(non_matches),
                "errors_count": len(errors),
                "match_percentage": round(
                    (len(matches) / len(photos) * 100) if len(photos) > 0 else 0, 2
                ),
                "threshold_used": SIMILARITY_THRESHOLD * 100
            }
        }
        
        logger.info("=== COMPARACIÓN COMPLETADA ===")
        logger.info(f"Resultados: {len(matches)} coincidencias, {len(non_matches)} no coincidencias")
        
        return JSONResponse(content=response, status_code=200)
    
    except HTTPException as e:
        logger.error(f"Error HTTP: {e.detail}")
        raise e
    
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)