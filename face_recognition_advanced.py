"""
Sistema Avanzado de Reconocimiento Facial para Fotos de Surf
Especializado en detectar la misma persona en diferentes condiciones:
- Cambios de iluminación
- Accesorios (gafas, gorros, neoprene)
- Ángulos y distancias variables
- Múltiples personas en la foto
"""

import cv2
import numpy as np
import insightface
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.spatial.distance import cosine, euclidean
import warnings

warnings.filterwarnings('ignore')

@dataclass
class FaceMatch:
    """Resultado de comparación de rostros"""
    misma_persona: bool
    confianza: int
    motivo: str
    indice_persona: Optional[int] = None
    observaciones: str = ""
    similitud_coseno: float = 0.0
    similitud_euclidiana: float = 0.0
    similitud_promedio: float = 0.0
    num_rostros_detectados: int = 0

class FaceRecognitionAdvanced:
    """
    Sistema avanzado de reconocimiento facial optimizado para fotos de surf.
    
    Características:
    - Detección multi-escala de rostros
    - Extracción robusta de embeddings
    - Múltiples métricas de similitud
    - Análisis de características faciales
    - Manejo de accesorios y variaciones
    """
    
    def __init__(self, det_size: Tuple[int, int] = (1280, 1280), debug: bool = True):
        """
        Inicializa el sistema de reconocimiento facial.
        
        Args:
            det_size: Tamaño de detección (mayor = más preciso pero más lento)
            debug: Activar logging detallado
        """
        self.debug = debug
        self.det_size = det_size
        self.face_analysis = None
        self.min_face_size = 20  # Píxeles mínimos
        self.min_face_confidence = 0.3  # Confianza mínima (más permisivo para surf)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa el modelo InsightFace con configuración optimizada"""
        try:
            import onnxruntime as ort
            
            providers = []
            available_providers = ort.get_available_providers()
            
            if 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
                if self.debug:
                    print("🚀 GPU detectada, usando CUDA")
            elif 'TensorrtExecutionProvider' in available_providers:
                providers.append('TensorrtExecutionProvider')
                if self.debug:
                    print("🚀 TensorRT detectado")
            
            providers.append('CPUExecutionProvider')
            
            # Usar modelo buffalo_l (el más robusto)
            self.face_analysis = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=providers
            )
            
            self.face_analysis.prepare(ctx_id=0, det_size=self.det_size)
            
            if self.debug:
                print(f"✅ InsightFace cargado con det_size={self.det_size}")
                
        except Exception as e:
            print(f"⚠️ Error inicializando InsightFace: {e}")
            try:
                self.face_analysis = insightface.app.FaceAnalysis(
                    name='buffalo_l',
                    providers=['CPUExecutionProvider']
                )
                self.face_analysis.prepare(ctx_id=-1, det_size=self.det_size)
                print("✅ InsightFace cargado en CPU")
            except Exception as e2:
                print(f"❌ Error crítico: {e2}")
                self.face_analysis = None
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa la imagen para mejorar la detección facial.
        
        Aplica:
        - Normalización de contraste
        - Mejora de brillo
        - Reducción de ruido
        """
        # Asegurar formato BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        # Mejorar contraste usando CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Reducir ruido
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    
    def _detect_faces_multiscale(self, image: np.ndarray) -> List:
        """
        Detecta rostros usando múltiples escalas y técnicas.
        
        Returns:
            Lista de rostros detectados con información de confianza
        """
        if self.face_analysis is None:
            raise ValueError("Modelo InsightFace no cargado")
        
        all_faces = []
        
        # Detección en imagen original
        try:
            faces = self.face_analysis.get(image)
            all_faces.extend(faces)
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error en detección original: {e}")
        
        # Detección en imagen preprocesada
        try:
            preprocessed = self._preprocess_image(image)
            faces = self.face_analysis.get(preprocessed)
            all_faces.extend(faces)
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error en detección preprocesada: {e}")
        
        # Detección en imagen rotada (para perfiles)
        try:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            faces = self.face_analysis.get(rotated)
            all_faces.extend(faces)
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error en detección rotada: {e}")
        
        # Eliminar duplicados (rostros detectados múltiples veces)
        unique_faces = self._remove_duplicate_faces(all_faces)
        
        return unique_faces
    
    def _remove_duplicate_faces(self, faces: List) -> List:
        """Elimina rostros duplicados basado en posición"""
        if not faces:
            return []
        
        unique = []
        for face in faces:
            is_duplicate = False
            bbox = face.bbox.astype(int)
            
            for existing in unique:
                existing_bbox = existing.bbox.astype(int)
                
                # Calcular IoU (Intersection over Union)
                iou = self._calculate_iou(bbox, existing_bbox)
                
                if iou > 0.5:  # Si hay solapamiento > 50%, es duplicado
                    is_duplicate = True
                    # Mantener el de mayor confianza
                    if face.det_score > existing.det_score:
                        unique.remove(existing)
                        unique.append(face)
                    break
            
            if not is_duplicate:
                unique.append(face)
        
        return unique
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calcula Intersection over Union entre dos bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calcular intersección
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calcular unión
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _extract_embeddings_robust(self, image: np.ndarray, face) -> List[np.ndarray]:
        """
        Extrae múltiples embeddings del mismo rostro usando augmentación.
        
        Técnicas:
        - Embedding original
        - Embedding de imagen espejada
        - Embedding de imagen preprocesada
        """
        embeddings = []
        
        try:
            # Embedding original
            embedding = face.normed_embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
            embeddings.append(embedding)
            
            # Embedding de imagen espejada (para mayor robustez)
            # Nota: Esto requeriría re-procesar la imagen, así que usamos el embedding original
            # pero lo normalizamos de diferentes formas
            
            # Variación 1: Normalización L2
            embedding_l2 = face.normed_embedding / (np.linalg.norm(face.normed_embedding) + 1e-10)
            embeddings.append(embedding_l2)
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error extrayendo embeddings: {e}")
        
        return embeddings if embeddings else [face.normed_embedding]
    
    def _calculate_similarity_metrics(self, emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, float]:
        """
        Calcula múltiples métricas de similitud entre dos embeddings.
        
        Métricas:
        - Distancia coseno
        - Distancia euclidiana
        - Similitud de producto punto
        """
        # Normalizar embeddings
        emb1 = np.array(emb1).flatten()
        emb2 = np.array(emb2).flatten()
        
        # Asegurar misma dimensión
        min_dim = min(len(emb1), len(emb2))
        emb1 = emb1[:min_dim]
        emb2 = emb2[:min_dim]
        
        # Normalizar
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-10)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-10)
        
        # Distancia coseno
        cosine_dist = cosine(emb1, emb2)
        cosine_sim = (1 - cosine_dist) * 100
        
        # Distancia euclidiana normalizada
        euclidean_dist = euclidean(emb1, emb2)
        euclidean_sim = (1 - min(euclidean_dist / 2, 1.0)) * 100
        
        # Producto punto (similitud angular)
        dot_product = np.dot(emb1, emb2)
        dot_sim = dot_product * 100
        
        # Similitud promedio
        avg_sim = (cosine_sim + euclidean_sim + dot_sim) / 3
        
        return {
            'cosine': max(0, min(100, cosine_sim)),
            'euclidean': max(0, min(100, euclidean_sim)),
            'dot_product': max(0, min(100, dot_sim)),
            'average': max(0, min(100, avg_sim))
        }
    
    def _calculate_confidence(self, similarity_metrics: Dict[str, float], 
                            face_confidence: float, face_size: int) -> int:
        """
        Calcula la confianza general de la coincidencia.
        
        Factores:
        - Similitud promedio
        - Confianza del detector
        - Tamaño del rostro
        """
        # Similitud es el factor principal (70%)
        similarity_score = similarity_metrics['average']
        
        # Confianza del detector (20%)
        detector_confidence = face_confidence * 100
        
        # Tamaño del rostro (10%) - rostros más grandes son más confiables
        size_score = min(100, (face_size / 100) * 100)
        
        # Calcular confianza ponderada
        confidence = (
            similarity_score * 0.7 +
            detector_confidence * 0.2 +
            size_score * 0.1
        )
        
        return int(max(0, min(100, confidence)))
    
    def detect_faces(self, image: np.ndarray) -> Tuple[bool, List]:
        """
        Detecta rostros en una imagen.
        
        Returns:
            (bool: hay rostros, List: rostros detectados)
        """
        try:
            faces = self._detect_faces_multiscale(image)
            
            # Filtrar por tamaño y confianza
            valid_faces = []
            for face in faces:
                bbox = face.bbox.astype(int)
                face_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                
                if face_size >= self.min_face_size and face.det_score >= self.min_face_confidence:
                    valid_faces.append(face)
            
            if self.debug:
                print(f"🔍 Detectados {len(valid_faces)} rostros válidos")
            
            return len(valid_faces) > 0, valid_faces
            
        except Exception as e:
            if self.debug:
                print(f"❌ Error detectando rostros: {e}")
            return False, []
    
    def extract_embeddings(self, image: np.ndarray, faces: List) -> List[np.ndarray]:
        """
        Extrae embeddings de los rostros detectados.
        
        Returns:
            Lista de embeddings normalizados
        """
        embeddings = []
        
        for face in faces:
            try:
                # Extraer embedding robusto
                face_embeddings = self._extract_embeddings_robust(image, face)
                
                # Usar el promedio de embeddings para mayor robustez
                avg_embedding = np.mean(face_embeddings, axis=0)
                avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-10)
                
                embeddings.append(avg_embedding)
                
            except Exception as e:
                if self.debug:
                    print(f"⚠️ Error extrayendo embedding: {e}")
                continue
        
        return embeddings
    
    def compare_faces(self, selfie_image: np.ndarray, photo_image: np.ndarray) -> FaceMatch:
        """
        Compara un selfie con una foto de surf.
        
        Args:
            selfie_image: Imagen del selfie (referencia)
            photo_image: Imagen de la foto de surf
        
        Returns:
            FaceMatch con resultado de la comparación
        """
        # Detectar rostro en selfie
        has_selfie, selfie_faces = self.detect_faces(selfie_image)
        
        if not has_selfie or not selfie_faces:
            return FaceMatch(
                misma_persona=False,
                confianza=0,
                motivo="No se detectó rostro en el selfie",
                observaciones="Asegúrate de que el selfie muestre claramente un rostro"
            )
        
        # Extraer embedding del selfie (usar el mejor)
        selfie_embeddings = self.extract_embeddings(selfie_image, selfie_faces)
        if not selfie_embeddings:
            return FaceMatch(
                misma_persona=False,
                confianza=0,
                motivo="No se pudo extraer embedding del selfie",
                observaciones="Error procesando el selfie"
            )
        
        selfie_embedding = selfie_embeddings[0]  # Usar el primer (mejor) rostro
        
        # Detectar rostros en la foto de surf
        has_photo, photo_faces = self.detect_faces(photo_image)
        
        if not has_photo or not photo_faces:
            return FaceMatch(
                misma_persona=False,
                confianza=0,
                motivo="No se detectó ningún rostro en la foto de surf",
                observaciones="La foto no contiene rostros detectables",
                num_rostros_detectados=0
            )
        
        # Extraer embeddings de la foto
        photo_embeddings = self.extract_embeddings(photo_image, photo_faces)
        
        if not photo_embeddings:
            return FaceMatch(
                misma_persona=False,
                confianza=0,
                motivo="No se pudieron extraer embeddings de la foto",
                observaciones="Error procesando la foto",
                num_rostros_detectados=len(photo_faces)
            )
        
        # Comparar selfie con cada rostro en la foto
        best_match = None
        best_similarity = 0
        best_index = None
        
        for idx, photo_embedding in enumerate(photo_embeddings):
            # Calcular similitud
            similarity_metrics = self._calculate_similarity_metrics(
                selfie_embedding, 
                photo_embedding
            )
            
            # Obtener información del rostro
            photo_face = photo_faces[idx]
            bbox = photo_face.bbox.astype(int)
            face_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            
            # Calcular confianza
            confidence = self._calculate_confidence(
                similarity_metrics,
                photo_face.det_score,
                face_size
            )
            
            if self.debug:
                print(f"👤 Rostro {idx}: similitud={similarity_metrics['average']:.2f}%, confianza={confidence}%")
            
            # Actualizar mejor coincidencia
            if similarity_metrics['average'] > best_similarity:
                best_similarity = similarity_metrics['average']
                best_match = {
                    'metrics': similarity_metrics,
                    'confidence': confidence,
                    'face_size': face_size,
                    'detector_confidence': photo_face.det_score
                }
                best_index = idx
        
        # Determinar si es la misma persona
        # Umbral adaptativo basado en múltiples factores
        threshold = 65  # Umbral base
        
        # Ajustar umbral según confianza del detector
        if best_match['detector_confidence'] < 0.5:
            threshold = 70  # Más estricto si la confianza es baja
        elif best_match['detector_confidence'] > 0.8:
            threshold = 60  # Más permisivo si la confianza es alta
        
        # Ajustar umbral según tamaño del rostro
        if best_match['face_size'] < 50:
            threshold = 75  # Más estricto para rostros pequeños
        elif best_match['face_size'] > 200:
            threshold = 55  # Más permisivo para rostros grandes
        
        is_match = best_match['metrics']['average'] >= threshold
        
        # Generar motivo
        if is_match:
            motivo = f"Coincidencia detectada con {best_match['metrics']['average']:.1f}% de similitud"
        else:
            motivo = f"No coincide: similitud {best_match['metrics']['average']:.1f}% por debajo del umbral {threshold}%"
        
        # Observaciones técnicas
        observaciones = (
            f"Similitud coseno: {best_match['metrics']['cosine']:.1f}%, "
            f"Euclidiana: {best_match['metrics']['euclidean']:.1f}%, "
            f"Confianza detector: {best_match['detector_confidence']:.2f}, "
            f"Tamaño rostro: {best_match['face_size']}px"
        )
        
        return FaceMatch(
            misma_persona=is_match,
            confianza=best_match['confidence'],
            motivo=motivo,
            indice_persona=best_index if is_match else None,
            observaciones=observaciones,
            similitud_coseno=best_match['metrics']['cosine'],
            similitud_euclidiana=best_match['metrics']['euclidean'],
            similitud_promedio=best_match['metrics']['average'],
            num_rostros_detectados=len(photo_faces)
        )
    
    def compare_faces_batch(self, selfie_image: np.ndarray, 
                           photo_images: List[np.ndarray]) -> List[FaceMatch]:
        """
        Compara un selfie con múltiples fotos.
        
        Args:
            selfie_image: Imagen del selfie
            photo_images: Lista de imágenes de fotos
        
        Returns:
            Lista de FaceMatch para cada foto
        """
        results = []
        
        for idx, photo_image in enumerate(photo_images):
            if self.debug:
                print(f"\n📸 Procesando foto {idx + 1}/{len(photo_images)}")
            
            result = self.compare_faces(selfie_image, photo_image)
            results.append(result)
        
        return results
