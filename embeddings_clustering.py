"""
Sistema de Pre-procesamiento de Embeddings con Clustering
Optimiza el reconocimiento facial agrupando fotos similares
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import insightface
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

class EmbeddingsClusteringSystem:
    """
    Sistema de clustering de embeddings faciales para optimizar búsquedas.
    
    Características:
    - Extracción masiva de embeddings
    - Clustering automático con K-Means
    - Número óptimo de clusters según cantidad de fotos
    - Guardado de índices para búsqueda rápida
    """
    
    def __init__(self, storage_dir: Path, embeddings_dir: Path, debug: bool = True):
        """
        Inicializa el sistema de clustering.
        
        Args:
            storage_dir: Directorio donde están las fotos
            embeddings_dir: Directorio donde guardar embeddings y clusters
            debug: Activar logging detallado
        """
        self.storage_dir = storage_dir
        self.embeddings_dir = embeddings_dir
        self.debug = debug
        self.face_analysis = None
        
        # Crear directorio de embeddings si no existe
        self.embeddings_dir.mkdir(exist_ok=True)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa el modelo InsightFace"""
        try:
            import onnxruntime as ort
            
            providers = []
            available_providers = ort.get_available_providers()
            
            if 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
                if self.debug:
                    print("🚀 GPU detectada para clustering")
            
            providers.append('CPUExecutionProvider')
            
            self.face_analysis = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=providers
            )
            
            self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
            
            if self.debug:
                print("✅ InsightFace cargado para clustering")
                
        except Exception as e:
            print(f"⚠️ Error inicializando InsightFace: {e}")
            self.face_analysis = None
    
    def _extract_single_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Extrae el embedding de una sola foto.
        
        Args:
            image_path: Ruta de la foto
        
        Returns:
            Embedding normalizado o None si falla
        """
        try:
            # Leer imagen
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Detectar rostros
            faces = self.face_analysis.get(image)
            
            if len(faces) == 0:
                if self.debug:
                    print(f"⚠️ No se detectó rostro en {image_path.name}")
                return None
            
            # Usar el rostro con mayor confianza
            best_face = max(faces, key=lambda f: f.det_score)
            
            # Obtener embedding normalizado
            embedding = best_face.normed_embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
            
            return embedding
            
        except Exception as e:
            if self.debug:
                print(f"❌ Error procesando {image_path.name}: {e}")
            return None
    
    def extract_all_embeddings(self, folder_name: str, day: Optional[str] = None) -> Dict:
        """
        Extrae embeddings de todas las fotos en una carpeta/día.
        
        Args:
            folder_name: Nombre de la carpeta
            day: Día específico (opcional)
        
        Returns:
            Dict con embeddings y metadata
        """
        if self.face_analysis is None:
            raise ValueError("Modelo InsightFace no cargado")
        
        # Determinar ruta de búsqueda
        folder_path = self.storage_dir / folder_name
        if day:
            search_path = folder_path / day
            label = f"{folder_name}/{day}"
        else:
            search_path = folder_path
            label = folder_name
        
        if not search_path.exists():
            raise ValueError(f"Carpeta no existe: {search_path}")
        
        # Obtener todas las fotos
        photo_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        photos = [f for f in search_path.iterdir() 
                 if f.suffix.lower() in photo_extensions 
                 and not f.name.startswith('cover')]
        
        if len(photos) == 0:
            raise ValueError(f"No hay fotos en {label}")
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"📸 EXTRAYENDO EMBEDDINGS - {label}")
            print(f"{'='*60}")
            print(f"Total de fotos: {len(photos)}")
        
        # Extraer embeddings
        embeddings_list = []
        filenames_list = []
        failed_count = 0
        
        start_time = datetime.now()
        
        for idx, photo_path in enumerate(photos, 1):
            if self.debug and idx % 50 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = idx / elapsed if elapsed > 0 else 0
                print(f"⏳ Progreso: {idx}/{len(photos)} ({rate:.1f} fotos/seg)")
            
            embedding = self._extract_single_embedding(photo_path)
            
            if embedding is not None:
                embeddings_list.append(embedding)
                filenames_list.append(photo_path.name)
            else:
                failed_count += 1
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        if self.debug:
            print(f"\n✅ Embeddings extraídos: {len(embeddings_list)}")
            print(f"❌ Fotos sin rostro: {failed_count}")
            print(f"⏱️  Tiempo total: {total_time:.2f} segundos")
            print(f"⚡ Velocidad: {len(photos) / total_time:.1f} fotos/segundo")
        
        if len(embeddings_list) == 0:
            raise ValueError("No se pudo extraer ningún embedding")
        
        # Convertir a array numpy
        embeddings_array = np.array(embeddings_list)
        
        return {
            "embeddings": embeddings_array,
            "filenames": filenames_list,
            "total_photos": len(photos),
            "successful": len(embeddings_list),
            "failed": failed_count,
            "processing_time": total_time
        }
    
    def _calculate_optimal_clusters(self, n_samples: int) -> int:
        """
        Calcula el número óptimo de clusters según la cantidad de fotos.
        
        Reglas:
        - < 50 fotos: 3-5 clusters
        - 50-100 fotos: 5-10 clusters
        - 100-500 fotos: 10-20 clusters
        - 500-2000 fotos: 20-50 clusters
        - > 2000 fotos: 50-100 clusters
        """
        if n_samples < 50:
            return max(3, min(5, n_samples // 10))
        elif n_samples < 100:
            return max(5, min(10, n_samples // 10))
        elif n_samples < 500:
            return max(10, min(20, n_samples // 25))
        elif n_samples < 2000:
            return max(20, min(50, n_samples // 40))
        else:
            return max(50, min(100, n_samples // 50))
    
    def create_clusters(self, embeddings: np.ndarray, n_clusters: Optional[int] = None) -> Dict:
        """
        Crea clusters de embeddings usando K-Means.
        
        Args:
            embeddings: Array de embeddings (n_samples, embedding_dim)
            n_clusters: Número de clusters (None = automático)
        
        Returns:
            Dict con información de clusters
        """
        n_samples = len(embeddings)
        
        if n_samples < 3:
            raise ValueError("Se necesitan al menos 3 fotos para crear clusters")
        
        # Calcular número óptimo de clusters
        if n_clusters is None:
            n_clusters = self._calculate_optimal_clusters(n_samples)
        
        # Asegurar que no haya más clusters que muestras
        n_clusters = min(n_clusters, n_samples)
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"🔬 CREANDO CLUSTERS")
            print(f"{'='*60}")
            print(f"Muestras: {n_samples}")
            print(f"Clusters: {n_clusters}")
        
        start_time = datetime.now()
        
        # Usar MiniBatchKMeans para datasets grandes (más rápido)
        if n_samples > 1000:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=256,
                max_iter=100,
                n_init=3
            )
        else:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                max_iter=300,
                n_init=10
            )
        
        # Entrenar modelo
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calcular centroides
        centroids = kmeans.cluster_centers_
        
        # Calcular silhouette score (calidad de clustering)
        try:
            if n_samples > 100:
                # Para datasets grandes, usar muestra
                sample_size = min(1000, n_samples)
                indices = np.random.choice(n_samples, sample_size, replace=False)
                silhouette = silhouette_score(
                    embeddings[indices], 
                    cluster_labels[indices],
                    metric='euclidean'
                )
            else:
                silhouette = silhouette_score(
                    embeddings, 
                    cluster_labels,
                    metric='euclidean'
                )
        except Exception as e:
            if self.debug:
                print(f"⚠️ No se pudo calcular silhouette score: {e}")
            silhouette = 0.0
        
        # Calcular estadísticas por cluster
        cluster_stats = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_size = np.sum(cluster_mask)
            
            cluster_stats.append({
                "cluster_id": int(i),
                "size": int(cluster_size),
                "percentage": float(cluster_size / n_samples * 100)
            })
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        if self.debug:
            print(f"\n✅ Clustering completado")
            print(f"⏱️  Tiempo: {total_time:.2f} segundos")
            print(f"📊 Silhouette Score: {silhouette:.3f}")
            print(f"\n📈 Distribución de clusters:")
            for stat in sorted(cluster_stats, key=lambda x: x['size'], reverse=True):
                print(f"  Cluster {stat['cluster_id']}: {stat['size']} fotos ({stat['percentage']:.1f}%)")
        
        return {
            "centroids": centroids,
            "labels": cluster_labels,
            "n_clusters": n_clusters,
            "silhouette_score": float(silhouette),
            "cluster_stats": cluster_stats,
            "processing_time": total_time
        }
    
    def save_embeddings_and_clusters(
        self, 
        folder_name: str, 
        embeddings_data: Dict, 
        clusters_data: Dict,
        day: Optional[str] = None
    ):
        """
        Guarda embeddings y clusters en disco.
        
        Args:
            folder_name: Nombre de la carpeta
            embeddings_data: Datos de embeddings
            clusters_data: Datos de clusters
            day: Día específico (opcional)
        """
        # Crear directorio para la carpeta
        if day:
            save_dir = self.embeddings_dir / folder_name / day
            label = f"{folder_name}/{day}"
        else:
            save_dir = self.embeddings_dir / folder_name
            label = folder_name
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar embeddings
        embeddings_path = save_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings_data["embeddings"])
        
        # Guardar centroides
        centroids_path = save_dir / "centroids.npy"
        np.save(centroids_path, clusters_data["centroids"])
        
        # Guardar labels
        labels_path = save_dir / "labels.npy"
        np.save(labels_path, clusters_data["labels"])
        
        # Guardar metadata
        metadata = {
            "folder_name": folder_name,
            "day": day,
            "created_at": datetime.now().isoformat(),
            "filenames": embeddings_data["filenames"],
            "total_photos": embeddings_data["total_photos"],
            "successful_embeddings": embeddings_data["successful"],
            "failed_embeddings": embeddings_data["failed"],
            "n_clusters": clusters_data["n_clusters"],
            "silhouette_score": clusters_data["silhouette_score"],
            "cluster_stats": clusters_data["cluster_stats"],
            "embedding_extraction_time": embeddings_data["processing_time"],
            "clustering_time": clusters_data["processing_time"]
        }
        
        metadata_path = save_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"💾 GUARDADO COMPLETADO - {label}")
            print(f"{'='*60}")
            print(f"📁 Directorio: {save_dir}")
            print(f"📄 Archivos guardados:")
            print(f"  - embeddings.npy ({embeddings_data['embeddings'].nbytes / 1024:.2f} KB)")
            print(f"  - centroids.npy ({clusters_data['centroids'].nbytes / 1024:.2f} KB)")
            print(f"  - labels.npy ({clusters_data['labels'].nbytes / 1024:.2f} KB)")
            print(f"  - metadata.json")

    def save_identity_groups(
        self,
        folder_name: str,
        day: Optional[str],
        filenames: List[str],
        identity_labels: np.ndarray,
        identity_centroids: np.ndarray,
        identity_stats: List[Dict],
        source: str,
        embedding_extraction_time: float,
        grouping_time: float,
    ):
        """
        Guarda el agrupamiento por identidad (persona) en disco.

        Nota:
        - identity_labels es por-foto (mismo orden que filenames)
        - identity_centroids es por-grupo (sin incluir ruido -1)
        """
        if day:
            save_dir = self.embeddings_dir / folder_name / day
            label = f"{folder_name}/{day}"
        else:
            save_dir = self.embeddings_dir / folder_name
            label = folder_name

        save_dir.mkdir(parents=True, exist_ok=True)

        # Archivos
        np.save(save_dir / "identity_labels.npy", identity_labels)
        np.save(save_dir / "identity_centroids.npy", identity_centroids)

        identity_metadata = {
            "folder_name": folder_name,
            "day": day,
            "created_at": datetime.now().isoformat(),
            "source": source,  # "full" o "incremental"
            "filenames": filenames,
            "n_photos": len(filenames),
            "n_identities": int(len(identity_stats)),
            "identity_stats": identity_stats,
            "embedding_extraction_time": embedding_extraction_time,
            "grouping_time": grouping_time,
        }

        with open(save_dir / "identity_metadata.json", "w") as f:
            json.dump(identity_metadata, f, indent=2)

        if self.debug:
            print(f"\n{'='*60}")
            print(f"👥 AGRUPAMIENTO POR IDENTIDAD GUARDADO - {label}")
            print(f"{'='*60}")
            print(f"📁 Directorio: {save_dir}")
            print(f"  - identity_labels.npy")
            print(f"  - identity_centroids.npy")
            print(f"  - identity_metadata.json")
    
    def load_clusters(self, folder_name: str, day: Optional[str] = None) -> Optional[Dict]:
        """
        Carga clusters pre-calculados desde disco.
        
        Args:
            folder_name: Nombre de la carpeta
            day: Día específico (opcional)
        
        Returns:
            Dict con datos de clusters o None si no existen
        """
        if day:
            load_dir = self.embeddings_dir / folder_name / day
        else:
            load_dir = self.embeddings_dir / folder_name
        
        # Verificar que existan los archivos
        embeddings_path = load_dir / "embeddings.npy"
        centroids_path = load_dir / "centroids.npy"
        labels_path = load_dir / "labels.npy"
        metadata_path = load_dir / "metadata.json"
        
        if not all([p.exists() for p in [embeddings_path, centroids_path, labels_path, metadata_path]]):
            return None

    def load_identities(self, folder_name: str, day: Optional[str] = None) -> Optional[Dict]:
        """
        Carga los grupos por identidad (persona) desde disco.
        """
        if day:
            load_dir = self.embeddings_dir / folder_name / day
        else:
            load_dir = self.embeddings_dir / folder_name

        labels_path = load_dir / "identity_labels.npy"
        centroids_path = load_dir / "identity_centroids.npy"
        metadata_path = load_dir / "identity_metadata.json"

        if not all([p.exists() for p in [labels_path, centroids_path, metadata_path]]):
            return None

        try:
            identity_labels = np.load(labels_path)
            identity_centroids = np.load(centroids_path)
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            return {
                "identity_labels": identity_labels,
                "identity_centroids": identity_centroids,
                "filenames": metadata.get("filenames", []),
                "metadata": metadata,
            }
        except Exception as e:
            if self.debug:
                print(f"❌ Error cargando identidades: {e}")
            return None

    def create_identity_groups(
        self,
        embeddings: np.ndarray,
        filenames: List[str],
        eps_cosine: float = 0.35,
        min_samples: int = 2,
    ) -> Dict:
        """
        Agrupa embeddings por identidad usando DBSCAN con métrica coseno.

        - eps_cosine: umbral de distancia coseno (menor = más estricto).
        - min_samples: mínimo de fotos para formar una identidad.

        Retorna:
        - identity_labels: label por foto (0..k-1) y -1 para ruido
        - identity_centroids: centroides por identidad (k, dim)
        - identity_stats: tamaño y % por identidad
        """
        n_samples = len(embeddings)
        if n_samples == 0:
            raise ValueError("No hay embeddings para agrupar")
        if n_samples != len(filenames):
            raise ValueError("embeddings y filenames deben tener la misma longitud")

        if self.debug:
            print(f"\n{'='*60}")
            print("👥 CREANDO GRUPOS POR IDENTIDAD (DBSCAN)")
            print(f"{'='*60}")
            print(f"Muestras: {n_samples}")
            print(f"eps_cosine: {eps_cosine}")
            print(f"min_samples: {min_samples}")

        start_time = datetime.now()

        # DBSCAN con métrica coseno: requiere embeddings normalizados (ya lo están)
        db = DBSCAN(eps=eps_cosine, min_samples=min_samples, metric="cosine")
        raw_labels = db.fit_predict(embeddings)

        # Re-mapear labels para que sean consecutivos 0..k-1 (manteniendo -1 como ruido)
        unique = sorted([l for l in set(raw_labels.tolist()) if l != -1])
        remap = {old: new for new, old in enumerate(unique)}
        identity_labels = np.array([remap.get(int(l), -1) for l in raw_labels], dtype=np.int32)

        n_identities = len(unique)
        identity_centroids = np.zeros((n_identities, embeddings.shape[1]), dtype=np.float32)

        identity_stats: List[Dict] = []
        for i in range(n_identities):
            mask = identity_labels == i
            group_embs = embeddings[mask]
            centroid = np.mean(group_embs, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
            identity_centroids[i] = centroid.astype(np.float32)
            size = int(np.sum(mask))
            identity_stats.append(
                {"identity_id": int(i), "size": size, "percentage": float(size / n_samples * 100)}
            )

        noise_count = int(np.sum(identity_labels == -1))
        total_time = (datetime.now() - start_time).total_seconds()

        if self.debug:
            print(f"\n✅ Agrupamiento completado")
            print(f"⏱️  Tiempo: {total_time:.2f} segundos")
            print(f"👤 Identidades: {n_identities}")
            print(f"🧩 Ruido (sin grupo): {noise_count}")
            if n_identities > 0:
                print("\n📈 Distribución de identidades:")
                for stat in sorted(identity_stats, key=lambda x: x["size"], reverse=True)[:20]:
                    print(f"  Persona {stat['identity_id']}: {stat['size']} fotos ({stat['percentage']:.1f}%)")

        return {
            "identity_labels": identity_labels,
            "identity_centroids": identity_centroids,
            "n_identities": n_identities,
            "noise_count": noise_count,
            "identity_stats": identity_stats,
            "processing_time": total_time,
        }

    def process_incremental(
        self,
        folder_name: str,
        day: Optional[str],
        new_filenames: List[str],
        eps_cosine: float = 0.35,
        min_samples: int = 2,
    ) -> Dict:
        """
        Procesamiento incremental:
        - extrae embeddings SOLO de nuevas fotos
        - concatena con embeddings existentes (si hay)
        - re-agrupa por identidad y guarda índices
        """
        label = f"{folder_name}/{day}" if day else folder_name

        # Cargar estado existente si existe
        existing = self.load_clusters(folder_name, day)
        if existing is None:
            # No hay estado previo: procesar carpeta completa (sin force)
            if self.debug:
                print(f"ℹ️  No hay estado previo para {label}, procesando carpeta completa")
            return self.process_folder(folder_name, day, force=True)

        existing_filenames = existing["metadata"].get("filenames", [])
        existing_embeddings = existing["embeddings"]

        # Filtrar solo filenames realmente nuevos
        existing_set = set(existing_filenames)
        filtered_new = [fn for fn in new_filenames if fn not in existing_set]

        if len(filtered_new) == 0:
            if self.debug:
                print(f"ℹ️  No hay fotos nuevas para {label} (incremental)")
            # También intentamos asegurar que identidades existan
            identities = self.load_identities(folder_name, day)
            return {
                "status": "already_up_to_date",
                "message": f"Sin fotos nuevas para {label}",
                "total_photos": len(existing_filenames),
                "successful_embeddings": len(existing_filenames),
                "failed_embeddings": 0,
                "n_clusters": existing["metadata"].get("n_clusters"),
                "n_identities": identities["metadata"]["n_identities"] if identities else None,
            }

        # Extraer embeddings de nuevas fotos
        folder_path = self.storage_dir / folder_name
        search_path = (folder_path / day) if day else folder_path

        if self.debug:
            print(f"\n🚀 Procesamiento incremental - {label}")
            print(f"🆕 Nuevas fotos: {len(filtered_new)}")

        start_time = datetime.now()
        new_embeddings: List[np.ndarray] = []
        ok_names: List[str] = []
        failed = 0

        for fn in filtered_new:
            emb = self._extract_single_embedding(search_path / fn)
            if emb is None:
                failed += 1
                continue
            new_embeddings.append(emb)
            ok_names.append(fn)

        embedding_time = (datetime.now() - start_time).total_seconds()

        if len(new_embeddings) == 0:
            return {
                "status": "error",
                "message": f"No se pudo extraer embeddings de las nuevas fotos en {label}",
                "failed_embeddings": failed,
            }

        # Concatenar (ojo: existing_embeddings es numpy array)
        new_embeddings_array = np.array(new_embeddings)
        merged_embeddings = np.concatenate([existing_embeddings, new_embeddings_array], axis=0)
        merged_filenames = list(existing_filenames) + list(ok_names)

        # Re-clustering rápido (KMeans) para la optimización actual de búsqueda por clusters
        clusters_data = self.create_clusters(merged_embeddings)

        # Guardar embeddings + clusters (estado base)
        embeddings_data = {
            "embeddings": merged_embeddings,
            "filenames": merged_filenames,
            "total_photos": len(merged_filenames),
            "successful": len(merged_filenames),
            "failed": failed,
            "processing_time": embedding_time,
        }
        self.save_embeddings_and_clusters(folder_name, embeddings_data, clusters_data, day)

        # Agrupar por identidad (persona) y guardar
        identity_groups = self.create_identity_groups(
            merged_embeddings,
            merged_filenames,
            eps_cosine=eps_cosine,
            min_samples=min_samples,
        )
        self.save_identity_groups(
            folder_name=folder_name,
            day=day,
            filenames=merged_filenames,
            identity_labels=identity_groups["identity_labels"],
            identity_centroids=identity_groups["identity_centroids"],
            identity_stats=identity_groups["identity_stats"],
            source="incremental",
            embedding_extraction_time=embedding_time,
            grouping_time=identity_groups["processing_time"],
        )

        return {
            "status": "success",
            "message": f"Incremental completado para {label}",
            "total_photos": len(merged_filenames),
            "new_photos_processed": len(ok_names),
            "failed_new_photos": failed,
            "n_clusters": clusters_data["n_clusters"],
            "n_identities": identity_groups["n_identities"],
            "total_time": float(embedding_time + clusters_data["processing_time"] + identity_groups["processing_time"]),
        }
        
        try:
            # Cargar datos
            embeddings = np.load(embeddings_path)
            centroids = np.load(centroids_path)
            labels = np.load(labels_path)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                "embeddings": embeddings,
                "centroids": centroids,
                "labels": labels,
                "filenames": metadata["filenames"],
                "metadata": metadata
            }
            
        except Exception as e:
            if self.debug:
                print(f"❌ Error cargando clusters: {e}")
            return None
    
    def process_folder(self, folder_name: str, day: Optional[str] = None, force: bool = False) -> Dict:
        """
        Procesa una carpeta completa: extrae embeddings y crea clusters.
        
        Args:
            folder_name: Nombre de la carpeta
            day: Día específico (opcional)
            force: Forzar re-procesamiento aunque ya existan clusters
        
        Returns:
            Dict con resultado del procesamiento
        """
        label = f"{folder_name}/{day}" if day else folder_name
        
        # Verificar si ya existen clusters
        if not force:
            existing_clusters = self.load_clusters(folder_name, day)
            if existing_clusters is not None:
                if self.debug:
                    print(f"ℹ️  Ya existen clusters para {label}")
                return {
                    "status": "already_exists",
                    "message": f"Clusters ya existen para {label}",
                    "metadata": existing_clusters["metadata"]
                }
        
        try:
            # Paso 1: Extraer embeddings
            if self.debug:
                print(f"\n🚀 Iniciando procesamiento de {label}")
            
            embeddings_data = self.extract_all_embeddings(folder_name, day)
            
            # Paso 2: Crear clusters
            clusters_data = self.create_clusters(embeddings_data["embeddings"])

            # Paso 2b: Crear identidades (persona)
            identity_groups = self.create_identity_groups(
                embeddings_data["embeddings"],
                embeddings_data["filenames"],
            )
            
            # Paso 3: Guardar todo
            self.save_embeddings_and_clusters(
                folder_name, 
                embeddings_data, 
                clusters_data,
                day
            )

            self.save_identity_groups(
                folder_name=folder_name,
                day=day,
                filenames=embeddings_data["filenames"],
                identity_labels=identity_groups["identity_labels"],
                identity_centroids=identity_groups["identity_centroids"],
                identity_stats=identity_groups["identity_stats"],
                source="full",
                embedding_extraction_time=embeddings_data["processing_time"],
                grouping_time=identity_groups["processing_time"],
            )
            
            return {
                "status": "success",
                "message": f"Procesamiento completado para {label}",
                "total_photos": embeddings_data["total_photos"],
                "successful_embeddings": embeddings_data["successful"],
                "failed_embeddings": embeddings_data["failed"],
                "n_clusters": clusters_data["n_clusters"],
                "n_identities": identity_groups["n_identities"],
                "silhouette_score": clusters_data["silhouette_score"],
                "total_time": embeddings_data["processing_time"] + clusters_data["processing_time"] + identity_groups["processing_time"]
            }
            
        except Exception as e:
            if self.debug:
                print(f"❌ Error procesando {label}: {e}")
            return {
                "status": "error",
                "message": f"Error procesando {label}: {str(e)}"
            }
    
    def find_relevant_clusters(
        self, 
        selfie_embedding: np.ndarray, 
        centroids: np.ndarray,
        top_k: int = 3
    ) -> List[int]:
        """
        Encuentra los clusters más relevantes para un selfie.
        
        Args:
            selfie_embedding: Embedding del selfie
            centroids: Centroides de los clusters
            top_k: Número de clusters a retornar
        
        Returns:
            Lista de índices de clusters más similares
        """
        from scipy.spatial.distance import cosine
        
        # Calcular distancia a cada centroide
        distances = []
        for i, centroid in enumerate(centroids):
            dist = cosine(selfie_embedding, centroid)
            distances.append((i, dist))
        
        # Ordenar por distancia (menor = más similar)
        distances.sort(key=lambda x: x[1])
        
        # Retornar top_k clusters
        top_clusters = [cluster_id for cluster_id, _ in distances[:top_k]]
        
        if self.debug:
            print(f"\n🎯 Clusters más relevantes:")
            for i, (cluster_id, dist) in enumerate(distances[:top_k], 1):
                similarity = (1 - dist) * 100
                print(f"  {i}. Cluster {cluster_id}: {similarity:.2f}% similitud")
        
        return top_clusters