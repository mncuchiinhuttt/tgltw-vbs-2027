import numpy as np
from typing import Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from preprocessing.config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY

class QdrantIndexer:
    """
    Manages connection to Qdrant, initializes collections, and pushes indexes.
    """
    def __init__(self, visual_dim: int = 4096, audio_dim: int = 768):
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
        )
        self._init_collections(visual_dim, audio_dim)

    def _ensure_collection(self, name: str, dim: int):
        """Create the collection if missing, or recreate it if its existing vector size doesn't match dim."""
        recreate = False
        if self.client.collection_exists(name):
            try:
                info = self.client.get_collection(name)
                vectors_config = info.config.params.vectors
                current_size = None
                if hasattr(vectors_config, "size"):
                    current_size = vectors_config.size
                elif isinstance(vectors_config, dict) and "size" in vectors_config:
                    current_size = vectors_config["size"]

                if current_size is not None and current_size != dim:
                    print(f"Collection '{name}' exists but has size {current_size} instead of {dim}. Recreating...")
                    self.client.delete_collection(name)
                    recreate = True
            except Exception as e:
                print(f"Warning: Failed to verify collection dimension: {e}. Recreating...")
                self.client.delete_collection(name)
                recreate = True
        else:
            recreate = True

        if recreate:
            print(f"Creating collection '{name}' with size={dim}...")
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def _init_collections(self, visual_dim: int = 4096, audio_dim: int = 768):
        self._ensure_collection("visual_index", visual_dim)
        self._ensure_collection("audio_env_index", audio_dim)

    def index_visual_point(self, point_id: str, vector: np.ndarray, payload: Dict[str, Any]):
        """
        Upload point to visual_index collection.
        Payload format matches Notion specifications.
        """
        point = PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload=payload
        )
        self.client.upsert(
            collection_name="visual_index",
            points=[point]
        )

    def index_audio_point(self, point_id: str, vector: np.ndarray, payload: Dict[str, Any]):
        """
        Upload point to audio_env_index collection.
        """
        point = PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload=payload
        )
        self.client.upsert(
            collection_name="audio_env_index",
            points=[point]
        )
