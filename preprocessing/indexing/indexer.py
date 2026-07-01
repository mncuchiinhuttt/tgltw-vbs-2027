import numpy as np
from typing import Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from preprocessing.config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY

class QdrantIndexer:
    """
    Manages connection to Qdrant, initializes collections, and pushes indexes.
    """
    def __init__(self, visual_dim: int = 1536):
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
        )
        self._init_collections(visual_dim)

    def _init_collections(self, visual_dim: int = 1536):
        # 1. Collection for visual index
        recreate = False
        if self.client.collection_exists("visual_index"):
            try:
                info = self.client.get_collection("visual_index")
                vectors_config = info.config.params.vectors
                current_size = None
                if hasattr(vectors_config, "size"):
                    current_size = vectors_config.size
                elif isinstance(vectors_config, dict) and "size" in vectors_config:
                    current_size = vectors_config["size"]
                
                if current_size is not None and current_size != visual_dim:
                    print(f"Collection 'visual_index' exists but has size {current_size} instead of {visual_dim}. Recreating...")
                    self.client.delete_collection("visual_index")
                    recreate = True
            except Exception as e:
                print(f"Warning: Failed to verify collection dimension: {e}. Recreating...")
                self.client.delete_collection("visual_index")
                recreate = True
        else:
            recreate = True
            
        if recreate:
            print(f"Creating collection 'visual_index' with size={visual_dim}...")
            self.client.create_collection(
                collection_name="visual_index",
                vectors_config=VectorParams(size=visual_dim, distance=Distance.COSINE)
            )
            
        # 2. Collection for audio index (M2D-CLAP space: typically 512 dims)
        if not self.client.collection_exists("audio_env_index"):
            print("Creating collection 'audio_env_index'...")
            self.client.create_collection(
                collection_name="audio_env_index",
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )

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
