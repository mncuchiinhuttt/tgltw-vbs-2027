import numpy as np
from typing import Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from preprocessing.config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY

class QdrantIndexer:
    """
    Manages connection to Qdrant, initializes collections, and pushes indexes.
    """
    def __init__(self):
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
        )
        self._init_collections()

    def _init_collections(self):
        # 1. Collection for visual index (Qwen-VL space: 1536 dims)
        if not self.client.collection_exists("visual_index"):
            print("Creating collection 'visual_index'...")
            self.client.create_collection(
                collection_name="visual_index",
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
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
