import numpy as np
from typing import Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, TextIndexParams, TextIndexType, TokenizerType,
    Filter, FieldCondition, MatchValue,
)
from preprocessing.config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, QDRANT_UPSERT_BATCH_SIZE,
    SECONDARY_EMBEDDER_ENABLED, QDRANT_ALLOW_RECREATE,
)

# Name of the secondary (SigLIP) named vector inside "visual_index" - queried
# via Qdrant's `using="siglip"` param in HybridSearcher.dense_search_secondary.
SECONDARY_VECTOR_NAME = "siglip"
SHOT_COLLECTION_NAME = "vbs_shot_index"

class QdrantIndexer:
    """
    Manages connection to Qdrant, initializes collections, and pushes indexes.
    """
    def __init__(self, visual_dim: int = 4096, audio_dim: int = 768, secondary_dim: Optional[int] = None):
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None
        )
        self.batch_size = QDRANT_UPSERT_BATCH_SIZE
        self._visual_buffer = []
        self._audio_buffer = []
        self._shot_buffer = []
        self.secondary_enabled = SECONDARY_EMBEDDER_ENABLED and secondary_dim is not None
        self._init_collections(visual_dim, audio_dim, secondary_dim if self.secondary_enabled else None)

    def _ensure_collection(self, name: str, dim: int, secondary_dim: Optional[int] = None):
        """
        Create the collection if missing, or recreate it if its existing
        vector config doesn't match. When secondary_dim is given, the
        collection uses named vectors ("default" + SECONDARY_VECTOR_NAME)
        instead of a single unnamed one - this is a schema change, so any
        existing single-vector collection gets recreated (old points are
        lost; re-run preprocessing over the whole dataset after enabling
        SECONDARY_EMBEDDER_ENABLED).
        """
        recreate = False
        collection_exists = self.client.collection_exists(name)
        if collection_exists:
            try:
                info = self.client.get_collection(name)
                vectors_config = info.config.params.vectors
                is_named = isinstance(vectors_config, dict) and not hasattr(vectors_config, "size")

                if secondary_dim is not None and not is_named:
                    print(f"Collection '{name}' exists as single-vector but secondary embedder is "
                          f"enabled. Recreating with named vectors...")
                    recreate = True
                elif secondary_dim is None and is_named:
                    print(f"Collection '{name}' exists as named-vector but secondary embedder is "
                          f"disabled. Recreating with a single vector...")
                    recreate = True
                elif is_named:
                    current_size = vectors_config.get("default").size if vectors_config.get("default") else None
                    if current_size != dim:
                        print(f"Collection '{name}' exists but 'default' vector size {current_size} "
                              f"!= {dim}. Recreating...")
                        recreate = True
                else:
                    current_size = vectors_config.size if hasattr(vectors_config, "size") else vectors_config.get("size")
                    if current_size is not None and current_size != dim:
                        print(f"Collection '{name}' exists but has size {current_size} instead of {dim}. Recreating...")
                        recreate = True
            except Exception as e:
                # A timeout/permission error is not evidence of a schema
                # mismatch.  Never destroy an existing collection merely
                # because its metadata could not be read.
                raise RuntimeError(
                    f"Could not verify Qdrant collection '{name}'; refusing to delete it: {e}"
                ) from e
        else:
            recreate = True

        if recreate:
            if collection_exists and not QDRANT_ALLOW_RECREATE:
                raise RuntimeError(
                    f"Qdrant collection '{name}' has an incompatible schema. "
                    "Set QDRANT_ALLOW_RECREATE=true only for an intentional full rebuild."
                )
            if collection_exists:
                self.client.delete_collection(name)
            if secondary_dim is not None:
                print(f"Creating collection '{name}' with named vectors (default={dim}, "
                      f"{SECONDARY_VECTOR_NAME}={secondary_dim})...")
                vectors_config = {
                    "default": VectorParams(size=dim, distance=Distance.COSINE),
                    SECONDARY_VECTOR_NAME: VectorParams(size=secondary_dim, distance=Distance.COSINE),
                }
            else:
                print(f"Creating collection '{name}' with size={dim}...")
                vectors_config = VectorParams(size=dim, distance=Distance.COSINE)
            self.client.create_collection(collection_name=name, vectors_config=vectors_config)

    def _ensure_text_index(self, collection_name: str, field_name: str):
        """Create a multilingual full-text payload index if it doesn't already exist."""
        info = self.client.get_collection(collection_name)
        if field_name in (info.payload_schema or {}):
            return
        print(f"Creating full-text index on '{collection_name}.{field_name}'...")
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.MULTILINGUAL,
                min_token_len=2,
                lowercase=True,
            )
        )

    def _init_collections(self, visual_dim: int = 4096, audio_dim: int = 768, secondary_dim: Optional[int] = None):
        self._ensure_collection("visual_index", visual_dim, secondary_dim)
        self._ensure_collection("audio_env_index", audio_dim)
        # H-EAGLE-lite uses a separate collection so enabling/disabling the
        # optional coarse stage never changes the live frame-index schema.
        self._ensure_collection(SHOT_COLLECTION_NAME, visual_dim)
        self._ensure_text_index("visual_index", "text_blob")
        self._ensure_text_index(SHOT_COLLECTION_NAME, "text_blob")

    def index_visual_point(
        self,
        point_id: str,
        vector: np.ndarray,
        payload: Dict[str, Any],
        secondary_vector: Optional[np.ndarray] = None,
    ):
        """
        Upload point to visual_index collection. Payload format matches
        Notion specifications. When this indexer was constructed with
        secondary_enabled=True, secondary_vector (the SigLIP embedding) is
        stored as the named "siglip" vector alongside the primary one -
        pass None (or omit) to leave that named vector empty for this
        point, e.g. if secondary embedding failed for a single frame.
        """
        if self.secondary_enabled and secondary_vector is not None:
            vector_payload = {
                "default": vector.tolist(),
                SECONDARY_VECTOR_NAME: secondary_vector.tolist(),
            }
        else:
            vector_payload = vector.tolist()

        point = PointStruct(
            id=point_id,
            vector=vector_payload,
            payload=payload
        )
        self._visual_buffer.append(point)
        if len(self._visual_buffer) >= self.batch_size:
            self.flush_visual()

    def index_audio_point(self, point_id: str, vector: np.ndarray, payload: Dict[str, Any]):
        """
        Upload point to audio_env_index collection.
        """
        point = PointStruct(
            id=point_id,
            vector=vector.tolist(),
            payload=payload
        )
        self._audio_buffer.append(point)
        if len(self._audio_buffer) >= self.batch_size:
            self.flush_audio()

    def index_shot_point(self, point_id: str, vector: np.ndarray, payload: Dict[str, Any]):
        """Buffer one H-EAGLE-lite shot aggregate vector."""
        point = PointStruct(id=point_id, vector=vector.tolist(), payload=payload)
        self._shot_buffer.append(point)
        if len(self._shot_buffer) >= self.batch_size:
            self.flush_shots()

    def delete_shots_for_video(self, video_name: str) -> None:
        """Make a shot rebuild idempotent without touching frame points."""
        self.client.delete(
            collection_name=SHOT_COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=video_name))]
            ),
            wait=True,
        )

    def delete_visual_for_video(self, video_name: str) -> None:
        """Explicit opt-in cleanup for a full video reindex."""
        self.client.delete(
            collection_name="visual_index",
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=video_name))]
            ),
            wait=True,
        )

    def flush_visual(self):
        """Upload all buffered visual/speech points in one Qdrant request."""
        if not self._visual_buffer:
            return
        points, self._visual_buffer = self._visual_buffer, []
        self.client.upsert(collection_name="visual_index", points=points)
        print(f"  Flushed {len(points)} visual points to Qdrant.")

    def flush_audio(self):
        """Upload all buffered ambient-audio points in one Qdrant request."""
        if not self._audio_buffer:
            return
        points, self._audio_buffer = self._audio_buffer, []
        self.client.upsert(collection_name="audio_env_index", points=points)
        print(f"  Flushed {len(points)} audio points to Qdrant.")

    def flush_shots(self):
        """Upload buffered H-EAGLE-lite shot aggregates."""
        points, self._shot_buffer = getattr(self, "_shot_buffer", []), []
        if not points:
            return
        self.client.upsert(collection_name=SHOT_COLLECTION_NAME, points=points)
        print(f"  Flushed {len(points)} shot points to Qdrant.")

    def flush(self):
        """Flush every pending collection; call at video and process boundaries."""
        self.flush_visual()
        self.flush_audio()
        self.flush_shots()
