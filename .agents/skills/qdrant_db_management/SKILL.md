---
name: Qdrant Vector Database Management
description: Instructions and guidelines for hosting, managing, querying, and updating Qdrant collections for multimedia search tasks.
---

# Qdrant Database Management Skill

This skill provides directions for configuring, debugging, and interacting with Qdrant databases in local and server environments.

## Connection Setup

Qdrant instances are connected via the `qdrant-client` python SDK.
- Port `6333`: Used for REST API interactions (e.g. creating collections, scrolling, and payload matches).
- Port `6334`: Used for high-throughput gRPC queries.

```python
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
```

## Collection Management

### 1. Vector Configuration
Ensure that the distance metric matches the embedding generator:
- **Cosine Distance** (recommended for CLIP/Qwen-VL space representations).
- Standard dimensionalities:
  - `visual_index`: 4096 dimensions (matching Qwen3-VL-Embedding-8B).
  - `audio_env_index`: 768 dimensions (matching M2D-CLAP with flat_features=True).

```python
from qdrant_client.models import Distance, VectorParams

client.create_collection(
    collection_name="visual_index",
    vectors_config=VectorParams(size=4096, distance=Distance.COSINE)
)
```

## Querying and Filtering

### 1. Modality Filtering
Always specify the `modality` condition (e.g. `"visual"`, `"speech"`, `"ambient_audio"`) in your payload filters to narrow search targets and improve efficiency:

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

query_filter = Filter(
    must=[
        FieldCondition(key="modality", match=MatchValue(value="visual"))
    ]
)
```

### 2. Full-text matches (Sparse Search)
Use `MatchText` filters to perform keyword search on payload elements:
```python
from qdrant_client.models import FieldCondition, MatchText

text_filter = Filter(
    must=[
        FieldCondition(key="text_blob", match=MatchText(text="xe máy"))
    ]
)
```

## Troubleshooting Local Instance

- Check server dashboard: Open [http://localhost:6333/dashboard](http://localhost:6333/dashboard) in a browser.
- Verify running container: `docker ps | grep qdrant`
- View logs: `tail -n 100 qdrant_local.log` or `docker logs qdrant_server`
- Port conflicts: If port `6333` is already in use, check processes with `lsof -i :6333` and stop conflicting services.
