#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path

# Add directories to sys.path to allow config and models imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    VLM_OPTION, EMBEDDING_OPTION, DETECTOR_OPTION, SUBMISSION_TOP_K, RERANK_TOP_K,
    SECONDARY_EMBEDDER_ENABLED,
)
from models.qwen_vlm import QwenVLM
from models.openai_vlm import OpenAIVLM
from models.embedding import QwenVL8BEmbedder, DashScopeCloudEmbedder
from models.siglip_embedder import SigLIPEmbedder
from models.object_detector import ObjectDetector

from search.query_processor import QueryProcessor
from search.hybrid_search import HybridSearcher
from search.reranker import Reranker, rerank_with_tail

def load_vlm():
    if VLM_OPTION == "local":
        return QwenVLM()
    elif VLM_OPTION == "openai":
        return OpenAIVLM()
    else:
        raise ValueError(f"Unknown VLM option: {VLM_OPTION}")

def load_embedder():
    if EMBEDDING_OPTION == "local":
        return QwenVL8BEmbedder()
    elif EMBEDDING_OPTION == "cloud":
        return DashScopeCloudEmbedder()
    else:
        raise ValueError(f"Unknown embedding option: {EMBEDDING_OPTION}")

def load_secondary_embedder():
    """None when disabled - see models/siglip_embedder.py."""
    return SigLIPEmbedder() if SECONDARY_EMBEDDER_ENABLED else None

def main():
    parser = argparse.ArgumentParser(description="Multimedia Retrieval Inference Engine")
    parser.add_argument("--type", type=int, choices=[1, 2, 3], required=True, 
                        help="Query type: 1 (Textual-KIS), 2 (VQA), 3 (Temporal-alignment)")
    parser.add_argument("--query", type=str, required=True, help="Input search text query")
    parser.add_argument("--dataset_dir", type=str, default="./data", 
                        help="Directory containing video frames (used in VQA cropping)")
    args = parser.parse_args()

    # 1. Initialize models
    print("=== Initializing Inference Models ===")
    vlm = load_vlm()
    detector = None
    if args.type == 2:
        # Load detector only for Type 2 VQA crop-reranking
        detector = ObjectDetector(option=DETECTOR_OPTION)
        
    embedder = load_embedder()
    secondary_embedder = load_secondary_embedder()

    # 2. Initialize search and reranking modules
    query_proc = QueryProcessor(vlm_client=vlm)
    searcher = HybridSearcher(embedder=embedder, secondary_embedder=secondary_embedder)
    reranker = Reranker(vlm_client=vlm, detector_client=detector)

    # 3. Query Processing Stage
    print(f"\nProcessing query: '{args.query}'")
    # Generate HyDE hypothetical description
    hyde_query = query_proc.generate_hyde(args.query)
    
    # 4. Candidate Retrieval (Dense + Sparse Hybrid Search)
    print("\nRetrieving candidate frames from Qdrant...")
    # Search on both original query and HyDE query, merge results. Widened to
    # SUBMISSION_TOP_K (100) - the AIC scoring rule rewards a ranked list of
    # up to 100 answers per query (R@1/5/20/50/100), not just the single best.
    query_hits = searcher.search(args.query, top_k=SUBMISSION_TOP_K)
    hyde_hits = searcher.search(hyde_query, top_k=SUBMISSION_TOP_K)
    # Fusionista2.0/VERGE-inspired secondary embedder ensemble - returns []
    # when SECONDARY_EMBEDDER_ENABLED is off, so this is always safe to include.
    secondary_hits = searcher.dense_search_secondary(args.query, top_k=SUBMISSION_TOP_K)

    # Merge candidates from both searches via RRF
    candidates = searcher.merge_rrf(query_hits, hyde_hits, secondary_hits)
    print(f"Retrieved {len(candidates)} unique candidate frames.")

    if not candidates:
        print("No candidates found in Qdrant database.")
        sys.exit(0)

    # Result Diversification: collapse candidates from the same scene down to
    # their best-scoring representative before reranking, so top-K isn't
    # flooded by near-duplicate keyframes from one event (see "Our method" ->
    # Result Diversification). Applied once here since all three query types
    # slice from this same candidate pool below.
    candidates = searcher.diversify_by_scene(candidates, top_k=SUBMISSION_TOP_K)
    print(f"Diversified to {len(candidates)} candidates (deduped by scene).")

    def frame_id_of(payload):
        frame_idx = payload.get("frame_idx")
        if frame_idx is None:
            print(f"[WARN] No frame_idx in payload for '{payload.get('source_file')}' - "
                  f"falling back to timestamp (re-run preprocessing to fix).")
            frame_idx = payload.get("timestamp", 0.0)
        return frame_idx

    # 5. Type-specific Reasoning and Reranking
    if args.type == 1:
        # Type 1: Textual-KIS - Output format: <Tên file video>, <Frame Idx>
        # Only the top RERANK_TOP_K candidates get the (expensive) VLM
        # rerank pass; the rest fill out the ranked list up to
        # SUBMISSION_TOP_K in their original retrieval-rank order.
        ranked = rerank_with_tail(
            lambda c: reranker.rerank_type1(args.query, c), candidates, RERANK_TOP_K, SUBMISSION_TOP_K
        )
        print(f"\n=== FINAL RESULTS ({len(ranked)} ranked answers) ===")
        for rank, item in enumerate(ranked, start=1):
            payload = item["payload"]
            print(f"{rank}. {payload.get('source_file', 'unknown')}, {frame_id_of(payload)}")

    elif args.type == 2:
        # Type 2: Visual Question Answering (VQA)
        # Decompose query
        decomp = query_proc.decompose_query(args.query)
        sub_queries = decomp.get("sub_queries", [args.query])
        print(f"Decomposed query into objects: {sub_queries}")

        # Crop-rerank candidates (same head/tail split as Type 1)
        ranked = rerank_with_tail(
            lambda c: reranker.rerank_type2_vqa(args.query, sub_queries, c, args.dataset_dir),
            candidates, RERANK_TOP_K, SUBMISSION_TOP_K,
        )

        # Answer generation using VLM on the best match only - generating a
        # distinct per-frame answer for up to 100 candidates would be far too
        # expensive, and the question is about the same fact regardless of
        # which candidate location it's paired with, so the single best-effort
        # answer travels with every ranked location guess.
        answer_prompt = f"Answer the following question about this image: {args.query}. Be concise."
        answer = vlm.generate(None, answer_prompt).strip()

        print(f"\n=== FINAL RESULTS ({len(ranked)} ranked answers) ===")
        for rank, item in enumerate(ranked, start=1):
            payload = item["payload"]
            print(f"{rank}. {payload.get('source_file', 'unknown')}, {frame_id_of(payload)}, {answer}")

    elif args.type == 3:
        # Type 3: Temporal-alignment
        # Output format: <Tên file video>, <Frame ID_1>, ..., <Frame ID_N>
        # DANTE-inspired DP alignment (see reranker.py) - decomposes the
        # query into ordered sub-events and aligns them against each
        # candidate video's full frame timeline, not just a single holistic
        # VLM score over the initial candidate hits.
        top_sequences = reranker.rerank_type3_temporal(args.query, candidates[:SUBMISSION_TOP_K], query_proc, searcher)
        print(f"\n=== FINAL RESULTS ({len(top_sequences)} ranked sequences) ===")
        for rank, seq in enumerate(top_sequences, start=1):
            frame_ids_str = ", ".join(str(fid) for fid in seq["frame_ids"])
            print(f"{rank}. {seq['video_name']}, {frame_ids_str}")

if __name__ == "__main__":
    main()
