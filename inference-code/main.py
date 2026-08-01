#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path

# Add directories to sys.path to allow config and models imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

from config import VLM_OPTION, EMBEDDING_OPTION, DETECTOR_OPTION
from models.qwen_vlm import QwenVLM
from models.openai_vlm import OpenAIVLM
from models.embedding import QwenVL8BEmbedder, DashScopeCloudEmbedder
from models.object_detector import ObjectDetector

from search.query_processor import QueryProcessor
from search.hybrid_search import HybridSearcher
from search.reranker import Reranker

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
    
    # 2. Initialize search and reranking modules
    query_proc = QueryProcessor(vlm_client=vlm)
    searcher = HybridSearcher(embedder=embedder)
    reranker = Reranker(vlm_client=vlm, detector_client=detector)

    # 3. Query Processing Stage
    print(f"\nProcessing query: '{args.query}'")
    # Generate HyDE hypothetical description
    hyde_query = query_proc.generate_hyde(args.query)
    
    # 4. Candidate Retrieval (Dense + Sparse Hybrid Search)
    print("\nRetrieving candidate frames from Qdrant...")
    # Search on both original query and HyDE query, merge results
    query_hits = searcher.search(args.query, top_k=15)
    hyde_hits = searcher.search(hyde_query, top_k=15)
    
    # Merge candidates from both searches via RRF
    candidates = searcher.merge_rrf(query_hits, hyde_hits)
    print(f"Retrieved {len(candidates)} unique candidate frames.")

    if not candidates:
        print("No candidates found in Qdrant database.")
        sys.exit(0)

    # Result Diversification: collapse candidates from the same scene down to
    # their best-scoring representative before reranking, so top-K isn't
    # flooded by near-duplicate keyframes from one event (see "Our method" ->
    # Result Diversification). Applied once here since all three query types
    # slice from this same candidate pool below.
    candidates = searcher.diversify_by_scene(candidates, top_k=20)
    print(f"Diversified to {len(candidates)} candidates (deduped by scene).")

    # 5. Type-specific Reasoning and Reranking
    if args.type == 1:
        # Type 1: Textual-KIS
        # Output format: <Tên file video>, <Frame Idx>
        top_candidates = reranker.rerank_type1(args.query, candidates[:10])
        if top_candidates:
            best = top_candidates[0]
            payload = best["payload"]
            video_name = payload.get("source_file", "unknown")
            frame_idx = payload.get("frame_idx")
            if frame_idx is None:
                print(f"[WARN] No frame_idx in payload for '{video_name}' - "
                      f"falling back to timestamp (re-run preprocessing to fix).")
                frame_idx = payload.get("timestamp", 0.0)
            print("\n=== FINAL RESULT ===")
            print(f"{video_name}, {frame_idx}")
            
    elif args.type == 2:
        # Type 2: Visual Question Answering (VQA)
        # Decompose query
        decomp = query_proc.decompose_query(args.query)
        sub_queries = decomp.get("sub_queries", [args.query])
        print(f"Decomposed query into objects: {sub_queries}")
        
        # Crop-rerank candidates
        top_candidates = reranker.rerank_type2_vqa(args.query, sub_queries, candidates[:10], args.dataset_dir)
        if top_candidates:
            best = top_candidates[0]
            payload = best["payload"]
            video_name = payload.get("source_file", "unknown")
            frame_idx = payload.get("frame_idx")
            if frame_idx is None:
                print(f"[WARN] No frame_idx in payload for '{video_name}' - "
                      f"falling back to timestamp (re-run preprocessing to fix).")
                frame_idx = payload.get("timestamp", 0.0)

            # Answer generation using VLM on best match
            answer_prompt = f"Answer the following question about this image: {args.query}. Be concise."
            # In a real setup we load the best crop/full image. Here we call VLM with empty image or load if present
            answer = vlm.generate(None, answer_prompt).strip()

            print("\n=== FINAL RESULT ===")
            print(f"{video_name}, {frame_idx}, {answer}")
            
    elif args.type == 3:
        # Type 3: Temporal-alignment
        # Output format: <Tên file video>, <Frame ID_1>, ..., <Frame ID_N>
        top_sequences = reranker.rerank_type3_temporal(args.query, candidates[:20])
        if top_sequences:
            best = top_sequences[0]
            video_name = best["video_name"]
            frame_ids = best["frame_ids"]
            
            # Format output sequence
            frame_ids_str = ", ".join([str(fid) for fid in frame_ids])
            print("\n=== FINAL RESULT ===")
            print(f"{video_name}, {frame_ids_str}")

if __name__ == "__main__":
    main()
