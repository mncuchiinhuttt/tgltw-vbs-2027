#!/usr/bin/env python3
import os
import sys
import json
import argparse
import csv
import cv2
from pathlib import Path
from PIL import Image

# Add directories to sys.path
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
    parser = argparse.ArgumentParser(description="Batch Multimedia Retrieval Inference Engine")
    parser.add_argument("--query_file", type=str, default="../queries/queries.json", 
                        help="Path to JSON file containing list of queries")
    parser.add_argument("--output_dir", type=str, default="../queries/", 
                        help="Directory to save the query results")
    parser.add_argument("--dataset_dir", type=str, default="../datasets", 
                        help="Directory containing video frames (used in VQA cropping)")
    args = parser.parse_args()

    # Load queries
    if not os.path.exists(args.query_file):
        print(f"Error: Query file not found at {args.query_file}")
        sys.exit(1)

    try:
        with open(args.query_file, "r") as f:
            queries = json.load(f)
    except Exception as e:
        print(f"Error reading query file: {e}")
        sys.exit(1)

    print(f"Loaded {len(queries)} queries for batch execution.")

    # 1. Initialize models
    print("=== Initializing Inference Models ===")
    vlm = load_vlm()
    embedder = load_embedder()
    
    # Load detector lazily only if any Type 2 query exists
    detector = None
    if any(q.get("type") == 2 for q in queries):
        detector = ObjectDetector(option=DETECTOR_OPTION)
        
    query_proc = QueryProcessor(vlm_client=vlm)
    searcher = HybridSearcher(embedder=embedder)
    reranker = Reranker(vlm_client=vlm, detector_client=detector)

    results = []

    # 2. Run queries
    for idx, q_info in enumerate(queries):
        q_type = q_info.get("type", 1)
        q_text = q_info.get("query", "")
        print(f"\n[{idx+1}/{len(queries)}] Processing Type {q_type} Query: '{q_text}'")

        if not q_text:
            continue

        # Generate HyDE description
        hyde_query = query_proc.generate_hyde(q_text)
        
        # Dense + Sparse Hybrid Search
        query_hits = searcher.search(q_text, top_k=15)
        hyde_hits = searcher.search(hyde_query, top_k=15)
        candidates = searcher.merge_rrf(query_hits, hyde_hits)

        output_data = {
            "query": q_text,
            "type": q_type,
            "result": "N/A"
        }

        if not candidates:
            print("No candidates found.")
            results.append(output_data)
            continue

        # Type Reranking
        if q_type == 1:
            top_candidates = reranker.rerank_type1(q_text, candidates[:10])
            if top_candidates:
                best = top_candidates[0]
                payload = best["payload"]
                video_name = payload.get("source_file", "unknown")
                timestamp = payload.get("timestamp", 0.0)
                output_data["result"] = f"{video_name}, {timestamp:.2f}"

        elif q_type == 2:
            decomp = query_proc.decompose_query(q_text)
            sub_queries = decomp.get("sub_queries", [q_text])
            top_candidates = reranker.rerank_type2_vqa(q_text, sub_queries, candidates[:10], args.dataset_dir)
            if top_candidates:
                best = top_candidates[0]
                payload = best["payload"]
                video_name = payload.get("source_file", "unknown")
                timestamp = payload.get("timestamp", 0.0)
                
                # Answer generation
                answer_prompt = f"Answer the following question about this image: {q_text}. Be concise."
                
                # Extract frame from video or load image if available
                frame_img = None
                frame_path = os.path.join(args.dataset_dir, video_name)
                if os.path.exists(frame_path):
                    if frame_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                        frame_img = Image.open(frame_path).convert("RGB")
                    else:
                        # Extract frame from video
                        cap = cv2.VideoCapture(frame_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps > 0:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
                            ret, frame = cap.read()
                            if ret:
                                frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        cap.release()
                
                answer = vlm.generate(frame_img, answer_prompt).strip()
                output_data["result"] = f"{video_name}, {timestamp:.2f}, {answer}"

        elif q_type == 3:
            top_sequences = reranker.rerank_type3_temporal(q_text, candidates[:20])
            if top_sequences:
                best = top_sequences[0]
                video_name = best["video_name"]
                frame_ids = best["frame_ids"]
                frame_ids_str = ", ".join([str(fid) for fid in frame_ids])
                output_data["result"] = f"{video_name}, {frame_ids_str}"

        print(f"Result: {output_data['result']}")
        results.append(output_data)

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)
    json_out = os.path.join(args.output_dir, "batch_results.json")
    csv_out = os.path.join(args.output_dir, "batch_results.csv")

    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)

    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Query Index", "Query Type", "Query String", "Result Output"])
        for idx, res in enumerate(results):
            writer.writerow([idx + 1, res["type"], res["query"], res["result"]])

    print(f"\nBatch processing finished successfully!")
    print(f"Results saved to:\n  JSON: {json_out}\n  CSV: {csv_out}")

if __name__ == "__main__":
    main()
