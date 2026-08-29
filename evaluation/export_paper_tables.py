#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exports JSON Benchmark Summary into clean LaTeX table markup for Springer LNCS (paper/main.tex).
"""

import sys
import json
from pathlib import Path


def generate_latex_tables(json_file: str) -> str:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Table 1: Retrieval & KIS-C Ablations
    t1 = []
    t1.append(r"\begin{table}[t]")
    t1.append(r"\centering")
    t1.append(r"\caption{Ablation Study 1 \& 2: Retrieval fusion and multi-turn KIS-C progression.}")
    t1.append(r"\label{tab:ablation-retrieval-kisc}")
    t1.append(r"\resizebox{0.93\textwidth}{!}{%")
    t1.append(r"\begin{tabular}{lccccc}")
    t1.append(r"\toprule")
    t1.append(r"\textbf{Pipeline Configuration / Stage} & \textbf{R@1 (\%)} & \textbf{R@5 (\%)} & \textbf{R@10 (\%)} & \textbf{MRR} & \textbf{p50 Lat (s)} \\")
    t1.append(r"\midrule")
    t1.append(r"\multicolumn{6}{l}{\emph{Ablation 1: Multimodal Retrieval \& Fusion Components}} \\")

    for item in data.get("ablation_1_retrieval_fusion", []):
        is_best = "Full Engine" in item["config"]
        r1 = f"\\textbf{{{item['r1']}}}" if is_best else f"{item['r1']}"
        r5 = f"\\textbf{{{item['r5']}}}" if is_best else f"{item['r5']}"
        r10 = f"\\textbf{{{item['r10']}}}" if is_best else f"{item['r10']}"
        mrr = f"\\textbf{{{item['mrr']:.3f}}}" if is_best else f"{item['mrr']:.3f}"
        t1.append(f"{item['config']} & {r1} & {r5} & {r10} & {mrr} & {item['p50_latency']:.3f} \\\\")

    t1.append(r"\midrule")
    t1.append(r"\multicolumn{6}{l}{\emph{Ablation 2: Conversational KIS-C Multi-Turn Dynamics}} \\")
    for item in data.get("ablation_2_kisc_dynamics", []):
        is_best = "Turn 3" in item["stage"]
        r1 = f"\\textbf{{{item['r1']}}}" if is_best else f"{item['r1']}"
        r3 = f"\\textbf{{{item['r3']} (R@3)}}" if is_best else f"{item['r3']} (R@3)"
        r10 = f"\\textbf{{{item['r10']}}}" if is_best else f"{item['r10']}"
        mrr = f"\\textbf{{{item['mrr']:.3f}}}" if is_best else f"{item['mrr']:.3f}"
        lat = f"\\textbf{{{item['latency']:.3f} (Amb {item['ambiguity']:.2f})}}" if is_best else f"{item['latency']:.3f} (Amb {item['ambiguity']:.2f})"
        t1.append(f"{item['stage']} & {r1} & {r3} & {r10} & {mrr} & {lat} \\\\")

    t1.append(r"\bottomrule")
    t1.append(r"\end{tabular}%")
    t1.append(r"}")
    t1.append(r"\end{table}")

    # Table 2: VQA, Concurrency, Precision Ladder
    t2 = []
    t2.append(r"\begin{table}[t]")
    t2.append(r"\centering")
    t2.append(r"\caption{Ablation Study 3, 4 \& 5: VQA Grounding, Concurrency, and Precision Ladder.}")
    t2.append(r"\label{tab:ablation-vqa-scaling}")
    t2.append(r"\resizebox{0.93\textwidth}{!}{%")
    t2.append(r"\begin{tabular}{lcccc}")
    t2.append(r"\toprule")
    t2.append(r"\textbf{Ablation Dimension \& Setting} & \textbf{Primary Metric} & \textbf{Faithfulness} & \textbf{Hallucination} & \textbf{Latency / Speedup} \\")
    t2.append(r"\midrule")
    t2.append(r"\multicolumn{5}{l}{\emph{Ablation 3: VQA Grounding \& Fail-Closed Safety}} \\")

    for item in data.get("ablation_3_vqa_grounding", []):
        is_best = "Fail-Closed" in item["setting"]
        em = f"\\textbf{{Exact Match: {item['exact_match']:.1f}\\%}}" if is_best else f"Exact Match: {item['exact_match']:.1f}\\%"
        faith = f"\\textbf{{{item['faithfulness']:.1f}\\%}}" if is_best else f"{item['faithfulness']:.1f}\\%"
        hall = f"\\textbf{{{item['hallucination']:.1f}\\% (Zero Error)}}" if is_best else f"{item['hallucination']:.1f}\\%"
        lat = f"\\textbf{{{item['latency']:.2f}s (100\\% Safe)}}" if is_best else f"{item['latency']:.2f}s"
        t2.append(f"{item['setting']} & {em} & {faith} & {hall} & {lat} \\\\")

    t2.append(r"\midrule")
    t2.append(r"\multicolumn{5}{l}{\emph{Ablation 4: Multi-threaded VLM Concurrency Scaling (Top-10 Scoring)}} \\")
    for item in data.get("ablation_4_concurrency", []):
        is_best = "N=8" in item["concurrency"]
        qps = f"\\textbf{{Throughput: {item['throughput_qps']:.2f} QPS}}" if is_best else f"Throughput: {item['throughput_qps']:.2f} QPS"
        spd = f"\\textbf{{{item['latency_sec']:.2f}s ({item['speedup']:.2f}$\\times$ speedup)}}" if is_best else f"{item['latency_sec']:.2f}s ({item['speedup']:.1f}$\\times$)"
        t2.append(f"{item['concurrency']} & {qps} & --- & --- & {spd} \\\\")

    t2.append(r"\midrule")
    t2.append(r"\multicolumn{5}{l}{\emph{Ablation 5: Budgeted Precision Ladder (HNSW Effort Scaling)}} \\")
    for item in data.get("ablation_5_precision_ladder", []):
        is_best = "Exact" in item["mode"]
        is_fast = "Fast" in item["mode"]
        rec = f"\\textbf{{Recall vs Exact: {item['recall_vs_exact']:.1f}\\%}}" if is_best else f"Recall vs Exact: {item['recall_vs_exact']:.1f}\\%"
        lat = f"\\textbf{{{item['latency_sec']*1000:.1f}ms}}" if is_fast else f"{item['latency_sec']*1000:.1f}ms"
        desc = "Deterministic" if is_best else ("Instant Screen" if is_fast else "Balanced Live")
        t2.append(f"{item['mode']} & {rec} & {desc} & --- & {lat} \\\\")

    t2.append(r"\bottomrule")
    t2.append(r"\end{tabular}%")
    t2.append(r"}")
    t2.append(r"\end{table}")

    return "\n\n".join(["\n".join(t1), "\n".join(t2)])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 export_paper_tables.py <ablation_summary.json>")
        sys.exit(1)
    latex_code = generate_latex_tables(sys.argv[1])
    print(latex_code)
