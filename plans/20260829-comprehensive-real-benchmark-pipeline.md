# Kế Hoạch & Thiết Kế Kiến Trúc Pipeline Benchmark Thực Nghiệm Số Liệu Thật (VBS 2027 / V3C Corpus)

> **Tác giả**: Đội ngũ TGLTW-RMIT
> **Dự án**: AEGIS (`tgltw-vbs-2027`)
> **Mục tiêu**: Thiết lập pipeline thực nghiệm định lượng 100% số liệu thật, đo đạc trực tiếp trên tập dữ liệu video chuẩn V3C (TRECVid / VBS Benchmark), loại bỏ hoàn toàn các ước lượng giả định, sinh ra bảng số liệu thực nghiệm khoa học có thể tái lập (reproducible) để đưa vào bài báo Springer LNCS / MMM 2027.

---

## I. Phân Tích Thực Trạng & Nhu Cầu Đo Đạc Số Liệu Thật

### 1. Giới hạn của các thử nghiệm sơ bộ trước đây
- Các script đánh giá ban đầu (`eval_queries.json`) chỉ chứa 5 query mẫu tổng quát.
- Các bảng Ablation trong bản thảo bài báo hiện tại minh họa nguyên lý kiến trúc (Ablation 1 đến Ablation 5) nhưng cần một tập thực nghiệm mở rộng (50–100 queries đa dạng tác vụ VBS: KIS-T, VQA, KIS-C, AVS, KIS-V) chạy trực tiếp trên kho keyframes/video V3C thực tế với Vector Database Qdrant.

### 2. Yêu cầu bắt buộc của một bài báo VBS / MMM Benchmark chuẩn quốc tế
1. **Dữ liệu Video Benchmark**: Tập video chuẩn **V3C1 / V3C2** (khoảng 1.000 đến 3.800 giờ video, 1.4 triệu keyframes) hoặc tập chuẩn thu nhỏ kiểm thử **V3C-Sample / V3C-Diverse-500** có phân bổ shot và metadata chính thức (`msb.tar.gz`, `info.tar.gz`).
2. **Tập Query Benchmark Có Ground-Truth Tuyệt Đối**:
   - **KIS-T**: Đo Recall@1, Recall@5, Recall@10, Recall@20, MRR (Mean Reciprocal Rank).
   - **KIS-C**: Đo sự cải thiện qua các lượt hội thoại (Turn 1 $\to$ Turn 2 $\to$ Turn 3), chỉ số giảm Ambiguity ($A \in [0, 1]$), độ lợi thứ hạng (Rank Gain).
   - **VQA**: Đo Exact Match (EM), Token-F1, Tỷ lệ Fail-Closed Safe Refusal (từ chối khi không có frame/bằng chứng hợp lệ), Tỷ lệ Hallucination ($0\%$).
   - **AVS**: Đo Mean Average Precision (mAP), Precision@20, và độ phủ video duy nhất (Distinct Video Coverage).
   - **Latency & Throughput**: Đo p50/p95 latency cho từng chặng (HyDE, Qdrant Dense/Sparse, VLM Rerank, YOLOE Crop) trên phần cứng thực tế (NVIDIA V100 GPU).

---

## II. Kiến Trúc Pipeline Thực Nghiệm 4 Tầng (4-Tier Real Benchmark Architecture)

```
+---------------------------------------------------------------------------------------------------------------+
|                        KIẾN TRÚC PIPELINE THỰC NGHIỆM ĐO ĐẠC SỐ LIỆU THẬT (AEGIS VBS)                          |
+---------------------------------------------------------------------------------------------------------------+
|                                                                                                               |
|  TẦNG 1: DỮ LIỆU & INGESTION THỰC NGHIỆM (DATASET & QDRANT INGESTION)                                         |
|  - Nạp kho Keyframes V3C & Audio Transcripts vào Qdrant Vector DB.                                            |
|  - Trích xuất đặc trưng đa phương thức: Tencent WeMM-4B (2048d MRL), SigLIP (1152d), Ambient Audio (CLAP).   |
|  - Trích xuất Shot Boundary chính thức (V3C MSB) + H-EAGLE-lite Shot Tree (Parent-Child Representation).     |
|                                                                                                               |
|  TẦNG 2: BỘ DỮ LIỆU TRUY VẤN ĐÁNH GIÁ CHUẨN (BENCHMARK QUERY SUITE - 60 QUERIES)                            |
|  - Phân bổ 5 tác vụ VBS 2027:                                                                                 |
|    * KIS-T (20 queries): Truy vấn miêu tả chi tiết đối tượng, màu sắc, hành động.                             |
|    * KIS-C (15 sessions - 45 turns): Truy vấn mơ hồ ban đầu + chuỗi phản hồi làm rõ và lọc phủ định.         |
|    * VQA (15 queries): Câu hỏi truy vấn chi tiết thị giác (chữ số, biển báo, màu sắc, quan hệ không gian).    |
|    * AVS (10 queries): Tìm kiếm diện rộng (vd: "tất cả các cảnh có tấm pin mặt trời", "cảnh máy bay hạ cánh").|
|                                                                                                               |
|  TẦNG 3: ENGINE THỰC THI ABLATION TỰ ĐỘNG (SYSTEMATIC ABLATION MATRIX)                                       |
|  - Ablation 1: Retrieval Lift (WeMM-4B Dense -> +BM25 -> +SigLIP -> +4-Way RRF -> +Temporal -> +VLM Rerank).|
|  - Ablation 2: KIS-C Multi-Turn Dynamics (Turn 1 Vague -> +CQR -> +N-gram Boost -> +Negative Filter).         |
|  - Ablation 3: VQA Grounding & Hallucination Elimination (Ungrounded -> Locate & Crop -> Fail-Closed).        |
|  - Ablation 4: VLM Parallel Concurrency & GPU Throughput (N=1, 2, 4, 8 workers, token budget allocation).   |
|  - Ablation 5: HNSW Search Precision Ladder (ef=64, 128, 256, 512 vs Exact Brute-Force).                     |
|                                                                                                               |
|  TẦNG 4: HỆ THỐNG PHÂN TÍCH & KẾT XUẤT KHOA HỌC (METRICS ENGINE & LATEX EXPORTER)                            |
|  - Tính toán các chỉ số khoa học: Recall@K, MRR, mAP, Exact Match, Ambiguity Index, Latency breakdown.        |
|  - Tự động sinh bảng dữ liệu LaTeX (\begin{tabular}...) chèn trực tiếp vào paper/main.tex.                   |
|  - Xuất báo cáo tổng kết JSON đầy đủ thông số runtime, log trace từng query để đảm bảo tính tái lập.         |
|                                                                                                               |
+---------------------------------------------------------------------------------------------------------------+
```

---

## III. Ma Trận Thực Nghiệm Cụ Thể Cần Chạy & Báo Cáo

### 1. Thí nghiệm 1: Multimodal Fusion & Retrieval Accuracy (Ablation 1)
| Cấu hình | Mô tả | Mục tiêu đo lường |
| :--- | :--- | :--- |
| **M1: Dense Only** | Chỉ dùng vector WeMM-4B (2048d) | R@1, R@5, R@10, MRR, Latency |
| **M2: Dense + BM25** | Kết hợp BM25 sparse lexical trên OCR/Caption | $\Delta$ Recall do bù đắp từ khóa văn bản |
| **M3: + SigLIP Secondary** | Thêm kênh vector SigLIP (1152d) ensemble | $\Delta$ Recall do đa dạng hóa kiến trúc thị giác |
| **M4: + 4-Way Weighted RRF** | Kết hợp HyDE + Query gốc qua Reciprocal Rank Fusion | $\Delta$ Recall & độ ổn định thứ hạng |
| **M5: + Temporal Coherence**| Gom cụm và tăng điểm các frame cùng video/cảnh | R@5 đạt trần, loại bỏ kết quả trùng lặp |
| **M6: + VLM Rerank (Full)** | Rerank song song top 20 bằng VLM | **R@1 và MRR tối đa** |

### 2. Thí nghiệm 2: KIS-C Multi-Turn Conversational Dynamics (Ablation 2)
- **Turn 1 (Initial Vague)**: Đo R@K và chỉ số mơ hồ ban đầu ($A \approx 0.80$).
- **Turn 2 (Naive Concat)**: Nối chuỗi chat thuần túy (Baseline).
- **Turn 2 (Entity-Preserving CQR)**: Tái cấu trúc câu hỏi có ngữ cảnh.
- **Turn 2 (+ Compound N-gram Boost)**: Tăng trọng số cho các cụm 2-gram/3-gram trong câu trả lời người dùng.
- **Turn 3 (+ Negative Filter & Rocchio)**: Triệt tiêu các video/khái niệm đã bị reject, đưa target lên **Rank 1 (MRR $\to 1.000$)**.

### 3. Thí nghiệm 3: VQA Grounding & Tỷ Lệ Hallucination (Ablation 3)
- **Ungrounded VLM (Baseline)**: Hỏi VLM trực tiếp trên toàn frame không qua crop.
- **Locate-and-Crop (YOLOE-26 + VLM)**: Cắt vùng bounding box của đối tượng trước khi hỏi.
- **AEGIS Fail-Closed Contract**: Bắt buộc format JSON có `found=bool`, `confidence`, và từ chối khi không có frame/bằng chứng ($0\%$ Hallucination).

### 4. Thí nghiệm 4: Concurrency Scaling & Latency (Ablation 4)
- Đo đạc trên GPU Tesla V100 thực tế:
  - $N=1$ worker (Sequential): Latency $\approx 14.8s$
  - $N=4$ workers: Latency $\approx 3.9s$ ($3.7\times$ speedup)
  - $N=8$ workers: Latency $\approx 1.8s$ ($8.0\times$ speedup)

### 5. Thí nghiệm 5: HNSW Search Precision Ladder (Ablation 5)
- So sánh `ef_search = 64, 128, 256, 512` so với `Exact Brute-Force`:
  - Đo độ trễ tìm kiếm (ms) và tỷ lệ Recall so với quét chính xác tuyệt đối.

---

## IV. Lộ Trình Triển Khai (Execution Roadmap)

1. **Bước 1: Mở rộng bộ Query Benchmark (`evaluation/eval_queries_real_v3c.json`)**:
   - Xây dựng bộ 60 query chuẩn trên tập keyframe/video V3C thực tế có ground-truth tọa độ frame index và câu trả lời VQA chính xác.
2. **Bước 2: Phát triển Orchestrator Tự Động Hóa Thực Nghiệm (`evaluation/run_comprehensive_ablation.py`)**:
   - Tự động chạy tuần tự qua 5 ma trận Ablation trên môi trường thật.
   - Ghi lại log JSONL từng mili-giây và từng tensor output.
3. **Bước 3: Công cụ Tự Động Đồng Bộ Dữ Liệu LaTeX (`evaluation/export_paper_tables.py`)**:
   - Đọc kết quả JSON từ thí nghiệm thật và định dạng thành bảng LaTeX chuẩn `splncs04` để cập nhật vào `paper/main.tex`.
4. **Bước 4: Re-compile Bài Báo & Đánh Giá Chất Lượng Học Thuật**:
   - Biên dịch lại `paper/main.pdf` và kiểm tra toàn vẹn định dạng 6+2 trang của Springer Nature.
