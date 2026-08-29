# Hướng Dẫn Chi Tiết & Giao Thức Đánh Giá Khoa Học (Evaluation Protocol & Ablation Benchmark Manual)

> **Tài liệu tham chiếu học thuật cho VBS 2027 System Paper**
> **Mã số**: `VBS-2027-EVAL-PROTOCOL-v1.0`

---

## 1. Nguyên Tắc Đo Đạc Không Định Kiến (Zero-Bias Principles)

Để đảm bảo kết quả thực nghiệm đạt chuẩn công bố khoa học tại Springer LNCS / ACM MM:
1. **Cô Lập Môi Trường (Environment Isolation)**:
   - Mỗi lần chạy thí nghiệm (Ablation Run) phải được làm sạch session state và cache truy vấn.
   - Các mô hình visual embedder (WeMM-4B, SigLIP) và LLM/VLM phải chạy với fixed random seed (`seed=42`).
2. **Không Rò Rỉ Ground-Truth Vào Quá Trình Retrieval**:
   - Quá trình tìm kiếm (Dense + Sparse + RRF) và Reranker hoàn toàn không biết trước thông tin video/frame ground-truth.
   - Ground-truth chỉ được nạp vào ở bước tính điểm sau khi đã có danh sách $K$ candidates trả về.
3. **Độ Trễ Phải Đo Bằng Hardware Timestamp Chính Xác**:
   - Sử dụng `time.perf_counter()` đo lường ở mức micro-giây ($\mu s$) cho từng phân đoạn:
     $$\text{Total Latency} = t_{\text{HyDE}} + t_{\text{Vector Search}} + t_{\text{RRF Fusion}} + t_{\text{Crop \& VLM Rerank}}$$

---

## 2. Công Thức & Chỉ Số Đánh Giá Chuẩn Quốc Tế

### A. Tác Vụ KIS (Known-Item Search)
1. **Recall@K ($R@K$)**:
   $$R@K = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{rank}(q) \le K)$$
   Đo lường với các ngưỡng chuẩn VBS: $K \in \{1, 5, 10, 20, 50, 100\}$.
2. **Mean Reciprocal Rank (MRR)**:
   $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}(q)}$$

### B. Tác Vụ KIS-C (Conversational Search)
1. **Chỉ Số Mơ Hồ (Ambiguity Index $A$)**:
   $$A = (1 - \lambda) \cdot DVR + \lambda \cdot \left(1.0 - \frac{s_1 - s_2}{s_1}\right)$$
   với $DVR = \frac{|\text{Distinct Videos in Top 10}|}{10}$, $s_1, s_2$ là điểm của top-1 và top-2 candidate ($\lambda=0.5$).
2. **Độ Lợi Thứ Hạng Qua Các Lượt (Turn-by-Turn Rank Gain)**:
   $$\Delta \text{Rank} = \text{rank}_{\text{turn}_i}(q) - \text{rank}_{\text{turn}_{i+1}}(q)$$

### C. Tác Vụ VQA (Visual Question Answering)
1. **Exact Match (EM)**:
   $$\text{EM} = \frac{1}{|Q_{\text{VQA}}|} \sum_{q \in Q_{\text{VQA}}} \mathbb{I}(\text{normalize}(\hat{y}_q) == \text{normalize}(y_q))$$
2. **Fail-Closed Safe Refusal Rate**:
   Tỷ lệ hệ thống trả về đúng `"N/A"` hoặc `"UNKNOWN"` khi frame bị mất hoặc câu hỏi không thể trả lời được từ frame (mục tiêu $100\%$).
3. **Hallucination Rate**:
   Tỷ lệ hệ thống tạo ra đáp án khẳng định sai khi không có bằng chứng thị giác (mục tiêu $0\%$).

### D. Tác Vụ AVS (Ad-hoc Video Search)
1. **Mean Average Precision (mAP)**:
   $$\text{mAP} = \frac{1}{|Q_{\text{AVS}}|} \sum_{q \in Q_{\text{AVS}}} \text{AP}(q)$$
2. **Distinct Video Count at Rank 20 ($D@20$)**:
   Số lượng video độc lập xuất hiện trong top 20 kết quả (tối đa hóa độ phủ cảnh).

---

## 3. Quy Trình Chạy Benchmark Tự Động Hóa (Automated Benchmark Runner)

Khi triển khai thực tế trên server có GPU:
```bash
# 1. Chạy toàn bộ 5 bài Ablation đo đạc số liệu thật
python3 evaluation/run_comprehensive_ablation.py \
    --queries evaluation/eval_queries_real_v3c.json \
    --dataset_dir datasets/v3c \
    --output_dir evaluation/benchmark_real_output \
    --concurrency 8

# 2. Tự động trích xuất bảng LaTeX và cập nhật vào bài báo
python3 evaluation/export_paper_tables.py \
    --input evaluation/benchmark_real_output/summary.json \
    --target paper/main.tex

# 3. Biên dịch bài báo để kiểm tra trực quan
cd paper && ./compile.sh
```
