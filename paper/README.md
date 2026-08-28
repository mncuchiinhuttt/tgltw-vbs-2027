# LaTeX Paper & Live Preview Server

Mục này chứa mã nguồn bài báo LaTeX cho **TGLTW-RMIT VBS 2027** cùng công cụ Live Preview Server.

Tên draft hiện tại: **Evidence-Carrying Live Search: Budgeted Multimodal Retrieval for Interactive Video Browsing**.

## 📂 Cấu trúc thư mục

```text
paper/
├── main.tex           # Mã nguồn LaTeX bài báo chính
├── references.bib     # Danh mục tài liệu tham khảo (BibTeX)
├── figures/           # Thư mục chứa hình ảnh minh họa
├── live_server.py     # Live Server preview PDF và recompile theo yêu cầu
├── compile.sh         # Script compile nhanh qua terminal
├── start_server.sh    # Script khởi động Live Server
└── STRUCTURE_PLAN.md  # Cấu trúc paper và evaluation checklist
```

## Cấu trúc bản draft

`main.tex` hiện được tổ chức theo dạng system paper rút gọn: Introduction; VBS
Setting and System-Level Delta; TGLTW-RMIT Architecture; Task-Specific Live
Pipelines; Evaluation Protocol; Limitations and Conclusion. Novelty được framing
ở ranh giới composition: evidence object có identity ổn định đi xuyên qua
indexing, task routing, precision escalation, grounded VQA và submission/logging;
paper không claim encoder hay rank-fusion rule mới. Các quantitative claims chỉ
được điền sau khi query manifest, ground truth, index snapshot và hardware được
freeze.

`STRUCTURE_PLAN.md` giải thích evaluation tối thiểu cần chạy trước khi thay
placeholder bằng số liệu: Recall/MRR/nDCG, VQA answer và grounding, latency
p50/p95, ablation, failure rate và provenance artifact.

## Figures và screenshot cần bổ sung

`main.tex` đã có các flowchart TikZ cho offline indexing, live query loop và
grounded VQA provenance. Các khung màu đỏ có chữ `SCREENSHOT NEEDED` là vị trí
cần thay bằng ảnh UI thật trước khi submit:

- live search workspace và result grid;
- in-video browsing/evidence inspection;
- grounded VQA result card với answer, frame index/timestamp và preview.

Author block giữ lại ORCID metadata từ paper BTC gốc.

## 🚀 Live Preview Server

Server đang chạy ở port **`8088`** (địa chỉ `http://localhost:8088`).

### Tính năng:
- **Nút Recompile** (hoặc phím tắt **`Ctrl + Enter`** / **`Cmd + Enter`**): Biên dịch nhanh LaTeX và tự động làm mới file PDF hiển thị trên giao diện mà không cần reload trang.
- **Hỗ trợ đa engine**: `latexmk` (tự động chạy đa bước và liên kết BibTeX), `pdflatex`, `xelatex` (hỗ trợ Unicode tiếng Việt đầy đủ).
- **Log / Terminal Viewer**: Bảng hiển thị output biên dịch, tự động bung lên khi có lỗi và tô màu trực quan (Errors, Warnings).
- **Phím tắt `Ctrl + L`**: Bật/tắt thanh Log.
- **Mở tab mới / Tải PDF**: Tải trực tiếp file PDF kết quả về máy.

### Khởi chạy Server thủ công:
```bash
# Chạy ở port mặc định 8088:
python3 paper/live_server.py --port 8088

# Hoặc dùng script:
./paper/start_server.sh 8088
```

## 🌐 Cài đặt Cloudflare Tunnel

Để đưa ra mạng ngoài hoặc truy cập từ xa qua Cloudflare Tunnel:
```bash
# Ví dụ chạy quick tunnel tạm thời:
cloudflared tunnel --url http://localhost:8088
```

## 🛠️ Biên dịch qua dòng lệnh (CLI)

```bash
cd paper
# Cách 1: dùng compile.sh
./compile.sh

# Cách 2: dùng latexmk trực tiếp
latexmk -pdf main.tex

# Cách 3: dùng pdflatex hoặc xelatex
pdflatex -interaction=nonstopmode main.tex
```
