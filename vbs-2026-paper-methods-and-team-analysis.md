# VBS 2026: Tổng hợp phương pháp, đội thi và phân tích novelty

## Nguồn và phạm vi

- Nguồn: [VBS-2026-978-981-95-6963-2.pdf](/Users/vominhlong/AIC_VBS/tgltw-vbs-2027/VBS-2026-978-981-95-6963-2.pdf).
- Tài liệu có 12 paper về các hệ thống/đội tham gia VBS 2026.
- Báo cáo tập trung vào pipeline, model, index, interaction, novelty và bài học rút ra.
- “Novelty” là phân tích từ nội dung paper, không phải claim chính thức của tác giả.
- Thành phần được mô tả là dự kiến hoặc không nêu rõ model được đánh dấu tương ứng.

## Toàn cảnh nhanh

| Đội/hệ thống | Hướng chính | Model/index nổi bật |
|---|---|---|
| TapesVRy | Panorama 360° và semantic universe | CLIP, hierarchical clustering, LLM, Meta Quest |
| Fusionista2.0 | Tối ưu tốc độ, đa modality, VQA reranking | CLIP-Sig400M, CLIP-ViT-5B, Vintern, faster-whisper, InternVL, GPT-4o |
| U-Cker | Exact search trên hàng triệu keyframe | OpenCLIP coca ViT-L/14, Qwen3-8B, GPU dense matrix |
| VIREO | Hybrid dense/sparse, SSM segmentation, object sketch | Milvus, HNSW, BM25, BLIP, PaddleOCR, Whisper-Turbo |
| SnapMind | LLM agent lập kế hoạch retrieval | OpenCLIP ViT-H/14, Milvus, Elasticsearch, RRF |
| vitrivr Emotion | Retrieval theo cảm xúc | DeepFace, VIT-Face-Expression, emotion classifier, wav2vec |
| MediaMix | Multimedia retrieval trong mixed reality | FERElight, vitrivr-engine, CLIP, DINOv2, t-SNE |
| H-EAGLE | Index phân cấp frame–shot–action | SigLIP, NV-Embed, VLM |
| VERGE | Nhiều modality và domain-specific retrieval | ResNet152, GoogleNet, Qwen2.5-VL, Qwen3-VL, YOLOv12 |
| PraK V4 | Localized query và spatial conjunction | CLIP, Grounded-SAM, Bayesian update, Rocchio/SVM |
| NII-UIT | VQA, answer localization, guided verification | SigLIP, BEIT-3, NVILA |
| Exquisitor | Temporal sequence-chain retrieval | LLM search, relevance feedback, eCP index, RRF |

Dataset chính: V3C, Marine Video Kit (MVK) và LapGynLHE/LHE75. V3C lớn và đa dạng; MVK có cảnh underwater tương đối đồng nhất; LHE75 gồm video phẫu thuật dài và chuyên ngành cao.

---

## 1. TapesVRy — Immersive Panoramic Exploration

**Paper:** “TapesVRy: Immersive Panoramic Exploration in Large-Scale Video Retrieval” — trang bài 159–166.

### Đội thi

- Viet-Tham Huynh — University of Science, VNU-HCM; Vietnam National University, Ho Chi Minh City.
- Nhut-Thanh Le-Hinh — University of Science, VNU-HCM; Vietnam National University, Ho Chi Minh City.
- Thang-Long Nguyen-Ho — Dublin City University.
- Trong-Thuan Nguyen — University of Science, VNU-HCM; Vietnam National University, Ho Chi Minh City.
- Cathal Gurrin — Dublin City University.
- Tam V. Nguyen — University of Dayton.
- Minh-Triet Tran — University of Science, VNU-HCM; Vietnam National University, Ho Chi Minh City.

### Phương pháp và model

1. **Tiền xử lý:** shot boundary detection, phân tích độ đa dạng cảnh, chọn 3–5 keyframe/segment; có thể trích clip 2–3 giây để giữ chuyển động.
2. **Embedding:** keyframe qua CLIP image encoder; title/transcript qua CLIP text encoder. Average các embedding ảnh rồi average với text embedding thành vector video.
3. **Clustering:** hierarchical clustering tạo các “video universes”; LLM đọc metadata và frame đại diện để sinh nhãn cluster.
4. **Tapestry:** chọn frame trong universe, chiếu lên mặt cầu, tối ưu độ phủ/giảm trùng lặp, blending biên, chèn clip ngắn lặp lại.
5. **Interaction:** Meta Quest cho phép xoay nhìn, zoom, point-and-select, mở video/clip gốc và lọc theo universe.

LLM không được nêu checkpoint cụ thể. Phần tham khảo nhắc Meta Llama 3 và Qwen1.5 nhưng không khẳng định model triển khai.

### Novelty

- Mở rộng video tapestry 2D của một video thành panorama 360° cho cả collection.
- Semantic universe giúp nhóm video trước khi browse, giảm clutter.
- VR được dùng như một không gian khám phá và truy hồi chứ không chỉ xem video 360°.

Novelty thiên về interaction và information visualization hơn là một model retrieval mới.

### Hạn chế

- Chưa nêu rõ thuật toán tối ưu vị trí frame trên sphere.
- Chưa có checkpoint LLM, threshold clustering hoặc đánh giá retrieval chi tiết.
- Panorama tạo offline nên chưa hỗ trợ personalization real-time.

---

## 2. Fusionista2.0 — Efficient Retrieval System

**Paper:** “Fusionista2.0: Efficiency Retrieval System for Large-Scale Datasets” — trang bài 167–175.

### Đội thi

- Huy M. Le — MBZUAI; AISIA Research Lab; UIT; Vietnam National University, Ho Chi Minh City.
- Dat Tien Nguyen — MBZUAI; AISIA Research Lab.
- Phuc Binh Nguyen — UIT; Vietnam National University, Ho Chi Minh City.
- Gia Bao Le Tran — UIT; Vietnam National University, Ho Chi Minh City.
- Phu Truong Thien — UIT; Vietnam National University, Ho Chi Minh City.
- Cuong Dinh — UIT; Vietnam National University, Ho Chi Minh City.
- Minh Nguyen — UIT; Vietnam National University, Ho Chi Minh City.
- Nga Nguyen — UIT; Vietnam National University, Ho Chi Minh City.
- Thuy T. N. Nguyen — UIT; Vietnam National University, Ho Chi Minh City.
- Huy Gia Ngo — UIT; Vietnam National University, Ho Chi Minh City.
- Tan Nhat Nguyen — UIT; Vietnam National University, Ho Chi Minh City.
- Binh T. Nguyen — AISIA Research Lab; Ho Chi Minh University of Science; Vietnam National University, Ho Chi Minh City.

### Phương pháp và model

#### Keyframe extraction

Pipeline cũ dùng CLIP-B/32, TransNetV2 và clustering. Pipeline mới dùng ffmpeg:

- Trích xuất intra-coded frame/I-frame.
- Lưu timestamp và frame mapping.
- Giảm GPU/memory cost trên V3C.

#### Text-to-image

Dùng ensemble CLIP-Sig400M và CLIP-ViT-5B. Điểm cuối là 0.7 × score CLIP-Sig400M cộng 0.3 × score CLIP-ViT-5B. Trọng số 0.7 được chọn qua thử nghiệm nội bộ và user study 50 người.

#### OCR, ASR và VQA

- OCR: Vintern-1B-v3.5, fine-tune từ InternVL2.5-1B.
- ASR: faster-whisper thay Whisper.
- Dataset QA nội bộ: 200 video–question, gồm counting, image information extraction và video information extraction.
- Model thử: InternVL-1B-Seq, InternVL-1B-ffn6, InternVL-1B-ffn6-Seq, LLaVA-0.5B-ffn6, SmolVLM-0.5B-ffn6.
- Model chọn: InternVL-1B-ffn6-Seq vì cân bằng accuracy và latency.

#### Reranking

- GPT-4o sinh ba câu hỏi yes/no từ query.
- VideoLLaMA hoặc BLIP-2 trả lời các câu hỏi trên từng ảnh.
- Xếp hạng theo số câu trả lời “yes”.

#### UI

Vite, shadcn/ui, WCAG, virtual scrolling, grouped result, sidebar shortcut, batch operation, multilingual query và conversational VQA.

### Novelty

- Thay pipeline keyframe chính xác nhưng nặng bằng I-frame extraction deterministic.
- Ensemble CLIP có trọng số được kiểm chứng thực nghiệm.
- Dùng model nhỏ dưới 1B cho VQA trong bối cảnh time-critical.
- GPT-4o phân rã query thành câu hỏi nhị phân để rerank các chi tiết nhỏ.

Đây là novelty kiểu system engineering và latency optimization.

### Hạn chế

- GPT-4o/VLM reranking có thể tăng latency.
- I-frame không nhất thiết là frame đại diện tốt nhất.
- Chất lượng phụ thuộc query decomposition.

---

## 3. U-Cker — Web-Based Multimodal Retrieval

**Paper:** “U-Cker at VBS2026: A Web-Based Interactive Video Retrieval System with Multimodal Query Support” — trang bài 176–182.

### Đội thi

- Kazuya Ueki — Meisei University.
- Ryo Muto — Meisei University.
- Takuya Wada — Meisei University.
- Ryota Akaba — Meisei University.
- Guannan Zhang — Meisei University.

### Phương pháp và model

- Keyframe encode offline bằng OpenCLIP.
- Text/image query cùng nằm trong embedding space 768 chiều.
- V3C có hơn 4 triệu keyframe; vector lưu trong GPU memory.
- Exact inner product với toàn bộ keyframe; không dùng FAISS/Milvus ANN.
- Checkpoint: coca_ViT-L-14, pretrained trên mscoco_finetuned_laion2b_s13b_b90k.
- Ma trận V3C: 768 × 4,143,673.
- Query: tối đa 3 text và 3 image input; mỗi input bật/tắt và gán weight.
- Query helper: Qwen3-8B sửa spelling, dịch và paraphrase thành 3 query tiếng Anh.
- Browse: phóng to keyframe, mở video segment, xem keyframe lân cận.

### Novelty

- Exact GPU search ở quy mô hơn 4 triệu keyframe, ưu tiên deterministic ranking và độ tin cậy trong VBS.
- Query assistance đóng gói spell correction, translation và paraphrasing vào một prompt duy nhất.

### Hạn chế

- Exact search phụ thuộc GPU memory.
- Chưa có temporal model cho event nhiều bước.
- General-purpose model có thể yếu trên MVK/LHE.

---

## 4. VIREO

**Paper:** “VIREO @ Video Browser Showdown 2026” — trang bài 183–190.

### Đội thi

- Yu-Tong Cheng — Singapore Management University.
- Phuong-Anh Nguyen — Singapore Management University.
- Kim-Thuy Kha — Ton Duc Thang University.
- Chong-Wah Ngo — Singapore Management University.

### Phương pháp và model

#### Hybrid backend

Milvus hợp nhất dense retrieval với HNSW/ANN, sparse retrieval với Sparse-BM25, metadata filtering và weighted late fusion.

- OCR: PaddleOCR v5.
- ASR: OpenAI Whisper-Turbo.
- Có metadata index.

#### SSM shot segmentation

1. Sample frame dày và encode bằng BLIP.
2. Tạo self-similarity matrix bằng cosine similarity.
3. Tính novelty signal bằng Gaussian checkerboard kernel.
4. Chọn novelty peak và thêm uniform grid làm candidate boundary.
5. Chạy Kernel Temporal Segmentation bằng dynamic programming.
6. Refine boundary ở frame gốc.
7. Merge boundary yếu dựa trên intra-shot/cross-boundary similarity.

Integral table được dùng để tính block average nhanh.

#### Object-sketch

Offline:

- Sample frame trong shot.
- Detector MS-COCO, ví dụ Co-DETR/Co-DINO ViT-L.
- Lấy class, confidence và segmentation mask.
- Chia frame thành grid H × W.
- Mỗi object class–cell là một atomic query.
- Cell score kết hợp probability và IoU.
- Max-pool qua frame của shot.
- Precompute rank list.

Online:

1. Chọn object.
2. Shading các cell theo result volume dự kiến.
3. Chọn cell.
4. Late-fuse rank list.
5. Lặp lại.

UI còn có CLIP, BLIP, OCR, ASR, metadata, temporal query, LLM rewrite và Stable Diffusion visual proxy.

### Novelty

- Milvus unified dense/sparse/filter/fusion backend.
- SSM + KTS thích nghi cho video ngắn và dài.
- Mở rộng color sketch thành object-by-cell sketch.
- Precomputed rank list và recommendation shading giữ latency thấp.

### Hạn chế

- Phụ thuộc detector/segmentation mask.
- SSM có nhiều hyperparameter.
- HNSW không bảo đảm exact ranking.

---

## 5. SnapMind

**Paper:** “From Expert Practices to Intelligent Agents: Autonomy in Interactive Video Retrieval” — trang bài 191–198.

### Đội thi

- Minh-Quan Ho-Le — University of Science, VNU-HCM; Vietnam National University, Ho Chi Minh City.
- Duy-Khang Ho — University of Science, VNU-HCM; Vietnam National University, Ho Chi Minh City.
- Tu V. Ninh — University of Science, VNU-HCM; Vietnam National University, Ho Chi Minh City; Dublin City University.
- Cathal Gurrin — Dublin City University.
- Minh-Triet Tran — University of Science, VNU-HCM; Vietnam National University, Ho Chi Minh City.

### Phương pháp và model

#### Registry và index

Registry có text, image, OCR, color, object và ADL component. Mỗi component có tool ID, modality, score range, calibration, default top-k, default weight và latency cost.

- Dense index: OpenCLIP ViT-H/14 trong Milvus.
- Keyword index: Elasticsearch cho OCR, object, color và ADL.
- Fuzzy matching cho symbolic query.

Mỗi component hỗ trợ Search, Rerank hoặc Filter.

#### LLM Planner

Planner nhận query/context/registry và sinh:

- Chuỗi component.
- Weight và top-k từng bước.
- CombSUM, CombMNZ hoặc RRF.
- Query rewrite, synonym, constraint, negative query.
- Protocol tương tác.

#### Ba mode

- Guide: Planner hướng dẫn; user chạy từng bước.
- Assist: Planner tạo nút; user chọn và chỉnh parameter.
- Auto: user approve toàn plan; hệ thống tự chạy và early stop.

#### Incremental fusion

- Normalize bằng calibration logistic hoặc min–max.
- Item không xuất hiện nhận score 0.
- Fuse vào ranking tích lũy.
- Lưu source contribution từng item.
- Early stopping dựa trên Jaccard top-k và delta NDCG@K.
- Audit trail lưu component, weight, fusion, top-k, ranked list và edit.

### Novelty

Novelty cốt lõi là **LLM agent bounded by registry**:

- LLM lập kế hoạch nhiều bước, không chỉ rewrite query.
- Planner chỉ được gọi component đã đăng ký.
- User vẫn kiểm soát plan.
- Có source attribution, audit trail và reproducibility.
- Ba mức autonomy phục vụ novice/expert.

### Hạn chế

- Chất lượng phụ thuộc registry.
- Planner có thể lập plan hợp lý về ngôn ngữ nhưng không tối ưu retrieval.
- Fusion chưa phải model học end-to-end.

---

## 6. Vitrivr với Emotion-Based Retrieval

**Paper:** “Extending vitrivr-engine with Emotion-Based Retrieval and a Modular User Interface” — trang bài 199–206.

### Đội thi

- Andrina Geller — University of Basel.
- Rahel Arnold — University of Basel.
- Raphael Waltenspül — University of Basel.
- Heiko Schuldt — University of Basel.

### Phương pháp và model

Backend dùng Kotlin, PostgreSQL + pgvector, OpenCLIP và DINOv2 qua external feature server, OpenAPI REST server. Pipeline gồm decoder, retriever, extractor, transformer, aggregator, exporter và resolver.

Emotion modality:

- Facial: DeepFace nhận diện mặt + VIT-Face-Expression phân loại biểu cảm.
- Text: emotion_text_classifier trên OCR/text.
- Speech: wav2vec-english-speech-emotion-recognition trên audio.

Các emotion vector được lưu trong database và truy hồi bằng nearest-neighbor thay vì Boolean filter.

Frontend React modular có CLIP, emotion, OCR, ASR; hỗ trợ kết hợp text/image query với emotion filter.

### Novelty

Đưa **emotion thành một retrieval modality có embedding** thay vì chỉ annotate hoặc filter cứng. Vector riêng từ mặt, text và speech cho phép tìm affective similarity và kết hợp nhiều nguồn cảm xúc.

### Hạn chế

- Emotion phụ thuộc ngữ cảnh.
- Speech model chỉ nêu tiếng Anh.
- Facial emotion, speech emotion và text sentiment có thể mâu thuẫn.

---

## 7. MediaMix

**Paper:** “MediaMix: Multimedia Retrieval with Dual Backend Support and Result Exploration in MR” — trang bài 207–213.

### Đội thi

- Rahel Arnold — University of Basel.
- Anna Pietzak — University of Basel.
- Heiko Schuldt — University of Basel.

### Phương pháp và model

#### Dual backend

FERElight:

- Python-based, nhẹ và nhanh.
- CLIP, OCR, ASR.

vitrivr-engine:

- OCR, ASR, CLIP, DINOv2.
- Modular indexing/query.
- PostgreSQL + pgvector.

#### MR frontend

- Apple Vision Pro.
- Gaze, gesture và keyboard.
- Kết quả trên sphere, floating window hoặc wall-mounted panel.
- Cấu hình số item/sphere.
- Merge sphere của nhiều query.
- Tùy chọn t-SNE để đặt các kết quả tương tự gần nhau.
- Playback, frame navigation và related segment.

### Novelty

- Dual backend giúp chọn giữa lightweight retrieval và multimodal retrieval đầy đủ.
- Sphere tương ứng với query; nhiều sphere có thể merge để so sánh.
- t-SNE biến visualization thành công cụ khám phá quan hệ.
- Floating/wall panel là fallback gần desktop UI.

### Hạn chế

- t-SNE có thể không ổn định.
- MR hardware khó triển khai.
- Hiệu quả phụ thuộc backend.

---

## 8. H-EAGLE — Hierarchical EAGLE

**Paper:** “H-EAGLE: Hierarchical Extension of EAGLE for Multi-level Semantic Video Retrieval” — trang bài 214–221.

### Đội thi

- Thang-Long Nguyen-Ho — Dublin City University.
- Viet-Tham Huynh — University of Science, VNU-HCM; Vietnam National University, Ho Chi Minh City.
- Allie Tran — Dublin City University.
- Minh-Triet Tran — University of Science, VNU-HCM; Vietnam National University, Ho Chi Minh City.
- Cathal Gurrin — Dublin City University.
- Graham Healy — Dublin City University.

### Phương pháp và model

#### Level 1: frame

- Mỗi frame encode bằng SigLIP.
- Dùng cho instance-level và chi tiết nhỏ.

#### Level 2: shot

- Gom frame gần nhau về thời gian và visual similarity.
- Encode shot sequence bằng NV-Embed/NVemb.

#### Level 3: action/event

- VLM sinh mô tả activity/event.
- Alignment tìm frame phù hợp.
- Gom frame thành action sequence.
- Encode bằng NV-Embed.

Paper không nêu checkpoint VLM cho activity; tài liệu tham khảo có GPT-4 nhưng không đủ cơ sở kết luận đây là model triển khai.

#### Search

- K=1: nearest-neighbor trên frame embedding.
- K=2: search shot embedding rồi search frame trong shot.
- K=3: search action embedding rồi search frame trong action group.

### Novelty

Thay flat frame index bằng ba abstraction:

- Frame cho chi tiết.
- Shot cho continuity.
- Action cho narrative semantics.

Level 3 có thể gom các shot không liền kề nhưng thuộc cùng một activity. User chọn vector space phù hợp với query thay vì ép mọi query vào frame-level.

### Hạn chế

- Action grouping phụ thuộc VLM/alignment.
- Chưa mô hình hóa rõ quan hệ nhân quả giữa sub-event.
- Multi-level index làm preprocessing phức tạp.

---

## 9. VERGE

**Paper:** “VERGE in VBS 2026” — trang bài 222–229.

### Đội thi

Tất cả thuộc Information Technologies Institute/Centre for Research and Technology Hellas (CERTH-ITI), Thessaloniki, Greece:

- Nick Pantelidis, Eleni Kosmidou, Damianos Galanopoulos, Dimitris Georgalis.
- Stefanos Pasios, Konstantinos Apostolidis, Andreas Goulas, Maria Pegia.
- Georgios Tsionkis, Konstantinos Gkountakos, Grigorios Kouvrakis, Anastasia Moumtzidou.
- Ilias Gialampoukidis, Stefanos Vrochidis, Vasileios Mezaris, Ioannis Kompatsiaris.

### Phương pháp và model

#### Keyframe

1. TransNetV2 phát hiện shot.
2. DCT feature tạo subshot.
3. Tìm Very Similar Segments.
4. Chọn frame sharp nhất.
5. Encode candidate bằng ResNet152.
6. Giữ frame đủ khác trong embedding space và đủ sharp.
7. Áp dụng temporal spacing và loại transition frame.
8. Sharpness dùng variance of Laplacian.

#### Retrieval

- Visual similarity: GoogleNet final pooling + IVFADC.
- Semantic similarity: cross-modal embedding từ text-to-video.
- Late fusion: text, concept, color, face.

#### Text-to-video

- Offline: năm family VLM tạo shot embedding.
- Online: Retrieval Module encode query; Similarity Module tính score từng family; Learnable Weighting Network fuse; Qwen2.5-VL rerank.
- Năm VLM family không được liệt kê cụ thể.

#### Detection/concept

- Human silhouette: YOLOv12 pretrained MS COCO.
- Head: YOLOv12 thứ hai, fine-tune dataset public tổng hợp.
- Concept: Qwen2.5-VL 7B, khoảng 50 concept, prompt từng concept và parse confidence.
- OCR: PaddleOCR.
- Color: 9 tile, CLD và Lab color, so sánh centroid.

#### VQA

- Qwen3-VL 8B Instruct.
- Xử lý toàn video.
- Sinh answer và timestamp.

#### Marine

- YOLOv12 fine-tune underwater cho fish, shark, coral, diver.
- Fish species detection.
- UISM xếp hạng blur/sharpness.

#### Surgical

- YOLOv12 cho instrument.
- YOLOv12 cho organ với Cholec8K Segmentation.
- ResNet50 phân loại số lượng instrument.
- Foundation model cho surgical phase/action; checkpoint không nêu rõ.

### Novelty

VERGE là **domain-aware ensemble platform**:

- Thay/ghép modality theo dataset.
- Thêm underwater object, surgical phase, instrument và image quality.
- Qwen2.5-VL thay ensemble concept detector.
- Qwen3-VL vừa trả lời VQA vừa grounding timestamp.
- Keyframe kết hợp segmentation, diversity và perceptual quality.

Đây là hệ thống có độ bao phủ modality/domain rộng nhất.

### Hạn chế

- Nhiều model chuyên biệt làm pipeline offline lớn.
- Năm VLM family không nêu cụ thể, khó tái lập.
- Qwen2.5/3-VL có thể tăng latency.

---

## 10. PraK V4

**Paper:** “PraK V4 at the Video Browser Showdown 2026” — trang bài 230–237.

### Đội thi

- Bastian Jäckl — University of Konstanz.
- Benjamin Verner — Charles University, Prague.
- Michael Stroh — University of Konstanz.
- Vojtěch Kloda — Charles University, Prague.
- Ladislav Nagy — Charles University, Prague.
- Oliver Deussen — University of Konstanz.
- Daniel A. Keim — University of Konstanz.
- Jakub Lokoč — Charles University, Prague.

### Phương pháp và model

- TransNetV2 phát hiện shot.
- Agglomerative clustering chọn keyframe.
- CLIP fine-tuned cho image-image retrieval.
- MVK/LHE dùng partition 5 vùng.
- Stateless in-memory service + stateful session backend.
- Bayesian update từ positive keyframe được click.

Localized query:

- Whole-image.
- Static 5-region với IoU.
- Dynamic region từ Grounded-SAM.
- Dynamic with CD: CLIP distance + centroid distance.
- Dynamic with IoU: CLIP distance + IoU.

Spatial conjunction:

- Nhiều box và text query đồng thời.
- Tất cả constraint không gian phải cùng đúng.
- Thử nghiệm trên 212 annotation MVK.

In-video:

- Tùy chỉnh replay speed.
- Scene boundary.
- Nhảy giữa scene.
- Rerank keyframe trong video.

Backend:

- Async backend sau NGINX.
- Replicated stateless service.
- Least-connections load balancing.
- Internal test 25 user song song, dưới 3 giây/query.

Online learning dự kiến kết hợp Bayesian update với Rocchio hoặc incremental SVM.

Layout:

- Top 16 frame giữ rank trong grid 4 cột.
- Các item còn lại dùng FLAS với CLIP feature.
- Group theo video hoặc một keyframe/video.

### Novelty

- Mở rộng localized query từ một vùng thành **spatial conjunction nhiều vùng**.
- Kết hợp static region và dynamic Grounded-SAM.
- Đưa scene boundary/in-video ranking vào giai đoạn sau khi tìm đúng video.
- Hybrid layout vừa giữ ranking top đầu vừa hỗ trợ loại nhanh frame không liên quan.
- Parallel backend giảm bottleneck synchronous.

### Hạn chế

- Grounded-SAM có thể không khớp vùng ngữ nghĩa như “school of fish”.
- Rocchio/SVM được mô tả là kế hoạch tích hợp.
- Nhiều box làm recall giảm nếu một box sai.

---

## 11. NII-UIT

**Paper:** “NII-UIT at VBS2026: Towards Effective Visual Question Answering for Interactive and Multimodal Video Retrieval” — trang bài 238–244.

### Đội thi

- Bao Tran — UIT; Vietnam National University, Ho Chi Minh City.
- Tien Do — UIT; Vietnam National University, Ho Chi Minh City.
- Thanh Duc Ngo — UIT; Vietnam National University, Ho Chi Minh City.
- Duy-Dinh Le — UIT; Vietnam National University, Ho Chi Minh City.
- Shin’ichi Satoh — National Institute of Informatics, Tokyo.

### Phương pháp và model

#### Vision-language

Phiên bản trước dùng CLIP và BEIT-3. Phiên bản 2026 thay CLIP bằng SigLIP để tăng semantic consistency và fine-grained grounding. Paper không nói rõ BEIT-3 đã bỏ hoàn toàn hay vẫn giữ.

#### Answer Span Prediction

1. Chia video thành temporal unit ngắn.
2. Sinh dense caption và transcript.
3. Đưa question vào LVLM NVILA.
4. Align question với caption timeline.
5. Chọn temporal hotspot.
6. Hiển thị hotspot để kiểm tra.

#### Candidate Answer Suggestion

- Lấy caption và transcript trong hotspot.
- Align text cue với frame.
- Kết hợp question, text và visual evidence.
- Sinh nhiều candidate answer.
- User xác minh/chọn đáp án.

Model sinh candidate cụ thể không được nêu rõ.

#### In-video retrieval

- Text retrieval trên keyframe.
- Image similarity.
- Object filter.
- OCR.
- ASR.
- Hiển thị candidate frame trên timeline.

#### Tối ưu/UI

- Profiling, xóa duplicate computation và branch không cần.
- Cache repeated/paraphrased query.
- Timeline hotspot + preview keyframe.
- Transcript/caption đồng bộ playback.
- Candidate answer panel.

### Novelty

NII-UIT chuyển VQA thành:

1. **Locate:** Answer Span Prediction.
2. **Suggest:** Candidate Answer Suggestion.
3. **Verify:** User kiểm tra evidence.

Novelty nằm ở việc tối ưu cho answer-bearing moment, không chỉ retrieval video-level hoặc frame-level.

### Hạn chế

- Caption/transcript sai làm hotspot sai.
- Candidate answer có thể hợp lý nhưng sai nếu thiếu grounding.
- Captioning/NVILA có thể tốn chi phí trên video dài.

---

## 12. Exquisitor

**Paper:** “Exquisitor at the Video Browser Showdown 2026: Temporal Queries Revisited” — trang bài 245–251.

### Đội thi

- Omar Shahbaz Khan — IT University of Copenhagen.
- Ujjwal Sharma — University of Amsterdam.
- Gonçalo Marcelino — University of Amsterdam.
- Stevan Rudinac — University of Amsterdam.
- Björn Þór Jónsson — Reykjavik University.

### Phương pháp và model

Exquisitor kết hợp conversational search bằng LLM, relevance feedback, metadata filtering, text search, grouped browsing và in-video search. Paper không nêu checkpoint LLM.

#### eCP index

Phiên bản trước dùng Product Quantization để giảm memory nhưng ảnh hưởng accuracy. Phiên bản mới dùng disk-based eCP index, giữ uncompressed embedding và tránh accuracy loss do PQ.

#### Temporal sequence-chain

Với q1, q2, ..., qn:

1. Chạy q1, lấy top r=1000 segment.
2. Thu thập segment nằm sau kết quả q1.
3. Chạy q2…qn trên candidate pool trong một batch.
4. Với mỗi video, xây chain theo thứ tự thời gian.
5. Ưu tiên chain dài nhất.
6. Nếu bằng nhau, dùng RRF.

Thông số:

- RRF smoothing k=60.
- Query weight wi=1.
- Tối đa v=50 video khác nhau.

#### Loại chain trùng

- Dùng IoU giữa hai chain.
- Nếu IoU > phi, giữ chain dài hơn.
- Nếu cùng độ dài, giữ chain có RRF cao hơn.

#### In-video search

User mở video summary panel, xem keyframe theo trình tự, nhập query nội bộ và rerank segment trong video.

### Novelty

- Temporal query không còn là giao các result set độc lập.
- Sequence chain trực tiếp theo thứ tự thời gian.
- Query trước giới hạn temporal neighborhood của query sau.
- Longest chain được ưu tiên.
- RRF kết hợp rank từng hop mà không cần calibration score.
- eCP thay PQ để giữ accuracy khi giảm memory pressure.

### Hạn chế

- Chain phụ thuộc q1; q1 sai thì candidate pool sau sai.
- Greedy chọn segment từng bước có thể bỏ qua chain toàn cục tốt hơn.
- IoU threshold cần điều chỉnh theo video.

---

# Phần rút ra từ toàn bộ các method

## 1. Flat frame retrieval không còn đủ

H-EAGLE thêm shot/action; Exquisitor xây temporal chain; NII-UIT tìm answer span; PraK hỗ trợ in-video query; VIREO cải thiện shot segmentation.

**Rút ra:** nên có ít nhất ba đơn vị truy hồi: frame, shot/segment và video/event. Frame tìm chi tiết; shot giữ continuity; event xử lý query nhiều bước.

## 2. Tách offline intelligence và online interaction

Offline nên làm keyframe, embedding, OCR/ASR, object, caption, shot/action grouping và precomputed query/index. Online nên làm query encoding, search/fusion, rerank top-N nhỏ, hiển thị và feedback.

**Rút ra:** model lớn nên dùng offline hoặc chỉ rerank candidate set nhỏ.

## 3. Không có một model duy nhất cho mọi task

- CLIP/OpenCLIP/SigLIP: cross-modal retrieval.
- BLIP: visual embedding và SSM.
- Qwen/VILA/InternVL: VQA/reasoning.
- YOLOv12/Co-DETR: object.
- PaddleOCR/Vintern: text.
- Whisper/faster-whisper: speech.
- ResNet/GoogleNet/DINOv2: visual feature.

**Rút ra:** nên có model registry và modality registry thay vì hard-code một backbone.

## 4. Fusion là trung tâm

- Weighted score fusion: Fusionista, VIREO.
- Rank fusion: SnapMind, Exquisitor.
- Learned fusion: VERGE.
- Query decomposition yes/no: Fusionista.
- LLM orchestration: SnapMind.
- Temporal chain: Exquisitor.

**Rút ra:** lưu score/rank riêng từng modality và hỗ trợ nhiều fusion mode; không merge thành score không giải thích được.

## 5. Exact search và ANN bổ sung cho nhau

- U-Cker dùng exact inner product trên GPU.
- VIREO dùng HNSW.
- Milvus hỗ trợ ANN, BM25 và filter.

**Rút ra:** dataset nhỏ/query chính xác dùng exact; dataset lớn dùng ANN; candidate quan trọng có thể exact rerank trên subset.

## 6. Domain adaptation là bắt buộc

- V3C: CLIP/SigLIP + OCR/ASR/object.
- MVK: underwater detector, fish species, blur/sharpness.
- LHE: instrument, organ, phase/action.

**Rút ra:** cần dataset-aware configuration cho sampling rate, shot granularity, detector, query template và modality weight.

## 7. VQA cần pipeline riêng

Pipeline nên là:

video retrieval → in-video localization → multimodal evidence → answer candidates → human verification

Không nên đưa toàn bộ video dài vào VLM ở mọi query.

## 8. Human-in-the-loop là một phần của method

SnapMind cho chỉnh plan/weight; PraK dùng positive feedback/localized box; Exquisitor dùng relevance feedback; U-Cker dùng query rewrite; TapesVRy/MediaMix dùng spatial exploration; NII-UIT dùng guided verification.

**Rút ra:** feedback nên dùng để rerank và logging, không chỉ thay đổi UI.

## 9. Explainability là lợi thế thực chiến

Mỗi kết quả nên giải thích được:

- Modality nào tìm thấy.
- Score/rank từng modality.
- Object/OCR/ASR evidence.
- Timestamp/keyframe hỗ trợ.
- Vì sao kết quả được rerank.

SnapMind và NII-UIT là hai ví dụ rõ nhất.

## 10. Novelty trong VBS thường là novelty hệ thống

Phần lớn paper không đề xuất foundation model mới. Novelty thường nằm ở tổ chức index, fusion, model nhỏ trong latency budget, domain-specific model, interaction và kết nối retrieval với verification.

**Rút ra:** hệ thống cạnh tranh cần tối ưu đồng thời offline indexing, online search, UI, feedback, logging và DRES integration.

---

# Gợi ý ưu tiên áp dụng cho hệ thống VBS hiện tại

## Ưu tiên 1 — Hybrid retrieval có giải thích

Kết hợp dense embedding, sparse OCR/ASR/metadata, object filter, weighted score fusion/RRF và lưu score riêng từng nhánh.

## Ưu tiên 2 — In-video search và temporal window

Sau khi xác định đúng video, search lại trong video, hiển thị keyframe theo timeline, hỗ trợ query A trước query B và sequence-chain cho query nhiều event.

## Ưu tiên 3 — VQA guided verification

1. Retrieve candidate video.
2. Lấy hotspot bằng caption/transcript/OCR/ASR.
3. Rerank frame trong hotspot.
4. VLM trả candidate answer kèm evidence.
5. User xác minh trước khi submit.

## Ưu tiên 4 — Domain-aware routing

- V3C: CLIP/SigLIP + OCR/ASR/object.
- MVK: underwater detector, fish species, blur/sharpness.
- LHE: instrument, organ, phase/action.

Không nên dùng cùng threshold và sampling strategy cho mọi dataset.

## Ưu tiên 5 — Feedback và operator UX

- Positive/negative feedback.
- Query-by-example.
- Query rewrite.
- In-video browse.
- Group theo video.
- Timestamp và evidence rõ ràng.
- Logging mọi interaction để đo latency, recall và hành vi operator.

## Ưu tiên 6 — Routing theo latency

- Exact search cho subset nhỏ/top candidate.
- ANN cho search quy mô lớn.
- Model lớn chỉ rerank top-N.
- Model nhỏ chạy online cho VQA thường gặp.
- Precompute mọi feature có thể precompute.

## Kiến trúc tổng hợp đề xuất

dataset-aware preprocessing  
→ frame + shot + event index  
→ dense + sparse + object/OCR/ASR retrieval  
→ fusion hoặc LLM-planned multi-step retrieval  
→ in-video temporal localization  
→ VLM reranking/VQA evidence  
→ human verification + feedback + DRES submission

Ba ý tưởng đáng ưu tiên nhất:

1. **H-EAGLE:** index phân cấp để không tìm mọi query ở frame level.
2. **Exquisitor/NII-UIT:** temporal localization và guided verification cho query nhiều bước/VQA.
3. **SnapMind/VIREO:** modality registry, fusion có giải thích và precomputed interaction để giữ latency thấp.

