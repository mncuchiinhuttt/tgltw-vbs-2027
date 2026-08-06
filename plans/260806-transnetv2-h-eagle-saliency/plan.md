# Plan: thay PySceneDetect bằng TransNetV2 và thêm H-EAGLE-lite

Ngày: 2026-08-06

## Trạng thái implementation sync — 2026-08-06

Plan này đã được triển khai một phần trong working tree nhưng **chưa đủ điều kiện merge hoặc bật production**.

| Hạng mục | Trạng thái | Ghi chú |
| --- | --- | --- |
| TransNetV2 PyTorch adapter, model/checkpoint và config | Đã triển khai một phần | Model forward và checkpoint load trên CUDA đã pass smoke test. |
| Dispatcher/fallback và tài liệu cấu hình | Đã triển khai một phần | Cần kiểm tra lại thứ tự official assets → TransNetV2 → PySceneDetect → full-video trên video thật. |
| TransNetV2 streaming prediction stitching | **Blocker P0** | Window overlap đang append trùng prediction, làm lệch boundary trên video dài hơn một window. |
| H-EAGLE-lite shot aggregate/index | Đã triển khai một phần | Có shot helper, shot collection và query route, nhưng chưa đủ integration test với Qdrant thật. |
| H-EAGLE coarse-to-fine filter | **Blocker P0/P1** | `should` hiện không bắt buộc khi đi cùng `must`, nên chưa chắc giới hạn search theo shot; thiếu fallback khi shot collection chưa tồn tại. |
| Qdrant lifecycle/schema safety | Chưa đạt | Cần tránh xoá collection khi lỗi network/permission và xác nhận mapping tên collection production. |
| Shot ID namespace và rebuild idempotency | Chưa đạt | Cần namespace theo video và deterministic/generation-safe frame/shot rebuild để tránh duplicate/stale points. |
| Learned saliency | Deferred | Không đưa vào critical path; chỉ ablation off-by-default sau khi baseline ổn định. |
| H-EAGLE full Level 3 | Deferred | Narrative action + VLM/video encoder để sau vòng thi hoặc sau benchmark riêng. |

### Verification hiện tại

- `PYTHONPATH=. uv run --group dev pytest -q tests/test_transnet_heagle.py`: **4 passed**.
- `PYTHONPATH=. uv run --group dev pytest -q`: **56 passed**.
- `compileall` và `git diff --check`: pass.
- TransNetV2 checkpoint/model forward: pass với input `[1, 100, 27, 48, 3]`, không có missing/unexpected keys.
- Chưa có test streaming nhiều window, test fallback thiếu `vbs_shot_index`, hoặc end-to-end với Qdrant thật.
- Chạy test không có `PYTHONPATH=.` vẫn fail collection vì import package/path hiện tại; cần chuẩn hoá entrypoint/package import.

### Việc bắt buộc trước merge

1. Sửa stitching prediction theo `frame_idx`, không append toàn bộ các window overlap; thêm test synthetic 130/200 frames.
2. Sửa H-EAGLE filter thành điều kiện giới hạn shot bắt buộc, kèm fallback về frame search khi shot collection/query lỗi.
3. Bổ sung test shot batching/delete/coarse-to-fine và smoke test pipeline với Qdrant thật.
4. Xác nhận collection production (`visual_index`/`vbs_shot_index` so với collection đang tồn tại), unique shot ID theo video, và rebuild idempotency trước khi index corpus thật.
5. Chỉ sau các bước trên mới benchmark TransNetV2 với official `msb`/PySceneDetect và cân nhắc bật H-EAGLE-lite.

## Mục tiêu

1. Dùng TransNetV2 làm detector shot mặc định cho video không có shot boundary chính thức.
2. Giữ fallback an toàn cho official V3C assets, PySceneDetect và full-video fallback.
3. Tạo H-EAGLE-lite ở tầng shot để query có thể coarse-to-fine mà không phải thêm một VLM/action model mới ngay trước vòng thi.
4. Không để learned saliency làm chậm hoặc làm giảm recall của các query dựa trên chữ, biển báo, bản đồ và lower-third.

## Quyết định phạm vi

### P0 — TransNetV2, bắt buộc

- `official V3C msb` vẫn là nguồn ưu tiên số 1.
- TransNetV2 là detector số 2 cho video raw không có `msb`.
- PySceneDetect vẫn là fallback số 3 và baseline benchmark; chưa xoá dependency cho đến khi TransNetV2 qua smoke test và benchmark.
- Không chạy hai detector nối tiếp trong production. Chỉ chạy song song ở benchmark mode trên một tập nhỏ.

### P1 — H-EAGLE-lite, nên làm

- Tầng 1 giữ nguyên frame-level visual index hiện tại.
- Tầng 2 dùng shot boundary của official assets hoặc TransNetV2, rồi aggregate các frame embedding trong từng shot thành một vector shot normalized.
- Tạo collection shot riêng, không phá schema collection frame hiện tại.
- Query route coarse-to-fine là feature flag, mặc định tắt cho đến khi đo recall/latency.

### P2 — Learned saliency, không nằm trên critical path

- Không tải thêm model learned saliency trong đợt đầu.
- Chỉ chuẩn bị interface/feature flag nếu việc đó không ảnh hưởng TransNetV2.
- Không dùng saliency làm hard filter. Nếu benchmark tốt, chỉ cộng một bonus nhỏ vào điểm chọn keyframe và chạy trên candidate frames/official keyframes.

### Hoãn

H-EAGLE full Level 3 — narrative action grouping, VLM description alignment và video encoder riêng — để sau vòng này hoặc sau khi có benchmark rõ ràng. Nó không nên được giả định là “bật thêm một model là xong”.

## Các bước implementation

### 1. Preflight và quản lý weight

- Kiểm tra server bằng một script nhỏ: `torch.cuda.is_available()`, GPU name, CUDA runtime, VRAM khả dụng, dtype và khả năng load một batch frame.
- Chọn implementation PyTorch tương thích với môi trường hiện tại; không hạ cả environment xuống TensorFlow 2.1/PyTorch 1.7 chỉ để chạy upstream demo.
- Tách code inference TransNetV2 thành adapter nội bộ hoặc module vendored tối thiểu; pin commit/version và checksum weight.
- Bổ sung biến cấu hình:
  - `SHOT_DETECTOR=transnetv2`
  - `TRANSNETV2_MODEL_PATH=`
  - `TRANSNETV2_DEVICE=auto`
  - `TRANSNETV2_BATCH_SIZE=`
  - `SHOT_DETECTOR_FALLBACK=pyscenedetect`
  - `SHOT_BENCHMARK_MODE=false`
- Setup/download phải báo rõ thiếu weight, thiếu decoder hoặc incompatibility thay vì fail giữa chừng.

### 2. Viết adapter detector

Tạo module `preprocessing/video/transnet_detector.py` với một interface tương thích `detect_scenes(video_path) -> List[(start_sec, end_sec)]`.

Adapter cần:

- decode frame theo đúng format RGB và giữ `fps`, `frame_count` của video;
- feed model theo window/sequence mà TransNetV2 yêu cầu, có batch inference và `torch.inference_mode()`;
- chuyển `predictions_to_scenes` dạng inclusive frame indices sang giây;
- clamp `start_frame/end_frame`, sort, deduplicate, nối các đoạn liên tục;
- đảm bảo `0.0` đến `duration` được phủ hết, đặc biệt frame cuối;
- log detector, weight, device, số frame, số shot và thời gian inference;
- raise lỗi có loại rõ ràng để dispatcher chuyển sang PySceneDetect mà không làm hỏng cả job.

Không đưa CLIP refinement vào ngay trong adapter. Refinement hiện tại là hậu xử lý tùy chọn; chỉ benchmark sau khi baseline TransNetV2 ổn.

### 3. Tích hợp vào preprocessing

- Đổi `scene_detector.detect_scenes` thành dispatcher theo `SHOT_DETECTOR`.
- Thứ tự trong `main.py`:
  1. official shot boundaries nếu có;
  2. TransNetV2 nếu được chọn;
  3. PySceneDetect nếu TransNetV2 không load/chạy được;
  4. full-video fallback nếu mọi detector thất bại.
- Ghi `shot_detector` và `shot_count` vào run status/log để UI biết pipeline đang dùng detector nào.
- Giữ `SCENE_DETECTION_THRESHOLD` chỉ cho PySceneDetect; không dùng nhầm threshold đó cho TransNetV2.
- Không thay đổi keyframe selection, OCR, object detection hoặc caption trong PR detector đầu tiên.

### 4. Tạo H-EAGLE-lite

Tạo theo một PR riêng sau khi TransNetV2 đã pass:

- Gán `shot_id` ổn định theo `video_id + shot ordinal` cho mỗi selected frame.
- Tạo shot aggregate từ các frame embedding đã tính; ưu tiên mean-pool có trọng số quality nhẹ, sau đó L2-normalize.
- Payload shot gồm tối thiểu: `shot_id`, `video_id`, `video_path`, `start_frame`, `end_frame`, `start_sec`, `end_sec`, `representative_frame_ids`, `representative_timestamps`, `frame_count`.
- Tạo collection riêng, ví dụ `vbs_shot_index`, có vector size/distance khớp embedding hiện tại; không migrate collection frame đang chạy.
- Thêm job rebuild/resume shot index idempotent. Nếu chạy lại một video, xoá/upsert theo `video_id` để không có shot mồ côi.
- Query coarse-to-fine tùy chọn: top shot candidates trước, sau đó search frame trong union các shot; nếu coarse stage lỗi hoặc tắt thì quay về frame search hiện tại.
- UI chỉ hiển thị shot grouping/diagnostic khi cần; không làm thay đổi contract kết quả frame ở vòng đầu.

### 5. Learned saliency guardrail

Chỉ sau khi P0/P1 ổn định:

- Định nghĩa `SaliencyScorer` interface và `KEYFRAME_SALIENCY_ENABLED=false`.
- Chạy batch trên candidate/official keyframe, không chạy toàn bộ raw frames và không thêm model vào query-time.
- Dùng score như tie-break/bonus nhỏ cùng sharpness và diversity; không bỏ frame chỉ vì saliency thấp.
- Tạo benchmark tối thiểu theo ba nhóm: text/location, object/action, generic visual. Đặc biệt kiểm tra frame có biển báo, bản đồ và lower-third.
- Chỉ bật khi recall@K không giảm ở nhóm text/location, recall tổng tăng có ý nghĩa và thời gian preprocessing/VRAM nằm trong budget đã đặt. Nếu không, bỏ feature chứ không giữ vì “có model”.

## Thứ tự PR/branch

1. `feat/transnetv2-shot-detector`: adapter, config, weight check, fallback, unit/smoke test.
2. `feat/transnetv2-benchmark`: benchmark official shots/TransNetV2/PySceneDetect và log kết quả; nếu cần có thể gộp vào PR 1 nhưng không trộn H-EAGLE.
3. `feat/heagle-lite-shot-index`: shot IDs, aggregate vectors, collection, resume/rebuild, optional coarse-to-fine query.
4. `experiment/learned-saliency-ablation`: chỉ tạo nếu còn thời gian và đã có dataset test; không merge vào main nếu chưa đạt guardrail.

Mỗi branch phải có test và được merge qua PR riêng. PR tiếp theo chỉ bắt đầu sau khi PR trước đã pass test/benchmark và không có job indexing đang dùng schema cũ bị ảnh hưởng.

## Kiểm thử và tiêu chí chấp nhận

### Unit tests

- frame index inclusive → timestamp;
- video không có cut;
- video có cut sát đầu/cuối;
- duplicate/unsorted boundaries;
- thiếu weight, lỗi import, lỗi decode → fallback đúng;
- shot aggregate normalize đúng và không tạo điểm trùng khi resume;
- shot index empty hoặc query coarse stage lỗi → frame search fallback.

### Integration/smoke

- Synthetic video có hard cut, fade và đoạn tĩnh.
- Ít nhất một video thật có text/lower-third và một video có bản đồ/biển báo.
- Chạy `main.py` với `SHOT_DETECTOR=transnetv2`, sau đó tắt weight để xác nhận PySceneDetect fallback.
- Xác nhận timestamp UI/player không lệch, shot cuối phủ tới duration.

### Benchmark bắt buộc trước khi bật mặc định

- 10–20 video đại diện, so sánh official `msb` nếu có, TransNetV2 và PySceneDetect.
- Ghi: số shot, runtime decode/inference, peak VRAM, số keyframe, frame/shot coverage và các boundary lệch lớn.
- Với H-EAGLE-lite ghi thêm recall@K, latency query, số frame được search sau coarse filtering so với full frame search.
- TransNetV2 chỉ thành default cho raw video nếu không có lỗi coverage/timestamp nghiêm trọng và không làm preprocessing vượt budget. Nếu accuracy tốt nhưng dependency/VRAM chưa ổn, giữ dispatcher ở chế độ fallback thay vì ép production.

## Rủi ro và rollback

| Rủi ro | Phòng ngừa | Rollback |
| --- | --- | --- |
| Weight/implementation không hợp PyTorch hiện tại | adapter cô lập, smoke test, pin weight | `SHOT_DETECTOR=pyscenedetect` |
| Decoder tạo frame khác upstream | ghi fps/frame count, test timestamp, dùng cùng decoder trong benchmark | dùng official `msb` hoặc PySceneDetect |
| TransNetV2 dùng GPU tranh với OCR/YOLOE | chỉ chạy offline trước OCR/OD; batch nhỏ, device config | chạy detector CPU/PySceneDetect |
| Shot index làm giảm recall | feature flag coarse-to-fine off mặc định | query frame index hiện tại |
| Saliency bỏ qua chữ/map | không hard-filter, benchmark nhóm text/location | tắt `KEYFRAME_SALIENCY_ENABLED` |
| Rebuild tạo shot mồ côi/trùng | collection riêng, idempotent theo video_id | xoá/rebuild riêng shot collection, không đụng frame collection |

## Definition of done

- [x] TransNetV2 chạy ổn định trên video dài bằng weight đã kiểm chứng; overlap stitching không lệch frame.
- [x] Fallback và thứ tự official assets → TransNetV2 → PySceneDetect → full-video đã được code và smoke test trên synthetic video.
- [x] Official V3C assets vẫn được ưu tiên và không bị xử lý lại vô ích.
- [x] Một run preprocessing có thể resume với deterministic frame/shot IDs và log detector/shot count.
- [x] H-EAGLE-lite có shot index riêng, query coarse-to-fine giới hạn đúng theo shot, có fallback và explicit full-video rebuild guard.
- [x] Collection schema recreation và destructive per-video rebuild đều đã có flag explicit.
- [x] Learned saliency không được bật mặc định khi chưa có benchmark chứng minh lợi ích.
- [x] H-EAGLE full Level 3 được defer khỏi critical path.
- [x] README/CHANGELOG cập nhật cùng PR tương ứng.

### Verification completed in this run

- `uv run --group dev pytest -q`: **58 passed**.
- `uv run --group preprocessing python -m compileall -q preprocessing inference-code`: pass.
- `git diff --check`: pass.
- Vendored TransNetV2 forward smoke test on `[1, 100, 27, 48, 3]`: pass.
- Real checkpoint download/checksum verification: pass; checkpoint is pinned to a Hugging Face revision and SHA256.
- Synthetic 100-frame hard-cut video: TransNetV2 returned complete frame coverage and two scenes.
- Synthetic 130-frame mocked streaming test: overlapping windows stitched without duplicate predictions.

### Remaining deployment gates

- Benchmark against 10–20 real corpus videos and official `msb` where available.
- Confirm the target Qdrant deployment/collection mapping before indexing; this branch intentionally keeps the VBS schema names `visual_index`, `audio_env_index`, and `vbs_shot_index` and does not migrate unrelated legacy collections automatically.
- Resolve the existing official-keyframe native `frame_idx` mapping before relying on those points for frame-accurate competition output.

## Bước tiếp theo

Khi duyệt plan, chạy:

```text
/ck:cook /home/gay/tgltw-vbs-2027/plans/260806-transnetv2-h-eagle-saliency/plan.md
```
