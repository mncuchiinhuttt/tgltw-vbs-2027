# Research report: TransNetV2, H-EAGLE và learned saliency

Ngày: 2026-08-06

## Kết luận ngắn

- Nên thay detector mặc định bằng **TransNetV2**, nhưng giữ PySceneDetect làm fallback trong giai đoạn xác thực. Đây là thay đổi phù hợp với toolchain mà VBS công bố cho V3C/SIRET và không làm mất khả năng dùng shot boundary chính thức của BTC.
- Nên triển khai **H-EAGLE-lite**: giữ frame index hiện tại làm tầng 1, thêm shot/group index làm tầng 2 dựa trên TransNetV2 và các embedding đã có. Chưa nên triển khai tầng 3 đầy đủ (narrative action + VLM/video encoder) trước vòng thi vì đó là một pipeline mới, không phải một tuỳ chọn nhỏ.
- **Không đưa learned saliency vào critical path lúc này.** Sharpness + DAKE + semantic diversity đang có đủ vai trò chọn keyframe với chi phí thấp hơn. Learned saliency chỉ nên là ablation/offline option sau khi TransNetV2 và H-EAGLE-lite đã ổn định; nó có rủi ro bỏ qua chữ, biển báo, bản đồ và lower-third — chính là các tín hiệu quan trọng của truy vấn VBS.

## Nguồn chính

1. [VBS — Existing data and tools](https://videobrowsershowdown.org/about-vbs/existing-data-and-tools/) — trang chính thức liệt kê V3C có segmentation, shot boundaries, keyframes, analysis data, ASR và dẫn tới TransNet V2 của SIRET.
2. [TransNetV2 repository](https://github.com/soCzech/TransNetV2) — pretrained shot-boundary detector, mục tiêu phân tích archive hiệu quả; repository công bố kết quả benchmark trên ClipShots, BBC Planet Earth và RAI.
3. [TransNetV2 paper](https://arxiv.org/abs/2008.04838) — mô hình được thiết kế cho phân tích kho video lớn và có checkpoint inference công khai.
4. [TransNetV2 inference README](https://github.com/soCzech/TransNetV2/tree/master/inference) — API `predict_video`, `predictions_to_scenes`, input dạng chuỗi frame RGB và output scene theo chỉ số frame inclusive; yêu cầu weight/ffmpeg cần được kiểm soát.
5. [H-EAGLE paper](https://doras.dcu.ie/32449/1/Heagle.pdf) — đề xuất ba tầng: frame embedding, shot-level temporal representation và narrative-action representation; tìm kiếm coarse-to-fine từ shot/action xuống frame.

## Đánh giá TransNetV2

VBS đã công khai TransNetV2 như một tool của SIRET, nên đây là lựa chọn có cơ sở hơn việc tiếp tục tinh chỉnh một ngưỡng cố định của PySceneDetect. TransNetV2 nhận chuỗi frame RGB, xuất xác suất boundary rồi chuyển thành các đoạn scene bằng cặp frame bắt đầu/kết thúc. Adapter của chúng ta phải đổi frame index inclusive thành khoảng thời gian `(start_sec, end_sec)` và bảo đảm đoạn cuối video không bị mất.

Điểm cần lưu ý là upstream inference khá cũ: TensorFlow README nêu TensorFlow 2.1, còn bản PyTorch reimplementation nêu PyTorch 1.7.1/CUDA 10.1 và quy trình convert weight. Vì môi trường hiện tại dùng PyTorch mới hơn và đã có nhiều model GPU khác, dependency của TransNetV2 không nên được đưa vào bằng cách hạ toàn bộ môi trường. Cần vendor/adapt phần inference nhỏ, pin riêng weight, hoặc dùng một implementation tương thích sau khi smoke-test trên server.

Official V3C `msb`/shot boundary vẫn phải có precedence cao nhất. TransNetV2 chỉ chạy cho video không có shot boundary chính thức hoặc ở benchmark mode. PySceneDetect giữ vai trò fallback để pipeline không bị chặn khi weight, decoder hoặc import TransNetV2 lỗi.

## Đánh giá H-EAGLE

H-EAGLE là một kiến trúc phân cấp hoàn chỉnh, không chỉ là thêm một field vào frame payload:

- Tầng 1: frame-level embeddings — hệ thống hiện tại đã có.
- Tầng 2: nhóm frame theo shot và tạo temporal/shot representation — có thể làm ngay bằng TransNetV2 + embedding hiện tại.
- Tầng 3: gom narrative action bằng VLM và video encoder — cần thêm model, schema, job offline và nhánh query mới.

Vì vậy plan chọn H-EAGLE-lite. Mỗi frame point giữ `shot_id`; một collection shot riêng lưu vector aggregate đã normalize, `video_id`, `start_frame`, `end_frame`, `start_sec`, `end_sec`, số frame đại diện và danh sách point/frame tham chiếu. Query có thể bật coarse shot retrieval rồi local-search frame trong các shot được chọn. Mặc định chưa bật coarse routing cho production cho tới khi benchmark chứng minh recall không giảm.

## Đánh giá learned saliency

Learned saliency có thể giúp chọn frame “đáng xem” hơn, nhưng không đồng nghĩa với query relevance. Mô hình saliency thường ưu tiên vùng nổi bật/thu hút thị giác; trong VBS, một frame có chữ nhỏ trên cột mốc, bản đồ hoặc lower-third có thể không nổi bật nhưng lại là bằng chứng quyết định.

Ở thời điểm hiện tại learned saliency còn có ba chi phí: tải và kiểm tra thêm checkpoint, chiếm GPU/VRAM với embedding/OCR/YOLOE, và cần benchmark riêng theo nhóm truy vấn text/location/action. Do đó dùng các tín hiệu đã có làm baseline: DAKE để giảm candidate, Laplacian sharpness để ưu tiên frame đọc được, và farthest-point sampling trong embedding space để giữ đa dạng ngữ nghĩa. Nếu thử learned saliency, chỉ dùng như một bonus nhỏ, không hard-filter frame và chỉ chạy trên candidate/official keyframe; phải có feature flag tắt được.

## Quyết định phạm vi

| Hạng mục | Quyết định | Lý do |
| --- | --- | --- |
| TransNetV2 | Làm ngay, detector mặc định | VBS khuyến nghị; giảm lệ thuộc ngưỡng fixed của PySceneDetect |
| PySceneDetect | Giữ fallback và benchmark baseline | Bảo vệ pipeline khi model/weight/decode lỗi |
| Official V3C shots | Ưu tiên cao nhất | Dữ liệu BTC đã có sẵn, không cần tái suy luận |
| H-EAGLE-lite | Làm sau khi TransNetV2 ổn | Tận dụng embedding hiện tại, tạo coarse-to-fine có ích với ít model mới |
| H-EAGLE full level 3 | Hoãn | Cần VLM/action/video encoder và route query mới |
| Learned saliency | Off-by-default ablation | Rủi ro bỏ qua text/sign/map; lợi ích chưa được đo |

## Implementation audit — 2026-08-06

Các blocker trong vòng audit đã được sửa:

- TransNetV2 chỉ ghép vùng prediction trung tâm tương ứng với 50 frame mới của mỗi cửa sổ overlap; có regression test mocked 130 frame.
- H-EAGLE dùng nested required `Filter(should=...)`, nên shot filter thật sự giới hạn frame search; thiếu collection hoặc lỗi query sẽ fallback về frame search.
- Official shot IDs được namespace theo video; frame/shot IDs deterministic; schema recreation và full-video rebuild đều cần flag explicit.
- Đã có 58 test pass, compileall, checkpoint forward smoke test và synthetic hard-cut smoke test.

Các deployment gate còn lại không phải blocker cho code path mặc định (`HEAGLE_LITE_ENABLED=false`): benchmark trên video thật, xác nhận collection mapping của Qdrant target, và bổ sung native `frame_idx` map cho official keyframes nếu cần frame-accurate competition output. Learned saliency và H-EAGLE Level 3 tiếp tục deferred.
