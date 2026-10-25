# Tiến độ hiện tại của dự án

> Cập nhật ngày 15/10/2026 trên nhánh `feature/preprocessing`, đối chiếu theo các điều kiện hoàn thành trong [Proposal.md](./Proposal.md), mã nguồn và artifact hiện có trong workspace.

## 1. Dự án đang ở giai đoạn nào?

Dự án hiện ở **cuối Phase 2 về mặt implementation**, nhưng Phase 1 và Phase 2 chưa đạt đầy đủ Definition of Done vì pipeline dữ liệu chưa được chạy và xác minh end-to-end trong workspace hiện tại. Một phần framework của Phase 3 đã có và ba backbone đã vượt qua smoke test bằng dữ liệu synthetic; chưa có baseline nào được huấn luyện trên dữ liệu thật.

| Phase trong Proposal | Trạng thái | Bằng chứng và phần còn thiếu |
|---|---|---|
| Phase 0 — Project Setup | Gần hoàn tất | Có repo, cấu trúc module, config, requirements, seed và smoke test; README và quy trình fresh-clone còn sơ lược |
| Phase 1 — Data Understanding & EDA | Đang hoàn thiện | Có EDA, thống kê lớp, annotation parser và quyết định đủ 14 finding; chưa có `dataset_specification.md` và bằng chứng sanity check 50–100 ảnh |
| Phase 2 — Data Pipeline | Đã code, chưa xác minh end-to-end | Có manifest, majority label, mask, split và `Dataset`; thư mục `data/` hiện không có artifact chứng minh toàn pipeline đã chạy thành công |
| Phase 3 — Classification Baselines | Framework ban đầu | Có model factory, weighted BCE, trainer, checkpoint/resume, early stopping và metrics; mới có smoke test synthetic, chưa có baseline thật |
| Phase 4 — Grad-CAM & Localization | Chưa bắt đầu | Chưa có Grad-CAM, Pointing Game, IoU hoặc Energy Inside Box |
| Phase 5 — Location-Aware Learning | Chưa bắt đầu | Chưa có spatial activation extraction, location-aware loss hoặc lambda ablation |
| Phase 6 — Final Research Analysis | Chưa bắt đầu | Chưa có kết quả RQ1/RQ2, error analysis hoặc multi-seed validation |
| Phase 7 — Prototype & Delivery | Chưa bắt đầu | `app/` còn rỗng; chưa có inference pipeline, Streamlit hoặc defense package |

Không nên xem Phase 3 là đã hoàn thành chỉ vì training code và smoke test tồn tại. Baseline vẫn phải có checkpoint, log và kết quả đánh giá trên dữ liệu thật.

## 2. Quyết định chính thức về 14 finding

Dự án sử dụng toàn bộ 14 finding bất thường của VinDr-CXR thay vì thu hẹp còn 3–5 lớp. Số lớp vẫn ở quy mô có thể quản lý và quyết định này giữ đầy đủ thông tin cho cả classification lẫn localization.

Thứ tự lớp chuẩn, cũng là thứ tự trong [`configs/default.yaml`](../configs/default.yaml), là:

1. `Aortic enlargement`
2. `Atelectasis`
3. `Calcification`
4. `Cardiomegaly`
5. `Consolidation`
6. `ILD`
7. `Infiltration`
8. `Lung Opacity`
9. `Nodule/Mass`
10. `Other lesion`
11. `Pleural effusion`
12. `Pleural thickening`
13. `Pneumothorax`
14. `Pulmonary fibrosis`

`No finding` không nằm trong vector output. Ảnh chỉ có `No finding` vẫn được giữ làm mẫu âm cho cả 14 finding bất thường.

Task 1.3 trong Proposal được xem là **đã hoàn thành**. Danh sách và thứ tự trên phải được giữ cố định trong manifest, split, label vector, model output, checkpoint và mọi file đánh giá.

EDA hiện có cho thấy mất cân bằng rất lớn: `Aortic enlargement` có 2.346 ảnh dương theo majority vote, trong khi `Atelectasis` chỉ có 62 và `Pneumothorax` có 58. Vì vậy cần dùng weighted BCE và báo cáo AUROC, AUPRC, Precision, Recall, F1 theo từng lớp bên cạnh macro metrics; không kết luận chỉ từ một giá trị macro.

## 3. Những phần đã triển khai

### Data và annotation

- `notebooks/01_eda.ipynb` đã đọc annotation thật, thống kê finding/reader và tạo bảng, biểu đồ trong `outputs/eda/`.
- `src/data/annotations.py` kiểm tra schema, reader roster và tạo hard label bằng majority vote tối thiểu hai trong ba reader.
- Bounding box được union theo từng reader rồi tổng hợp thành soft mask, union mask, `mask_valid` và số reader dương.
- `scripts/build_manifest.py` tạo manifest, reader vote, class summary và metadata annotation protocol.
- `scripts/make_splits.py` dùng multilabel stratification hai bước, seed cố định và fingerprint assignment để tạo split 70/15/15 tái lập.
- `scripts/preprocess.py` resize JPG về 224×224, sinh mask, kiểm tra label và tạo processed manifest/split.
- `src/data/dataset.py` trả về image, 14 labels, masks, metadata và kiểm tra class order, shape, giá trị, tính nhất quán.

### Classification framework

- `src/models/factory.py` tạo ResNet-50, DenseNet-121 hoặc ViT-B/16 với 14 output.
- `src/losses/classification_loss.py` tính `pos_weight` từ train split và tạo weighted BCE.
- `src/training/engine.py` cung cấp train/validation epoch dùng chung.
- `scripts/train.py` hỗ trợ AMP FP16, progress bar, throughput, config, seed, checkpoint `best.pt`/`last.pt`, resume, early stopping và validation metrics.
- `src/metrics/classification.py` tính AUROC, average precision, Precision, Recall và F1 theo từng finding cùng macro average.
- Smoke test synthetic đã thành công với cả ba backbone: forward, backward, finite gradient, optimizer step và checkpoint reload đều đạt. Đây là kiểm tra kỹ thuật, không phải kết quả nghiên cứu.
- `python -m compileall -q src scripts` hiện hoàn thành với exit code 0.

## 4. Các đầu việc chưa hoàn thành

- Workspace có manifest và processed artifact cho 15.000 ảnh; split được cố định thành 10.500 train, 2.250 validation và 2.250 internal test.
- Chưa có `dataset_specification.md` và chưa có bằng chứng sanity check ngẫu nhiên 50–100 ảnh sau preprocessing.
- Tests bao phủ split 70/15/15, processed split projection, checkpoint fingerprint và prediction schema; các phần annotation/mask/loss vẫn cần thêm coverage.
- Có standalone `evaluate.py`; threshold mặc định hiện là 0,5.
- Chưa có learning-rate scheduler dù Proposal yêu cầu.
- Chưa có checkpoint, prediction, history hoặc test metrics baseline thật cho bất kỳ backbone nào.
- Chưa triển khai Grad-CAM/localization, location-aware learning, RQ1/RQ2 analysis và prototype.

## 5. Cấu trúc và luồng hiện tại

```text
configs/        # Default protocol + config cho ba backbone
notebooks/      # Colab runner và EDA
scripts/        # Download, manifest, split, preprocess, smoke test, train
src/data/       # Annotation, preprocessing, transforms, Dataset
src/models/     # Model factory cho ResNet/DenseNet/ViT
src/losses/     # Weighted classification loss
src/metrics/    # Classification metrics
src/training/   # Epoch engine
src/utils/      # Config, seed, checkpoint
src/xai/        # Chưa có implementation
tests/          # Rỗng
app/            # Rỗng
```

Luồng đã được code:

```text
Kaggle JPG → manifest 14 labels → multilabel split
           → ảnh 224×224 + soft/union masks → CXRDataset
           → ResNet/DenseNet/ViT → weighted BCE
           → validation metrics + checkpoint
```

Phần từ manifest đến training chưa có artifact dữ liệu thật trong workspace để chứng minh đã chạy hoàn chỉnh. Luồng tiếp theo phải là baseline checkpoints → Grad-CAM → localization metrics → RQ1 → location-aware loss → RQ2 → prototype.

## 6. Cách chạy phần hiện có

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/prepare_data.py

python scripts/build_manifest.py --config configs/default.yaml
python -m scripts.make_splits --config configs/default.yaml --overwrite
python -m scripts.preprocess --config configs/default.yaml --splits-only --overwrite

python scripts/smoke_test.py --config configs/resnet50_baseline.yaml
python scripts/smoke_test.py --config configs/densenet121_baseline.yaml
python scripts/smoke_test.py --config configs/vit_baseline.yaml
```

Sau khi data pipeline hoàn thành, lệnh training có dạng:

```powershell
python scripts/train.py `
  --config configs/resnet50_baseline.yaml `
  --checkpoint-dir outputs/checkpoints/resnet50_baseline_s701515_seed42_amp `
  --result-dir outputs/metrics/resnet50_baseline_s701515_seed42_amp
```

Sau khi khóa cấu hình bằng validation, dùng `scripts.evaluate` để đánh giá `best.pt` trên internal test đúng một lần.

## 7. Rủi ro cần theo dõi

- **Mất cân bằng nghiêm trọng:** phải theo dõi AUPRC, Recall và F1 từng lớp, đồng thời kiểm tra split giữ được đại diện cho các lớp hiếm.
- **Quy mô mask:** lưu mask 14 lớp cho mỗi ảnh có thể tạo I/O và dung lượng đáng kể trên Colab/Drive; cần đo thực tế.
- **JPG/DICOM chưa thống nhất:** Proposal nhắc DICOM nhưng code dùng bộ JPG 1024 px; báo cáo phải ghi đúng protocol thực tế.
- **Test leakage:** internal test 2.250 ảnh chỉ được dùng sau khi khóa model/hyperparameter bằng validation; `data/raw/test` không được sử dụng vì thiếu ground truth trong workspace.
- **Threshold cố định:** nếu tune threshold, chỉ được dùng validation và phải cố định trước final evaluation.
- **Thiếu regression test:** nhiều invariant chỉ được kiểm tra runtime.
- **Artifact synthetic:** tuyệt đối không dùng smoke-test loss hoặc checkpoint làm kết quả baseline.

## 8. Ưu tiên tiếp theo

1. Chạy lại split 70/15/15 và projection processed split, sau đó đóng gói lại archive dùng trên Colab.
2. Viết `dataset_specification.md` và sanity-check trực quan ít nhất 50–100 ảnh/mask.
3. Bổ sung automated test còn thiếu cho annotation, masks, dataset, loss và metrics.
4. Chạy ResNet-50 baseline từ đầu, sau đó đánh giá best checkpoint bằng standalone evaluator.
5. Sau khi ResNet ổn định, chạy DenseNet-121 và ViT-B/16 theo cùng split/protocol.
6. Triển khai Grad-CAM và localization metrics trước khi bắt đầu location-aware loss.
