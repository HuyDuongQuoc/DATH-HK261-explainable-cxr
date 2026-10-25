# Training Framework và Data Pipeline

> Tài liệu kỹ thuật cho codebase hiện tại, cập nhật ngày 15/10/2026. Nội dung mô tả phần **đã được triển khai** trên nhánh `feature/preprocessing`; các khả năng dự kiến nhưng chưa có code được ghi riêng ở phần giới hạn.

## 1. Phạm vi và mục tiêu thiết kế

Hai subsystem được mô tả trong tài liệu này là:

- **Data Pipeline:** chuyển annotation CSV và ảnh VinBigData JPG thành manifest, split 70/15/15, ảnh 224×224 và location mask có cùng class order.
- **Training Framework:** dùng một interface chung để huấn luyện ResNet-50, DenseNet-121 và ViT-B/16 cho bài toán multi-label classification 14 finding.

Thiết kế này phục vụ ba yêu cầu nghiên cứu:

1. Mọi backbone phải dùng cùng label rule, split và preprocessing để kết quả classification có thể so sánh công bằng.
2. Dataset phải giữ đồng thời image-level label và spatial mask để baseline classification có thể được đánh giá localization ở Phase 4.
3. Config, seed, checkpoint và history phải đủ để tái lập experiment và tiếp tục training khi Colab bị ngắt.

Luồng tổng thể:

```text
Raw JPG + annotation CSV
          │
          ▼
  Manifest 14 labels
          │
          ▼
Multilabel train/val/test split
          │
          ▼
Ảnh 224×224 + soft/union masks
          │
          ▼
      CXRDataset
          │
          ▼
       DataLoader
          │
          ▼
ResNet-50 / DenseNet-121 / ViT-B/16
          │
          ▼
 Weighted BCE → Metrics → Checkpoint
```

Workspace hiện có raw/processed data và split 70/15/15 đã được kiểm tra; chưa có checkpoint baseline thật để chứng minh toàn bộ luồng training/evaluation đã chạy end-to-end.

## 2. Contract dùng chung

### 2.1 Config inheritance

[`src/utils/config.py`](../src/utils/config.py) cung cấp hai hàm:

- `merge_dict(base, override)`: merge đệ quy dictionary; giá trị trong config model ghi đè giá trị tương ứng của default config.
- `load_config(config_path)`: luôn tìm `default.yaml` trong cùng thư mục với config được truyền vào, sau đó merge nếu file đầu vào không phải chính `default.yaml`.

Ba config backbone hiện chỉ override `model.name`:

| Config | `model.name` |
|---|---|
| [`resnet50_baseline.yaml`](../configs/resnet50_baseline.yaml) | `resnet50` |
| [`densenet121_baseline.yaml`](../configs/densenet121_baseline.yaml) | `densenet121` |
| [`vit_baseline.yaml`](../configs/vit_baseline.yaml) | `vit_b_16` |

Các giá trị còn lại được kế thừa từ [`default.yaml`](../configs/default.yaml):

| Nhóm | Giá trị hiện tại | Vai trò |
|---|---|---|
| Reproducibility | `seed: 42` | Seed cho Python, NumPy, PyTorch, CUDA, split và DataLoader shuffle |
| Annotation | `expected_readers: 3`, `positive_votes: 2` | Hard label theo majority vote 2/3 |
| Split | `train/val/test_fraction: 0.70/0.15/0.15` | 10.500 train, 2.250 validation, 2.250 internal test |
| Image | `image_size: 224` | Kích thước ảnh và mask sau preprocessing |
| Normalization | ImageNet mean/std | Phù hợp pretrained torchvision backbones |
| Training | batch 8, 30 epoch, AdamW LR `1e-4`, AMP FP16 | Protocol baseline trên Colab T4 |
| DataLoader | 2 workers, persistent workers, prefetch 2 | Giảm thời gian GPU chờ dữ liệu |
| Early stopping | patience 5, min delta `1e-4` | Theo validation macro-AUROC |

`load_config` từ chối danh sách finding rỗng hoặc trùng tên. Các key bắt buộc khác chưa có schema validator tổng quát; thiếu key sẽ phát sinh `KeyError` tại nơi sử dụng.

### 2.2 Class-order contract

Mọi artifact phải giữ nguyên thứ tự 14 finding sau:

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

Thứ tự này chi phối column order trong manifest, label vector, mask tensor, model output và checkpoint. [`CXRDataset`](../src/data/dataset.py) kiểm tra `class_names` trong `.npz` phải khớp chính xác với danh sách được truyền vào.

`No finding` không phải output class thứ 15. Khi một ảnh không có vote dương cho 14 finding, vector label của ảnh là toàn số 0; vì vậy ảnh `No finding` vẫn đóng vai trò mẫu âm.

## 3. Data Pipeline

### 3.1 Stage 0 — Tải raw data

Entrypoint: [`scripts/prepare_data.py`](../scripts/prepare_data.py)

Script đọc Kaggle credentials từ `.env`, tải và giải nén dataset `sunghyunjun/vinbigdata-1024-jpg-dataset` vào `data/raw`.

Input được các stage sau kỳ vọng:

```text
data/raw/
├── train.csv
└── train/
    └── <image_id>.jpg
```

Pipeline hiện dùng bản **JPG 1024 px**, không đọc DICOM dù Proposal ban đầu có nhắc DICOM preprocessing.

### 3.2 Stage 1 — Đọc và kiểm tra annotation

Module: [`src/data/annotations.py`](../src/data/annotations.py)

`read_annotations(csv_path)`:

- Xóa các cột có tên bắt đầu bằng `Unnamed:`.
- Yêu cầu các cột `image_id`, `class_name`, `rad_id`, `x_min`, `y_min`, `x_max`, `y_max`.
- Từ chối giá trị thiếu trong ba cột định danh.
- Chuẩn hóa cột định danh thành string đã strip whitespace.
- Ép bốn tọa độ bbox về numeric; giá trị không hợp lệ thành `NaN` và sẽ bị từ chối nếu bbox đó được dùng.

`get_reader_roster(df, expected_readers=3)` gom reader theo `image_id` và yêu cầu mỗi ảnh có đúng ba `rad_id` duy nhất. Nếu sai, exception chứa tối đa năm ví dụ đầu tiên.

`validate_bbox(row, width, height)` yêu cầu tọa độ hữu hạn và thỏa:

```text
0 <= x_min < x_max <= width
0 <= y_min < y_max <= height
```

### 3.3 Stage 2 — Majority label và spatial consensus

Hai dạng target được tạo từ cùng annotation:

#### Image-level hard label

`make_majority_labels` đếm số reader duy nhất đánh dấu mỗi finding trên mỗi ảnh:

```text
label(image, class) = 1 nếu positive_reader_count >= 2
                      0 nếu ngược lại
```

Hàm kiểm tra danh sách finding không rỗng và mọi finding đều tồn tại trong annotation ngoài `No finding`.

#### Pixel-level mask

`build_image_consensus` xử lý riêng từng finding và reader:

1. Rasterize mọi bbox của một reader thành binary mask.
2. Union các bbox cùng reader bằng cách gán vùng bbox bằng 1.
3. Cộng mask của ba reader thành `vote_counts` có giá trị từ 0 đến 3.
4. Tạo các output:

```text
soft_mask  = vote_counts / 3
union_mask = vote_counts > 0
label      = positive_reader_count >= 2
mask_valid = label AND union_mask có ít nhất một pixel dương
```

Cách làm “union trong từng reader rồi average giữa reader” tránh việc một reader vẽ nhiều bbox được tính thành nhiều vote.

### 3.4 Stage 3 — Build manifest

Entrypoint: [`scripts/build_manifest.py`](../scripts/build_manifest.py)

```powershell
python -m scripts.build_manifest --config configs/default.yaml
```

Thêm `--overwrite` nếu chủ động thay đổi data protocol và muốn ghi đè artifact cũ. Nếu không có flag này, script dừng khi manifest đã tồn tại.

`build_manifest` kiểm tra file `train/<image_id>.jpg`, đọc kích thước thật và tạo một dòng cho mỗi ảnh:

| Column | Kiểu/ý nghĩa |
|---|---|
| `image_id` | ID duy nhất của ảnh |
| `raw_image_path` | Đường dẫn tương đối, dạng `train/<image_id>.jpg` |
| `raw_width`, `raw_height` | Kích thước ảnh JPG gốc |
| 14 finding columns | Hard majority label 0/1 theo class order chuẩn |

Các artifact được ghi vào `data/processed/`:

| Artifact | Nội dung |
|---|---|
| `manifest.csv` | Một dòng mỗi ảnh và 14 labels |
| `reader_votes.csv` | Số reader dương theo ảnh/finding |
| `class_summary.csv` | Positive, negative và positive rate theo finding |
| `metadata.json` | Findings, số reader, vote threshold và consensus rule |

### 3.5 Stage 4 — Tạo train/validation/test split

Entrypoint: [`scripts/make_splits.py`](../scripts/make_splits.py)

```powershell
python -m scripts.make_splits --config configs/default.yaml
```

Script dùng `MultilabelStratifiedShuffleSplit` hai bước với seed 42: tách 70% train/30% holdout, sau đó chia holdout thành 15% validation và 15% internal test. Split được tạo ở mức `image_id`, không phải annotation row. `data/raw/test` không tham gia protocol vì workspace không có ground truth tương ứng.

Các invariant được kiểm tra:

- `image_id` trong manifest phải unique.
- Train, validation và test đôi một không giao nhau và phủ đủ manifest.
- Mỗi finding trong mỗi split phải có cả positive và negative sample.
- Split cũ không bị ghi đè nếu thiếu `--overwrite`.

Artifact trong `data/processed/splits/`:

| Artifact | Nội dung |
|---|---|
| `train.csv`, `val.csv`, `test.csv` | Các dòng manifest thuộc từng split, sort theo `image_id` |
| `summary.csv` | Positive, negative và positive rate theo split/finding |
| `split_metadata.json` | Seed, method, fractions, số ảnh, findings, SHA-256 manifest và fingerprint assignment |

Checksum giúp phát hiện trường hợp split được tạo từ một manifest khác với manifest đang sử dụng.

### 3.6 Stage 5 — Preprocess ảnh và mask

Entrypoint: [`scripts/preprocess.py`](../scripts/preprocess.py)

Core module: [`src/data/preprocessing.py`](../src/data/preprocessing.py)

```powershell
python -m scripts.preprocess --config configs/default.yaml
```

Với từng dòng manifest, script:

1. Mở JPG và chuyển thành grayscale `L`.
2. Tạo consensus mask ở resolution gốc.
3. Resize ảnh về 224×224 bằng bilinear interpolation.
4. Resize soft mask bằng area interpolation rồi clamp về `[0, 1]`.
5. Resize union mask bằng area interpolation và coi mọi coverage lớn hơn 0 là pixel dương.
6. Đối chiếu label sinh lại với label trong manifest.
7. Ghi ảnh PNG, mask NPZ và processed manifest.

Cấu trúc output mặc định:

```text
data/processed/consensus_v1_224/
├── images/<image_id>.png
├── masks/<image_id>.npz
├── manifest.csv
├── metadata.json
└── splits/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── summary.csv
    └── split_metadata.json
```

Mỗi file `.npz` có contract:

| Key | Shape | Dtype lưu trữ | Ý nghĩa |
|---|---:|---|---|
| `soft_masks` | `[14, 224, 224]` | `float16` | Tỷ lệ reader phủ mỗi pixel |
| `union_masks` | `[14, 224, 224]` | `uint8` | Có ít nhất một reader phủ pixel |
| `labels` | `[14]` | `uint8` | Hard majority labels |
| `mask_valid` | `[14]` | `bool` | Class dương và có spatial mask |
| `positive_reader_counts` | `[14]` | `uint8` | Số reader dương |
| `class_names` | `[14]` | Unicode string | Class order được đóng gói cùng artifact |
| `num_readers` | scalar | `uint8` | Số reader của ảnh |

Processed train/validation/test split phải có cùng số dòng với raw split tương ứng; nếu không, script dừng thay vì tạo dataset thiếu ảnh âm thầm. Khi ảnh/mask đã tồn tại, `--splits-only` chỉ tạo lại các split manifest.

### 3.7 Stage 6 — `CXRDataset`

Class: [`CXRDataset`](../src/data/dataset.py)

Constructor nhận:

```python
CXRDataset(
    manifest_path,
    processed_root,
    findings,
    image_size,
    mean,
    std,
    load_masks=True,
)
```

Mỗi sample luôn trả về `image`, `labels`, `image_id`. Khi `load_masks=True`, sample trả thêm:

| Key | Runtime type/shape |
|---|---|
| `image` | `torch.float32 [3, 224, 224]`, grayscale được lặp thành 3 channel và normalize |
| `labels` | `torch.float32 [14]` |
| `soft_masks` | `torch.float32 [14, 224, 224]` |
| `union_masks` | `torch.float32 [14, 224, 224]` |
| `mask_valid` | `torch.bool [14]` |
| `positive_reader_counts` | `torch.int64 [14]` |
| `image_id` | Python `str` |

Dataset kiểm tra:

- `image_id` không trùng và đủ 14 label columns.
- Label chỉ có 0/1.
- Ảnh đúng kích thước cấu hình.
- Class order trong NPZ khớp config.
- Label trong manifest khớp label trong NPZ.
- Mask đúng shape, hữu hạn và soft mask nằm trong `[0, 1]`.
- `mask_valid` khớp trạng thái label/mask thực tế.

Các kiểm tra mask chỉ chạy khi `load_masks=True`. Baseline classification truyền `False` để không mở/giải nén NPZ không được sử dụng; location-aware pipeline vẫn giữ chế độ đầy đủ. Các kiểm tra này cố ý fail fast để tránh train trên artifact không cùng protocol.

## 4. Training Framework

### 4.1 Training entrypoint

Entrypoint: [`scripts/train.py`](../scripts/train.py)

CLI bắt buộc:

| Argument | Ý nghĩa |
|---|---|
| `--config` | Config backbone được merge với `default.yaml` |
| `--checkpoint-dir` | Thư mục ghi `best.pt` và `last.pt` |
| `--result-dir` | Thư mục ghi config, history và metrics |

CLI tùy chọn:

| Argument | Ý nghĩa |
|---|---|
| `--resume <checkpoint>` | Khôi phục model, optimizer, epoch, best score, bad epoch count và history |

Script gọi `set_seed` trước khi tạo Dataset, DataLoader và model. Device được chọn tự động: CUDA nếu khả dụng, ngược lại CPU.

### 4.2 Dataset và DataLoader

`train.py` luôn đọc:

```text
<processed_root>/splits/train.csv
<processed_root>/splits/val.csv
```

Train DataLoader dùng `shuffle=True` và một `torch.Generator` seed cố định. Validation không shuffle. Trên CUDA, loader dùng pinned memory; mặc định có 2 persistent workers và prefetch 2 batch. Image/label được chuyển sang GPU với `non_blocking=True`.

Test không được đọc trong `train.py`; `scripts/evaluate.py` chỉ được chạy sau khi đã khóa model và hyperparameter bằng validation.

Mask không được load trong baseline classification. `CXRDataset(load_masks=True)` vẫn giữ nguyên contract mask để phục vụ localization evaluation và location-aware loss ở các phase sau.

### 4.3 Model factory

Module: [`src/models/factory.py`](../src/models/factory.py)

`build_model(model_name, num_classes, pretrained=True)` hỗ trợ:

| Model name | Torchvision model | Head được thay |
|---|---|---|
| `resnet50` | `resnet50` | `model.fc` |
| `densenet121` | `densenet121` | `model.classifier` |
| `vit_b_16` | `vit_b_16` | `model.heads.head` |

Mọi head mới đều là `nn.Linear(in_features, 14)`. Khi `pretrained: true`, code dùng ImageNet weights tương ứng; tên model khác ba giá trị trên gây `ValueError`.

### 4.4 Classification loss

Module: [`src/losses/classification_loss.py`](../src/losses/classification_loss.py)

Với mỗi class `c`, weight được tính chỉ từ train split:

```text
positive_c   = tổng label dương của class c
negative_c   = số ảnh train - positive_c
pos_weight_c = negative_c / positive_c
```

`build_classification_loss` tạo `BCEWithLogitsLoss(pos_weight=...)`. Nếu bất kỳ class nào không có positive hoặc negative sample, code dừng bằng `ValueError`.

Weighted BCE giúp lớp hiếm đóng góp nhiều hơn vào gradient. Tuy nhiên đây không phải bảo đảm rằng F1/Recall của lớp hiếm sẽ tốt; metrics theo từng class vẫn bắt buộc trong phân tích.

### 4.5 Epoch engine

Module: [`src/training/engine.py`](../src/training/engine.py)

`run_epoch` dùng cùng flow cho train và validation, hỗ trợ CUDA autocast FP16, GradScaler, tqdm progress và đo throughput:

- Có optimizer: `model.train()`, bật gradient, zero grad → forward → loss → backward → optimizer step.
- Không có optimizer: `model.eval()`, tắt gradient, chỉ forward và tính loss.
- Kiểm tra shape logits phải bằng shape labels và loss phải hữu hạn.
- Tích lũy loss theo số ảnh, sigmoid probabilities, targets và `image_id`.
- AMP tự tắt khi device không phải CUDA; CPU smoke test vẫn chạy FP32.
- Progress bar hiển thị phần trăm batch, running loss, images/second và peak GPU memory.

Return contract:

```python
{
    "loss": float,
    "targets": np.ndarray,       # [N, 14]
    "probabilities": np.ndarray, # [N, 14]
    "image_ids": list[str],
    "elapsed_seconds": float,
    "images_per_second": float,
    "peak_gpu_memory_gb": float,
}
```

Engine từ chối DataLoader rỗng.

### 4.6 Classification metrics

Module: [`src/metrics/classification.py`](../src/metrics/classification.py)

`compute_classification_metrics` yêu cầu target và probability cùng shape, với chiều class bằng 14. Với threshold mặc định `0.5`, hàm tạo prediction nhị phân và tính theo từng finding:

- AUROC.
- Average Precision, được dùng như AUPRC summary.
- Precision.
- Recall.
- F1.
- Số positive và negative.

Macro metrics là trung bình không trọng số của 14 class. Nếu một class trong validation chỉ có một label value, AUROC không xác định và hàm dừng bằng `ValueError`; split stage được thiết kế để phát hiện trường hợp này sớm.

### 4.7 Optimizer, early stopping và checkpoint

Optimizer hiện tại là AdamW với learning rate và weight decay từ config. Chưa có learning-rate scheduler.

Sau mỗi epoch:

1. Chạy train epoch.
2. Chạy validation epoch.
3. Tính validation metrics ở threshold 0,5.
4. So `macro_auroc` với `best_macro_auroc + min_delta`.
5. Nếu cải thiện: reset `bad_epochs` và ghi `best.pt`.
6. Nếu không: tăng `bad_epochs`.
7. Luôn ghi `last.pt`, `history.csv` và `latest_val_metrics.json`; khi cải thiện ghi thêm `best_val_metrics.json`.
8. Dừng trước epoch tiếp theo khi `bad_epochs >= patience`.

[`src/utils/checkpoint.py`](../src/utils/checkpoint.py) ghi checkpoint qua file `.tmp` rồi replace sang path đích, giảm rủi ro checkpoint bị hỏng nếu quá trình ghi bị gián đoạn.

Checkpoint payload gồm:

| Field | Nội dung |
|---|---|
| `epoch` | Epoch index zero-based vừa hoàn thành |
| `model_name` | Backbone identifier |
| `model_state_dict` | Model weights |
| `optimizer_state_dict` | AdamW state |
| `amp_enabled` | AMP có thực sự hoạt động trên CUDA hay không |
| `amp_scaler_state_dict` | GradScaler state để resume AMP đúng giai đoạn |
| `best_macro_auroc` | Best validation score đến hiện tại |
| `bad_epochs` | Số epoch liên tiếp không cải thiện |
| `findings` | Danh sách/class order 14 finding |
| `pos_weight` | Weight của weighted BCE |
| `config` | Full merged config |
| `history` | Danh sách epoch records |

Khi resume, code kiểm tra checkpoint có cùng class order, model name, split fingerprint và AMP scaler state. Training tiếp tục từ `checkpoint["epoch"] + 1`.

### 4.8 Training artifact

`result_dir` chứa:

| Artifact | Nội dung |
|---|---|
| `config.json` | Full merged config của run |
| `history.csv` | Loss, metrics, thời gian, throughput và peak GPU memory theo epoch |
| `latest_val_metrics.json` | Per-class và macro metrics của epoch gần nhất |
| `best_val_metrics.json` | Metrics của best checkpoint; notebook đọc trực tiếp để tránh chạy lại validation |

`checkpoint_dir` chứa:

| Artifact | Nội dung |
|---|---|
| `best.pt` | Checkpoint ở epoch có macro-AUROC tốt nhất |
| `last.pt` | Checkpoint epoch gần nhất, dùng để resume |

Chưa có prediction CSV, test metrics, TensorBoard log hoặc standardized export package.

### 4.9 Synthetic smoke test

Entrypoint: [`scripts/smoke_test.py`](../scripts/smoke_test.py)

Smoke test tạo batch ngẫu nhiên `[2, 3, 224, 224]`, labels ngẫu nhiên `[2, 14]`, chạy một optimizer step rồi lưu/reload state dict. Nó xác nhận:

- Model factory tạo đúng output shape.
- Forward/backward hoạt động và gradient hữu hạn.
- Optimizer step chạy được.
- Checkpoint reload không đổi output.

Output mặc định nằm tại `outputs/smoke_test/<model_name>/`. Loss và checkpoint của smoke test dùng dữ liệu synthetic, **không phải baseline result** và không được dùng trong báo cáo nghiên cứu.

## 5. Cách chạy end-to-end

### 5.1 Chuẩn bị môi trường và dữ liệu

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Tạo .env theo .env.example trước khi tải Kaggle data
python -m scripts.prepare_data

python -m scripts.build_manifest --config configs/default.yaml
python -m scripts.make_splits --config configs/default.yaml --overwrite
python -m scripts.preprocess --config configs/default.yaml --splits-only --overwrite
```

Các script được gọi bằng `python -m scripts.<module>` từ repo root để repo root nằm trên Python import path và các import `src.*` hoạt động ổn định.

Notebook Colab cài `requirements-common.txt` và dùng PyTorch có sẵn trong runtime sau khi xác nhận CUDA, T4 và FP16 autocast. Local/controlled environment tiếp tục dùng `requirements.txt` để pin torch/torchvision.

`--overwrite` chỉ nên dùng khi chủ động thay đổi data protocol. Split cũ nên được tái sử dụng cho mọi backbone.

### 5.2 Smoke test ba backbone

```powershell
python -m scripts.smoke_test --config configs/resnet50_baseline.yaml
python -m scripts.smoke_test --config configs/densenet121_baseline.yaml
python -m scripts.smoke_test --config configs/vit_baseline.yaml
```

### 5.3 Train baseline

ResNet-50 nên được chạy trước để kiểm tra toàn pipeline:

```powershell
python -m scripts.train `
  --config configs/resnet50_baseline.yaml `
  --checkpoint-dir outputs/checkpoints/resnet50_baseline_s701515_seed42_amp `
  --result-dir outputs/metrics/resnet50_baseline_s701515_seed42_amp
```

Sau khi protocol ResNet ổn định, giữ nguyên data/split/training settings và chỉ đổi config/output directory cho DenseNet-121 và ViT-B/16.

Resume một run bị gián đoạn:

```powershell
python -m scripts.train `
  --config configs/resnet50_baseline.yaml `
  --checkpoint-dir outputs/checkpoints/resnet50_baseline_s701515_seed42_amp `
  --result-dir outputs/metrics/resnet50_baseline_s701515_seed42_amp `
  --resume outputs/checkpoints/resnet50_baseline_s701515_seed42_amp/last.pt
```

### 5.4 Đánh giá internal test

Chỉ chạy sau khi model và hyperparameter đã được khóa bằng validation:

```powershell
python -m scripts.evaluate `
  --config configs/resnet50_baseline.yaml `
  --checkpoint outputs/checkpoints/resnet50_baseline_s701515_seed42_amp/best.pt `
  --output-dir outputs/metrics/resnet50_baseline_s701515_seed42_amp/test `
  --split test `
  --threshold 0.5
```

Evaluator từ chối checkpoint không mang đúng `assignment_sha256` của split hiện tại.

## 6. Failure modes và cách chẩn đoán

| Triệu chứng | Nguyên nhân được code kiểm tra | Nơi phát sinh |
|---|---|---|
| `Missing columns` | Annotation CSV thiếu cột bắt buộc | `read_annotations` |
| `Unexpected reader count` | Một ảnh không có đúng ba reader | `get_reader_roster` |
| `BBox outside JPG dimensions` | Bbox không hữu hạn hoặc ra ngoài ảnh | `validate_bbox` |
| `already exists` | Artifact cũ tồn tại nhưng không có `--overwrite` | Các data script |
| `Image leakage` | Hai split bất kỳ trùng `image_id` | `make_splits.py` |
| `Checkpoint split fingerprint mismatch` | Resume/evaluate checkpoint từ protocol split khác | `train.py`, `evaluate.py` |
| Class không có positive/negative | Split không đủ hai label values | Split, loss hoặc metrics |
| `Class order mismatch` | Config, NPZ hoặc checkpoint khác thứ tự finding | Dataset hoặc resume |
| `Manifest/mask label mismatch` | Artifact được sinh từ protocol khác | `CXRDataset` |
| `Nonfinite loss` | Numerical instability hoặc input/model bất thường | `run_epoch` |
| `Model output and label shapes do not match` | Head hoặc class count không đồng bộ | `run_epoch` |

Nguyên tắc xử lý là không sửa artifact thủ công. Cần tìm stage đầu tiên vi phạm contract, xóa hoặc ghi đè có chủ đích các output phụ thuộc rồi chạy lại từ stage đó với cùng config.

## 7. Liên hệ với RQ1 và RQ2

Framework hiện tại tạo nền tảng cho hai câu hỏi nghiên cứu:

- **RQ1 — Classification tốt hơn có đồng nghĩa localization tốt hơn?** Ba backbone dùng cùng split và classification protocol; masks được giữ trong Dataset để sau này so AUROC/AUPRC/F1 với Pointing Game, IoU và Energy Inside Box.
- **RQ2 — Location-aware loss có cải thiện localization không?** `soft_masks`, `union_masks` và `mask_valid` đã nằm trong batch contract, cho phép thêm spatial activation loss mà không thay đổi data pipeline.

Điểm quan trọng là baseline và location-aware experiment phải dùng cùng manifest, split, class order và preprocessing. Nếu những yếu tố này thay đổi giữa hai nhóm, paired comparison không còn hợp lệ.

## 8. Giới hạn hiện tại

- Automated tests mới bao phủ split, projection processed split, fingerprint checkpoint và prediction schema; các module khác vẫn cần thêm coverage.
- Đã có internal test 2.250 ảnh và standalone `evaluate.py`; chưa có result packager tổng quát.
- Chưa có learning-rate scheduler hoặc TensorBoard integration; tqdm progress và timing đã được triển khai.
- `use_augmentation` và `use_clahe` đang là `false`; code cũng chưa triển khai nhánh bật hai tính năng này.
- `image_resize_mode`, `soft_mask_resize_mode` và `grayscale_to_rgb` được ghi trong config/metadata nhưng implementation hiện dùng cố định bilinear, area và grayscale-to-3-channel.
- `training.monitor` khai báo `macro_auroc`, nhưng `train.py` hiện hard-code chính metric này thay vì đọc key.
- `consensus_boxes_path` tồn tại trong config nhưng chưa được script hiện tại sử dụng.
- Chưa có Grad-CAM, localization metrics hoặc location-aware loss implementation.
- Chưa có checkpoint/metrics baseline trên dữ liệu thật trong workspace hiện tại.
- Pipeline thực tế dùng JPG; cần cập nhật hoặc giải thích khác biệt với mô tả DICOM trong Proposal.

## 9. File nên đọc tiếp

Theo thứ tự luồng thực thi:

1. [`configs/default.yaml`](../configs/default.yaml) — protocol và class-order contract.
2. [`src/data/annotations.py`](../src/data/annotations.py) — majority labels và spatial consensus.
3. [`scripts/build_manifest.py`](../scripts/build_manifest.py) — artifact đầu tiên của pipeline.
4. [`scripts/make_splits.py`](../scripts/make_splits.py) — split và reproducibility metadata.
5. [`scripts/preprocess.py`](../scripts/preprocess.py) — ảnh/mask artifact.
6. [`src/data/dataset.py`](../src/data/dataset.py) — runtime sample contract.
7. [`src/models/factory.py`](../src/models/factory.py) — backbone interface.
8. [`src/training/engine.py`](../src/training/engine.py) — train/validation loop.
9. [`scripts/train.py`](../scripts/train.py) — orchestration, early stopping và checkpoint.
10. [`src/metrics/classification.py`](../src/metrics/classification.py) — evaluation contract.

Tài liệu tiến độ tổng thể nằm tại [`docs/Description.md`](./Description.md); milestone nghiên cứu và điều kiện hoàn thành nằm tại [`docs/Proposal.md`](./Proposal.md).
