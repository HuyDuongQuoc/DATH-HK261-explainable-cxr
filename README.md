# CXR Location-Aware XAI

Nghiên cứu phân loại đa nhãn và explainable AI trên ảnh X-quang ngực VinDr-CXR. Mục tiêu cuối của dự án là so sánh chất lượng phân loại với khả năng định vị tổn thương, sau đó đánh giá location-aware learning trên CNN và Vision Transformer.

> **Trạng thái hiện tại:** codebase đã có data protocol tái lập được, pipeline tiền xử lý, ba baseline phân loại (ResNet-50, DenseNet-121, ViT-B/16), train/resume và evaluate. Grad-CAM, localization metrics, location-aware loss và Streamlit prototype vẫn thuộc roadmap, chưa được implement trong source hiện tại.

## Phạm vi đã implement

| Thành phần | Trạng thái |
| --- | --- |
| Tổng hợp nhãn 14 findings từ annotation của ba reader | Đã có |
| Tiền xử lý ảnh 224×224 và tạo soft/union masks | Đã có |
| Protocol train/validation cố định + external test | Đã có |
| ResNet-50, DenseNet-121, ViT-B/16 classification baseline | Đã có |
| Weighted BCE / sigmoid focal, AdamW, AMP, early stopping, resume checkpoint | Đã có |
| Train-time augmentation bảo thủ và chọn checkpoint bằng macro-AUPRC | Đã có |
| Threshold F1 riêng cho từng lớp, chỉ calibration trên validation | Đã có |
| AUROC, Average Precision, Precision, Recall và F1 | Đã có |
| Grad-CAM và localization metrics | Chưa có |
| Location-aware loss | Chưa có |
| Streamlit prototype | Chưa có |

Roadmap nghiên cứu đầy đủ nằm trong [`docs/Proposal.md`](docs/Proposal.md).

## Data protocol

Protocol hiện hành là `vindr_s8515_exttest_v1`:

- 12.750 ảnh train và 2.250 ảnh validation cố định từ corpus 15.000 ảnh có annotation của ba reader.
- 3.000 ảnh external test có consensus ground truth, không trùng train/validation và chỉ dùng cho final evaluation.
- 14 output classes là các finding bất thường; `No finding` được giữ làm mẫu âm cho cả 14 classes, không phải một output class.
- Nhãn train/validation dùng hard majority vote (`2/3` reader). Soft mask được tạo bằng cách union bbox theo từng reader rồi lấy trung bình trên toàn bộ reader.
- External test dùng một consensus reader và binary resized masks.
- Split assignments, metadata và checksums được version-control trong [`data/protocols/vindr_s8515_exttest_v1`](data/protocols/vindr_s8515_exttest_v1). Ảnh đã xử lý, masks và checkpoints không được commit.

## Cấu trúc repository

```text
configs/        Cấu hình mặc định và config cho ba baseline
data/           Annotation, protocol artifacts và thư mục dữ liệu local
notebooks/      EDA, Colab runner và orchestration thí nghiệm
scripts/        CLI chuẩn bị dữ liệu, train, evaluate và verify
src/data/       Annotation consensus, preprocessing, dataset và protocol
src/models/     Model factory cho ResNet-50, DenseNet-121 và ViT-B/16
src/losses/     Weighted BCE và sigmoid focal classification loss
src/metrics/    Classification metrics và validation-tuned thresholds
src/training/   Training/evaluation epoch engine
src/utils/      Config, seed, checkpoint và checksum helpers
tests/          Test cho split, protocol, threshold, evaluate và training
```

`src/xai/` và `app/` hiện chưa có implementation chức năng.

## Cài đặt

Yêu cầu một môi trường Python có thể cài PyTorch. File `requirements.txt` đang pin PyTorch `2.7.0+cu126` và torchvision `0.22.0+cu126` cho CUDA 12.6.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Trên Linux/macOS, thay lệnh activate bằng:

```bash
source .venv/bin/activate
```

Nếu runtime không dùng CUDA 12.6, hãy cài bản PyTorch phù hợp trước rồi cài các package chung:

```bash
pip install -r requirements-common.txt
```

## Chuẩn bị dữ liệu

### 1. Tải ảnh từ Kaggle

Tạo `.env` từ file mẫu và điền Kaggle credentials:

```powershell
Copy-Item .env.example .env
python scripts/prepare_data.py
```

Script tải dataset `sunghyunjun/vinbigdata-1024-jpg-dataset` vào `data/raw`. Không commit `.env` hoặc Kaggle token.

Sau khi tải, code kỳ vọng tối thiểu:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/train/*.jpg
data/raw/test/*.jpg
```

### 2. Materialize protocol đã chốt

Các validation IDs và split assignments đã được commit. Workflow thông thường không tạo split mới:

```powershell
python scripts/build_manifest.py --config configs/default.yaml
python scripts/preprocess.py --config configs/default.yaml --overwrite
python scripts/preprocess_external_test.py --config configs/default.yaml --overwrite
python scripts/verify_data_protocol.py --config configs/default.yaml
```

`--overwrite` ở hai lệnh preprocessing cho phép tái tạo các manifest protocol đã được version-control. Chỉ dùng `scripts/make_splits.py --overwrite` khi chủ động thay đổi data protocol; thao tác đó có thể làm thay đổi split/checksum chuẩn của dự án.

## Kiểm tra nhanh

Chạy test suite không cần train model:

```powershell
python -m pytest -q
```

Kiểm tra forward/backward và checkpoint round-trip bằng dữ liệu synthetic:

```powershell
python scripts/smoke_test.py --config configs/resnet50_baseline.yaml
```

Có thể thay config bằng `configs/densenet121_baseline.yaml` hoặc `configs/vit_baseline.yaml`.

## Train baseline

Ví dụ với ResNet-50:

```powershell
python scripts/train.py `
  --config configs/resnet50_baseline.yaml `
  --checkpoint-dir outputs/checkpoints/resnet50 `
  --result-dir outputs/metrics/resnet50
```

Resume từ checkpoint cuối:

```powershell
python scripts/train.py `
  --config configs/resnet50_baseline.yaml `
  --checkpoint-dir outputs/checkpoints/resnet50 `
  --result-dir outputs/metrics/resnet50 `
  --resume outputs/checkpoints/resnet50/last.pt
```

Training sử dụng weighted BCE và AdamW, chọn best checkpoint theo validation macro-AUROC. Mỗi epoch calibration một threshold tối đa F1 cho từng class trên validation; threshold và hash của validation manifest được lưu trong checkpoint.

## Train với augmentation + AUPRC

Ba config thí nghiệm mới bật augmentation on-the-fly chỉ trên train set và chọn best checkpoint bằng validation macro-AUPRC:

```text
configs/resnet50_aug_auprc.yaml
configs/densenet121_aug_auprc.yaml
configs/vit_aug_auprc.yaml
```

Ví dụ chạy ResNet-50 trong output directory tách biệt với baseline:

```powershell
python scripts/train.py `
  --config configs/resnet50_aug_auprc.yaml `
  --checkpoint-dir outputs/checkpoints/aug_auprc_v1/resnet50 `
  --result-dir outputs/metrics/aug_auprc_v1/resnet50
```

Augmentation gồm affine nhỏ (±7°, translation 3%, scale 0.95–1.05) và brightness/contrast ±10% với xác suất 0.5. Không dùng flip hoặc crop. Validation và external test không được augmentation.

Notebook `notebooks/02_experiment.ipynb` lưu run mới dưới `checkpoints/aug_auprc_v1` và `run_logs/aug_auprc_v1` trên Drive. Vì augmentation chạy lúc load ảnh, hai archive `consensus_v1_224.zip` và `consensus_test_v1_224.zip` không đổi và không cần upload lại nếu đã tồn tại và vượt qua bước kiểm tra checksum.

## Stage 1: positive weighting với notebook 03

Notebook `notebooks/03_testing.ipynb` chạy ma trận 12 validation experiments gồm ba architecture và bốn positive-weighting: raw ratio, square-root ratio, square-root ratio cap 5 và cap 3. Tất cả run giữ seed 42, augmentation hiện hành, train batch 8 và chọn checkpoint bằng validation macro-AUPRC.

Artifact trên Drive được tách khỏi notebook 02:

```text
MyDrive/DATH_CXR/checkpoints/aug_auprc_v1/03_testing/stage1_pos_weight_v1/
MyDrive/DATH_CXR/run_logs/aug_auprc_v1/03_testing/stage1_pos_weight_v1/
```

Notebook tự resume `last.pt`, skip run hoàn tất, cập nhật bảng so sánh/figures sau từng run và tạo export:

```text
/content/exports/03_testing_stage1_pos_weight_v1_results.zip
```

`RUN_FINAL_TEST=False` là mặc định và Stage 1 không giải nén hoặc đánh giá external test. Có thể chia nhiều phiên Colab bằng `ACTIVE_MODELS`, `ACTIVE_WEIGHTINGS` và `MAX_NEW_RUNS_PER_SESSION`; registry 12 run và tiêu chí lựa chọn không thay đổi.

Mỗi run lưu thêm `loss_metadata.json` và `best_val_predictions.csv`. Báo cáo tổng hợp nằm trong `analysis/`, gồm macro/per-class metrics, delta so với raw control, ranking, selection JSON, Markdown report và figures. Dataset archive không thay đổi và không cần upload lại.

## Stage 2: focal loss với ba notebook 04

Stage 2 khóa positive-weighting theo kết quả Stage 1 và train lại một BCE control trong cùng protocol với ba focal run:

| Notebook | Model | Positive weighting | Runs |
| --- | --- | --- | --- |
| `notebooks/04_1_densenet121_focalloss.ipynb` | DenseNet-121 | sqrt-cap-3 | BCE, focal γ=0.5/1.0/1.5 |
| `notebooks/04_2_resnet50_focalloss.ipynb` | ResNet-50 | raw ratio | BCE, focal γ=0.5/1.0/1.5 |
| `notebooks/04_3_vit_b16_focalloss.ipynb` | ViT-B/16 | sqrt-cap-3 | BCE, focal γ=0.5/1.0/1.5 |

ViT dùng sqrt-cap-3 theo phân tích Stage 1, chủ động override kết quả automatic selection ban đầu. Mỗi notebook chạy trên một Colab T4 riêng; có thể mở đồng thời ba session. Cả ba phải dùng cùng commit/dependency/data contract và đúng runtime Stage 1 đã khóa. Nếu runtime hoặc shared contract lệch, notebook dừng trước training.

Artifact được tách theo model để không có ghi đè chéo:

```text
MyDrive/DATH_CXR/checkpoints/aug_auprc_v1/03_testing/stage2_focalloss/
├── densenet121/
├── resnet50/
└── vit_b_16/

MyDrive/DATH_CXR/run_logs/aug_auprc_v1/03_testing/stage2_focalloss/
├── stage2_contract.json
├── densenet121/{runs,analysis,hardware}/
├── resnet50/{runs,analysis,hardware}/
└── vit_b_16/{runs,analysis,hardware}/
```

Để chạy, người dùng tự push source đã review lên branch `update_data`, mở từng notebook trong ba Colab T4 session rồi chọn **Run all**. Khi session bị ngắt, chạy lại cùng notebook: run hoàn tất được skip, run dở resume từ `last.pt`. Không xóa hoặc đổi tên thư mục checkpoint giữa các lần chạy.

Mỗi notebook cập nhật partial analysis sau từng run và tạo export riêng dưới `/content/exports/`. Export chỉ chứa metrics, validation predictions, reports, figures, manifest và hardware telemetry; không chứa checkpoint, dataset, NPZ hoặc ZIP nguồn. Stage 2 không khai báo hay giải nén external-test archive và luôn khóa `RUN_FINAL_TEST=False`.

Focal loss sử dụng weighted BCE theo phần tử, nhân với `(1-p_t)^gamma`, rồi lấy mean trên batch × class. Không dùng alpha vì `pos_weight` đã xử lý mất cân bằng lớp. Checkpoint lưu toàn bộ loss protocol, gồm gamma và positive-weighting; resume/evaluate từ protocol khác bị từ chối.

## Evaluate

Evaluate validation:

```powershell
python scripts/evaluate.py `
  --config configs/resnet50_baseline.yaml `
  --checkpoint outputs/checkpoints/resnet50/best.pt `
  --output-dir outputs/metrics/resnet50/val `
  --split val
```

Final evaluation trên external test:

```powershell
python scripts/evaluate.py `
  --config configs/resnet50_baseline.yaml `
  --checkpoint outputs/checkpoints/resnet50/best.pt `
  --output-dir outputs/metrics/resnet50/test `
  --split test
```

Mặc định evaluate dùng threshold đã calibration trên validation từ checkpoint. Không dùng `--threshold` khi báo cáo kết quả chuẩn, trừ khi đang thực hiện một thí nghiệm override có chủ đích.

Các output chính gồm:

- `best.pt`, `last.pt` trong checkpoint directory.
- `config.json`, `history.csv`, `best_val_metrics.json`, `latest_val_metrics.json` trong result directory.
- `{split}_metrics.json`, `{split}_predictions.csv`, `evaluation_metadata.json` trong evaluation directory.

## Luồng xử lý

```text
Kaggle JPG + CSV annotations
        ↓
majority-label manifest + fixed 85/15 assignment
        ↓
224×224 images + soft/union masks
        ↓
ResNet-50 / DenseNet-121 / ViT-B/16
        ↓
validation metrics + per-class F1 thresholds
        ↓
protocol-bound checkpoint
        ↓
validation hoặc external-test evaluation
```

## Lưu ý tái lập thí nghiệm

- Seed mặc định là `42`.
- Checkpoint bị ràng buộc với model name, thứ tự 14 findings, protocol ID và hash split assignment.
- External test không dùng để chọn model hoặc calibration threshold.
- `configs/default.yaml` là base config; các config baseline chỉ override `model.name`.
- `requirements-common.txt` không pin toàn bộ phiên bản package, vì vậy cần lưu runtime/package snapshot cùng kết quả thí nghiệm nếu yêu cầu reproducibility tuyệt đối.
