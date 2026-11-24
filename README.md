# CXR Location-Aware XAI

Nghiên cứu phân loại đa nhãn và explainable AI trên ảnh X-quang ngực VinDr-CXR. Mục tiêu cuối của dự án là so sánh chất lượng phân loại với khả năng định vị tổn thương, sau đó đánh giá location-aware learning trên CNN và Vision Transformer.

> **Trạng thái hiện tại:** codebase đã có data protocol tái lập được, pipeline tiền xử lý, ba baseline phân loại (ResNet-50, DenseNet-121, ViT-B/16), train/resume và evaluate. Grad-CAM, localization metrics (Pointing Game, EIB, IoU), location-aware loss và lưới λ cũng đã có. Streamlit prototype vẫn thuộc roadmap.

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
| Grad-CAM (ResNet-50, DenseNet-121, ViT-B/16) và localization metrics | Đã có |
| Attribution độc lập với loss (Grad-CAM lớp áp chót, occlusion) và box-occlusion test | Đã có |
| Prior giải phẫu, metric chuẩn hóa theo diện tích box, bootstrap CI, so sánh paired | Đã có |
| Location-aware loss `L_cls + λ·L_loc` (mask union/majority/soft, softplus/relu) | Đã có |
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

`app/` hiện chưa có implementation chức năng.

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

## Đánh giá localization

```bash
python scripts/evaluate_localization.py \
  --config configs/resnet50_location.yaml \
  --checkpoint outputs/full/runs/r50_l1/ckpt/best.pt \
  --output-dir outputs/full/runs/r50_l1/loc_test_occlusion \
  --split test --method occlusion --box-occlusion --iou-threshold 0.30
```

- `--method gradcam` (mặc định) dùng feature map cuối. Với GAP + linear, heatmap này trùng CAM mà `L_loc` tối ưu, nên **không độc lập với loss**. Khi báo cáo tác động của `L_loc`, cần kèm `gradcam_penultimate` (layer3/denseblock3) hoặc `occlusion`.
- `--box-occlusion` đo logit giảm khi che toàn bộ vùng trong box (`drop_inside`) và ngoài box (`drop_outside`): phép thử hành vi, không qua heatmap.
- Prior giải phẫu (trung bình union mask train theo finding, không nhìn ảnh) được build một lần và cache ở `outputs/priors/train_union_prior.npz`. Records có `prior_eib`, `prior_hit`, `eib_minus_prior`, `prior_corr`. Mô hình nên được so với prior, không chỉ với 0.
- `box_area` là mức ngẫu nhiên của EIB/Pointing; `eib_over_area` so được giữa validation (union của 3 reader, box to hơn) và external test (một consensus, box nhỏ hơn).
- `iou_mean` (trung bình mọi ngưỡng) hoặc `--iou-threshold` cố định giúp so IoU giữa các run mà không phụ thuộc ngưỡng chọn riêng từng run.
- `localization_summary.json` có CI bootstrap theo ảnh cho các metric chính.

So sánh paired hai run (localization và macro-AUROC):

```bash
python scripts/compare_runs.py --config configs/resnet50_location.yaml \
  --a-loc <baseline>/test_localization_records.csv --b-loc <guided>/test_localization_records.csv \
  --a-pred <baseline>/test_predictions.csv --b-pred <guided>/test_predictions.csv \
  --output outputs/comparisons/base_vs_guided.json
```

## Tùy chọn location-aware loss

Trong `localization_loss` của config:

- `mask`: `union` (mặc định, ≥1 reader), `majority` (≥2/3 reader), `soft` (tỉ lệ reader vẽ box làm trọng số).
- `activation`: `softplus` (mặc định) hoặc `relu` (như Energy loss của Rao et al., ICCV 2023).

`model.dilate_last_stage: true` (chỉ ResNet-50) cho CAM 14×14 thay vì 7×7 mà không thêm tham số. Các giá trị mặc định giữ nguyên hành vi của các run đã có.

### ViT-GAP

`vit_b_16_gap` ([`src/models/vit_gap.py`](src/models/vit_gap.py)) giữ backbone ViT-B/16 ImageNet nhưng phân loại bằng trung bình các patch token thay vì CLS token. CAM 14×14 = head áp lên từng patch token, logits = trung bình CAM, nên `L_loc`, Grad-CAM và các metric dùng y như CNN. Đây là kiến trúc khác `vit_b_16` (head CLS): ablation phải dùng cặp `configs/vit_gap_baseline.yaml` (λ=0) và `configs/vit_gap_location.yaml`.

## Chạy và phân tích RQ1/RQ2 trên Modal

```bash
# train (mỗi (model, λ, seed) một container); MODAL_GPU mặc định L4, các run hiện tại dùng A10G
MODAL_GPU=A10G ~/.venvs/modal/bin/python -m modal run --detach scripts/modal_sweep.py::sweep --model vit_b_16 --prefix vit --lambdas 0 --seeds 42,43
# đánh giá external test: classification, Grad-CAM (+prior), occlusion + box-occlusion; config đọc lại từ res/config.json
MODAL_GPU=A10G ~/.venvs/modal/bin/python -m modal run --detach scripts/modal_sweep.py::evaluate_runs --runs r50_base_s42,r50_l1_s42
# tải về và phân tích
~/.venvs/modal/bin/python -m modal volume get cxr-runs / outputs/modal_runs/volume
python scripts/rq_analysis.py outputs/modal_runs/volume --output-dir outputs/rq
```

`rq_analysis.py` ghép run theo model/λ/seed trong config và xuất:

- **RQ1** (`rq1_<method>.md`): bảng theo kiến trúc (baseline λ=0), Spearman *trong từng finding* giữa AUROC/AUPRC và Pointing/EIB/EIB−prior (CI bootstrap theo finding), và bảng can thiệp (λ>0 trừ baseline cùng kiến trúc + seed).
- **RQ2** (`rq2.md`): so sánh paired baseline vs λ>0 với CI bootstrap theo ảnh cho Grad-CAM, occlusion, box-occlusion và macro-AUROC, kèm bảng theo finding.

## Classification loss cho dữ liệu mất cân bằng

Trong `training.loss` (xem chú thích ở `configs/default.yaml`):

- `name`: `weighted_bce`, `sigmoid_focal` hoặc `asymmetric` (ASL, Ridnik et al. 2021: `gamma_pos`, `gamma_neg`, `clip`).
- `positive_weight.mode`: `raw_ratio`, `sqrt_ratio`, `sqrt_ratio_capped`, `class_balanced` (Cui et al. 2019, cần `beta`) hoặc `none`.
- `label_policy` (chỉ khi train): `majority` (mặc định), `ignore_minority` (bỏ cặp ảnh–finding chỉ 1/3 reader đánh dấu) hoặc `soft_votes` (target = số reader / 3). Số vote đọc từ `data.annotation_csv`.
- `training.prior_bias_init: true`: khởi tạo bias của head theo tỉ lệ dương (đã tính `pos_weight`).

Các config `configs/loss_r50_*.yaml` là ma trận ablation trên ResNet-50 (λ=0), giữ protocol của lưới λ. Giá trị mặc định giữ nguyên loss protocol của các checkpoint đã có.

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
