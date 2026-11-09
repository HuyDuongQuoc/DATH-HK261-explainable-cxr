# Bàn giao artifact Classification cho luồng Grad-CAM

**Ngày chốt:** 2026-11-09  
**Data protocol:** `vindr_s8515_exttest_v1`  
**Mục tiêu:** bàn giao các kết quả classification đã khóa để thành viên phụ trách Grad-CAM có thể chạy luồng XAI song song và ghép kết quả vào bảng RQ1.  
**Phạm vi tài liệu:** chỉ mô tả artifact và contract trao đổi dữ liệu; không hướng dẫn triển khai Grad-CAM.

## 1. Trạng thái bàn giao

- Stage 2 đã hoàn thành đủ `12/12` run: ba architecture × BCE/focal gamma `0.5/1.0/1.5`.
- Mỗi run có đủ 2.250 validation predictions.
- Mỗi run được đánh dấu `complete`, có `best.pt`, `last.pt`, history và metrics.
- Không còn `training_failure.json` đang hiệu lực.
- External test chưa được sử dụng trong Stage 2.
- Một classification checkpoint đã được chọn cho mỗi architecture.

## 2. Kết quả classification đã chốt

| Model | Loss chốt | Positive weighting | Best epoch | Macro-AUPRC | Macro-AUROC | Macro precision | Macro recall | Macro F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| DenseNet-121 | Weighted BCE | sqrt-cap-3 | 13 | 0.519432 | 0.953660 | 0.504388 | 0.619268 | 0.539459 |
| ResNet-50 | Weighted BCE | raw ratio | 21 | 0.528484 | 0.962757 | 0.562494 | 0.591082 | 0.555766 |
| ViT-B/16 | Focal gamma 1.5 | sqrt-cap-3 | 22 | 0.388739 | 0.935464 | 0.451159 | 0.516711 | 0.427370 |

Quyết định này áp dụng cho validation/RQ1 handoff. Không tự động thay checkpoint bằng run có một metric riêng lẻ cao hơn.

## 3. DenseNet-121 artifact bundle

### Run đã chọn

```text
densenet121_stage2v2_bce_sqrtcap3_aug_auprc_s8515_exttest_v1_seed42_fp32
```

### Checkpoint

```text
checkpoint/stage2_focalloss_v2_local_fp32/densenet121/
  densenet121_stage2v2_bce_sqrtcap3_aug_auprc_s8515_exttest_v1_seed42_fp32/
  best.pt
```

Expected SHA-256:

```text
1304394d97a683da86c6907ba101091881496ab6a8108ea49bb96912749ffb47
```

### Resolved runtime config

```text
outputs/stage2_focalloss_v2_local_fp32/runtime_configs/densenet121/
  densenet121_stage2v2_bce_sqrtcap3_aug_auprc_s8515_exttest_v1_seed42_fp32/
  default.yaml
```

### Run artifacts

```text
outputs/stage2_focalloss_v2_local_fp32/densenet121/runs/
  densenet121_stage2v2_bce_sqrtcap3_aug_auprc_s8515_exttest_v1_seed42_fp32/
```

Các file cần giữ cùng bundle:

```text
config.json
best_val_metrics.json
best_val_predictions.csv
loss_metadata.json
history.csv
run_status.json
train_subprocess.log
```

### Lý do chọn

Focal gamma 1.0 đạt macro-AUPRC `0.528095`, nhưng precision tại cùng recall của BCE giảm từ `0.504388` xuống `0.477409` và có bốn finding giảm AP trên `0.02`. Vì vậy DenseNet giữ BCE sqrt-cap-3.

## 4. ResNet-50 artifact bundle

### Run đã chọn

```text
resnet50_stage2v2_bce_raw_aug_auprc_s8515_exttest_v1_seed42_fp32
```

### Checkpoint

```text
checkpoint/stage2_focalloss_v2_local_fp32_resnet50/resnet50/
  resnet50_stage2v2_bce_raw_aug_auprc_s8515_exttest_v1_seed42_fp32/
  best.pt
```

Expected SHA-256:

```text
92cc3c907c84549ac0a37a22236221582fb30159587c45ee7352510c64beb235
```

### Resolved runtime config

```text
outputs/stage2_focalloss_v2_local_fp32_resnet50/runtime_configs/resnet50/
  resnet50_stage2v2_bce_raw_aug_auprc_s8515_exttest_v1_seed42_fp32/
  default.yaml
```

### Run artifacts

```text
outputs/stage2_focalloss_v2_local_fp32_resnet50/resnet50/runs/
  resnet50_stage2v2_bce_raw_aug_auprc_s8515_exttest_v1_seed42_fp32/
```

Các file cần giữ cùng bundle:

```text
config.json
best_val_metrics.json
best_val_predictions.csv
loss_metadata.json
history.csv
run_status.json
train_subprocess.log
```

### Lý do chọn

Focal gamma 0.5 đạt macro-AUPRC `0.545510`, nhưng precision tại cùng recall của BCE giảm từ `0.562494` xuống `0.494003`. Cải thiện AP cũng tập trung mạnh ở một số finding rất hiếm, trong khi Calcification, Consolidation và Pneumothorax giảm. Vì vậy ResNet giữ BCE raw ratio.

## 5. ViT-B/16 artifact bundle

### Run đã chọn

```text
vit_b_16_stage2v2_focal_g15_sqrtcap3_aug_auprc_s8515_exttest_v1_seed42_fp32
```

### Checkpoint

```text
checkpoint/stage2_focalloss_v2_local_fp32_vit_b16/vit_b_16/
  vit_b_16_stage2v2_focal_g15_sqrtcap3_aug_auprc_s8515_exttest_v1_seed42_fp32/
  best.pt
```

Expected SHA-256:

```text
986101faffbd822d10f8fe2f9bb58dad0e3ad99fa9f64ac83832f88a164d3238
```

### Resolved runtime config

```text
outputs/stage2_focalloss_v2_local_fp32_vit_b16/runtime_configs/vit_b_16/
  vit_b_16_stage2v2_focal_g15_sqrtcap3_aug_auprc_s8515_exttest_v1_seed42_fp32/
  default.yaml
```

### Run artifacts

```text
outputs/stage2_focalloss_v2_local_fp32_vit_b16/vit_b_16/runs/
  vit_b_16_stage2v2_focal_g15_sqrtcap3_aug_auprc_s8515_exttest_v1_seed42_fp32/
```

Các file cần giữ cùng bundle:

```text
config.json
best_val_metrics.json
best_val_predictions.csv
loss_metadata.json
history.csv
run_status.json
train_subprocess.log
```

### Lý do chọn

So với BCE sqrt-cap-3, focal gamma 1.5:

```text
delta macro-AUPRC:                 +0.016704
delta tuned macro precision:       +0.074588
delta macro recall:                +0.000070
delta macro F1:                    +0.019546
delta precision at control recall: +0.009727
classes AP drop > 0.02:             2
stable validation peak:             true
```

Đây là focal run duy nhất trong ba architecture đạt đầy đủ selection gate.

## 6. Analysis artifacts nguồn

### DenseNet-121

```text
outputs/stage2_focalloss_v2_local_fp32/densenet121/analysis/
```

### ResNet-50

```text
outputs/stage2_focalloss_v2_local_fp32_resnet50/resnet50/analysis/
```

### ViT-B/16

```text
outputs/stage2_focalloss_v2_local_fp32_vit_b16/vit_b_16/analysis/
```

Trong mỗi analysis directory, các file cần dùng để đối chiếu gồm:

```text
stage2_macro_comparison.csv
stage2_per_class_comparison.csv
stage2_delta_vs_bce.csv
stage2_ranking.csv
stage2_selection.json
stage2_analysis.json
stage2_analysis_report.md
```

`stage2_per_class_comparison.csv` là nguồn chính để ghép classification metric theo model/finding vào bảng Grad-CAM/RQ1.

Lưu ý: khi không focal run nào đủ eligibility, trường `runner_up` trong selection artifact đi theo thứ tự registry và không nhất thiết là run có AUPRC cao thứ hai. Không dùng `runner_up` làm checkpoint bàn giao.

## 7. Validation data artifacts

```text
Protocol ID:
vindr_s8515_exttest_v1

Validation manifest:
data/protocols/vindr_s8515_exttest_v1/val_manifest.csv

Processed validation root:
data/processed/consensus_v1_224

Validation size:
2250 images
```

Manifest chứa:

```text
image_id
image_path
mask_path
14 finding label columns
```

Mask NPZ được tham chiếu qua `mask_path` và chứa:

```text
class_names
soft_masks
union_masks
labels
mask_valid
positive_reader_counts
```

Khi ghép kết quả, `image_id` trong manifest, prediction artifact và Grad-CAM artifact phải trùng tuyệt đối.

## 8. Thứ tự findings đã khóa

Không được đổi tên hoặc đổi thứ tự:

```text
0  Aortic enlargement
1  Atelectasis
2  Calcification
3  Cardiomegaly
4  Consolidation
5  ILD
6  Infiltration
7  Lung Opacity
8  Nodule/Mass
9  Other lesion
10 Pleural effusion
11 Pleural thickening
12 Pneumothorax
13 Pulmonary fibrosis
```

## 9. Prediction artifact contract

Mỗi `best_val_predictions.csv` có 2.250 unique image IDs và các nhóm cột:

```text
image_id
target::<finding>
probability::<finding>
```

Classification decision threshold và per-class metric nằm trong:

```text
best_val_metrics.json
best.pt
```

Threshold strategy hiện tại:

```text
per_class_max_f1
```

Threshold được tune trên validation và gắn với validation manifest hash trong checkpoint. Không tạo classification threshold mới trong luồng Grad-CAM.

## 10. Contract ghép Classification và Grad-CAM

Khóa join cấp image/finding:

```text
model
image_id
finding
```

Khóa join cấp bảng tổng hợp:

```text
model
finding
```

Artifact Grad-CAM bàn giao ngược lại tối thiểu cần các trường:

```text
model
image_id
finding
checkpoint_sha256
mask_valid
localization metrics
```

Bảng RQ1 hợp nhất dự kiến:

```text
model
finding
n_validation_positive
n_valid_localization
classification_auroc
classification_average_precision
classification_precision
classification_recall
classification_f1
hit_rate
mean_iou
mean_eib
checkpoint_sha256
```

Không đổi tên model giữa hai luồng:

```text
densenet121
resnet50
vit_b_16
```

## 11. External-test gate

Các artifact dưới đây chưa được phép dùng trong giai đoạn chạy song song hiện tại:

```text
data/protocols/vindr_s8515_exttest_v1/external_test_manifest.csv
data/processed/consensus_test_v1_224
```

Luồng Grad-CAM hiện chỉ được chạy trên validation.

External test chỉ được mở sau khi nhóm đã khóa:

- Ba checkpoint và SHA-256.
- Classification thresholds.
- Grad-CAM protocol.
- Localization metrics/thresholds.
- Source hash.
- Final test plan.

Không được thay checkpoint hoặc classification threshold sau khi đã xem external-test result.

## 12. Checklist giao/nhận

### Classification side

- [x] Hoàn thành 12/12 Stage 2 runs.
- [x] Chọn một checkpoint cho mỗi architecture.
- [x] Ghi expected checkpoint SHA-256.
- [x] Ghi resolved runtime config path.
- [x] Có 2.250 validation predictions cho mỗi selected run.
- [x] Có macro và per-class classification metrics.
- [x] External test chưa được dùng.

### Grad-CAM side xác nhận khi nhận

- [ ] Ba `best.pt` tồn tại.
- [ ] SHA-256 của từng checkpoint khớp tài liệu.
- [ ] Ba resolved runtime YAML tồn tại.
- [ ] Ba `best_val_predictions.csv` có 2.250 unique image IDs.
- [ ] Finding names/order khớp tuyệt đối.
- [ ] Validation manifest và processed mask root có thể truy cập.
- [ ] Output Grad-CAM giữ `model`, `image_id`, `finding` để join.
- [ ] Chỉ sử dụng validation trước final unlock.

## 13. Kết luận bàn giao

Ba classification artifact chính thức để chạy song song với Grad-CAM là:

```text
DenseNet-121: weighted BCE + sqrt-cap-3
ResNet-50:    weighted BCE + raw ratio
ViT-B/16:     focal gamma 1.5 + sqrt-cap-3
```

Mọi bảng RQ1 phải truy vết được về đúng checkpoint SHA-256, validation manifest và per-class classification artifact được liệt kê trong tài liệu này.

