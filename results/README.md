# results/

## Nội dung

| File | Mô tả |
|---|---|
| [`full_lambda_grid.csv`](full_lambda_grid.csv) | Kết quả chạy lưới λ cho `L_total = L_cls + λ·L_loc`: ResNet50, toàn bộ dataset. Mỗi dòng là một run, gồm các cột đo trên **validation** (`val_*`, dùng để chọn λ) và các cột đo trên **external test** (`test_*`, chỉ để xác nhận). |

## Cột của `full_lambda_grid.csv`

| Cột | Ý nghĩa |
|---|---|
| `dataset` | Luôn là `full` (toàn bộ dataset: train 12.750, validation 2.250 ảnh) |
| `run` | Tên run |
| `model` | Backbone: `resnet50` hoặc `densenet121` |
| `lambda` | Hệ số λ của `L_loc`; `0` là baseline (chỉ `L_cls`) |
| `seed` | Seed huấn luyện |
| `gpu` | GPU đã chạy run |
| `role` | `grid` là một điểm của lưới λ (ResNet-50); `experiment2` là run của thí nghiệm 2 trên DenseNet-121 (baseline và λ=1, λ chọn trên ResNet, không tune lại); `baseline_replicate` là baseline lặp lại ở seed khác, chỉ để đối chiếu mức nhiễu |
| `epochs_run` | Số epoch đã chạy (early stopping có thể dừng sớm) |
| `best_epoch` | Epoch có macro-AUROC validation cao nhất (checkpoint được chọn) |
| `val_macro_auroc`, `val_macro_auprc` | Classification: AUROC và AUPRC, trung bình trên 14 finding |
| `val_macro_precision`, `val_macro_recall`, `val_macro_f1_tuned` | Precision, Recall, F1 ở ngưỡng từng class tối ưu F1 trên validation (nên hơi lạc quan) |
| `iou_threshold_selected_on_val` | Ngưỡng nhị phân hóa heatmap để tính IoU, chọn trên validation; **cùng ngưỡng này được dùng cho các cột `test_*`** |
| `val_loc_n_pairs` | Số cặp (ảnh, finding) có bbox hợp lệ dùng để tính localization |
| `val_pointing_hit_rate` | Pointing Game: điểm nóng nhất của Grad-CAM có nằm trong bbox không |
| `val_eib` | Energy Inside Box: tỉ lệ năng lượng của Grad-CAM nằm trong bbox |
| `val_iou` | IoU giữa vùng sáng của Grad-CAM và bbox |
| `test_macro_auroc`, `test_macro_auprc`, `test_macro_precision`, `test_macro_recall`, `test_macro_f1` | Classification trên external test (3.000 ảnh). Precision, Recall, F1 dùng ngưỡng từng class đã lưu trong checkpoint (tối ưu trên validation) |
| `test_loc_n_pairs` | Số cặp (ảnh, finding) có bbox hợp lệ trên external test |
| `test_pointing_hit_rate`, `test_eib`, `test_iou` | Localization trên external test, cùng định nghĩa như các cột `val_*` |
| `passes_auroc_constraint` | `True` nếu AUROC không thấp hơn baseline quá 0,01 (chỉ có nghĩa với λ > 0; baseline luôn `False`) |
| `selected_by_preregistered_rule` | `True` cho λ có EIB cao nhất trong các λ qua ràng buộc AUROC (quy tắc chốt trước). **Không có nghĩa là λ\* đã được chốt** |

## Lưu ý khi đọc

- **Chỉ một seed (42) cho lưới λ**, nên khác biệt giữa các λ gần nhau (đặc biệt EIB, nhiễu khoảng 0,02 đến 0,03) chưa kết luận được.
- Baseline và λ=0,25 chạy trên RTX 3060; các λ còn lại trên A10G (Modal). Khác biệt giữa hai nhóm có lẫn nhiễu môi trường.
- Dòng `r50_base_s43` (seed 43, GPU L4) chỉ để đối chiếu nhiễu, không thuộc lưới; nó có số test để ước lượng nhiễu trên test.
- **Các cột `test_*` chỉ để xác nhận, không dùng để chọn λ.** λ được chọn trên các cột `val_*`. 
- Annotation của external test khác train: một reader consensus (mask nhị phân), không phải majority vote của ba bác sĩ.
- Số liệu gốc (checkpoint, log, kết quả chi tiết theo finding và theo epoch) nằm trong `outputs/full/runs/` (run local) và `outputs/modal_runs/pull/` (run Modal).

## Quyết định λ\*

**λ\* = 1**, chốt trên các cột `val_*` **trước khi xem các cột `test_*`**.

- **Tiêu chí:** (1) ràng buộc: AUROC validation không thấp hơn baseline quá 0,01; (2) mục tiêu: hạng trung bình của Pointing, EIB, IoU trên validation tốt nhất, hòa thì chọn λ nhỏ hơn. λ=1 có hạng trung bình 2,00 (λ=10: 2,33; λ=2: 3,00).
- **Khác với quy tắc chốt trước ban đầu** (EIB cao nhất, cho λ=10; cột `selected_by_preregistered_rule`). Tiêu chí được đổi sau khi đã thấy kết quả validation, vì EIB bão hòa từ λ≈1 (chênh 0,026 giữa λ=1 và λ=10, nằm trong nhiễu), còn IoU và AUPRC giảm ở λ lớn.
- Hạn chế: một seed, lưới trộn hai loại GPU, nên λ\* này là sơ bộ cho ResNet-50.

### Kết quả xác nhận trên external test (xem sau khi đã chốt λ\*)

- **Localization cải thiện rõ trên test**: λ=1 so với baseline, Pointing 0,216 → 0,413, EIB 0,116 → 0,271, IoU 0,115 → 0,201.
- **Classification trên test thấp hơn baseline ở mọi λ > 0** (AUROC giảm 0,008 đến 0,030; λ=1 giảm 0,0265, vượt dung sai 0,01 đã đặt trên validation). Điều này **không thấy trên validation** (nơi AUROC không giảm). Một seed, chưa có baseline lặp trên test để biết mức nhiễu, nên chưa kết luận được đây là đánh đổi thật hay nhiễu.
- Trên test, λ=10 có localization và AUROC tốt hơn λ=1. **Không đổi λ\*** theo test, nhưng cần ghi rõ khi báo cáo.
- Số tuyệt đối trên test thấp hơn nhiều so với validation (AUROC baseline 0,887 so với 0,955), do khác biệt nguồn và cách tạo annotation (một reader consensus).

### Thí nghiệm 2 (một seed, external test): baseline so với λ=1

| | AUROC | AUPRC | Pointing | EIB | IoU |
|---|---:|---:|---:|---:|---:|
| ResNet-50 baseline | 0,887 | 0,341 | 0,216 | 0,116 | 0,115 |
| ResNet-50 λ=1 | 0,861 | 0,320 | 0,413 | 0,271 | 0,201 |
| DenseNet-121 baseline | 0,880 | 0,322 | 0,168 | 0,085 | 0,114 |
| DenseNet-121 λ=1 | 0,896 | 0,371 | 0,404 | 0,248 | 0,201 |

- Localization tăng rõ ở cả hai backbone (Pointing +0,20 và +0,24; EIB +0,16 và +0,16), lớn hơn nhiễu giữa hai baseline ResNet (seed 42 và 43) trên test (0,01 đến 0,03).
- **Ảnh hưởng lên classification không nhất quán giữa hai backbone**: AUROC giảm 0,027 ở ResNet nhưng tăng 0,016 ở DenseNet (nhiễu giữa hai baseline ResNet trên test: 0,012). Chưa kết luận được có đánh đổi hệ thống hay không.
- Ngưỡng IoU chọn trên validation khác nhau giữa các run (DenseNet baseline: 0,70), nên so IoU giữa các run cần thận trọng; Pointing và EIB không phụ thuộc ngưỡng.
- Một seed; DenseNet chưa được tune λ riêng.
