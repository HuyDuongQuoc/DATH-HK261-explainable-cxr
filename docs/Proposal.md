Dựa trên proposal hiện tại, tui khuyên nên quản lý ĐATH theo kiểu **research project có milestone rõ ràng**, thay vì chia đơn giản theo “code model → làm web → viết báo cáo”. Lý do là đề tài của mình có hai câu hỏi nghiên cứu RQ1 và RQ2, nên từng phase phải tạo ra bằng chứng để trả lời được hai câu này. Proposal hiện tại cũng đã xác định pipeline từ EDA → baseline → localization → location-aware loss → analysis → prototype. 

## Tổng thể project

| Phase | Tên phase                | Mục tiêu chính                                      | Output quan trọng              |
| ----- | ------------------------ | --------------------------------------------------- | ------------------------------ |
| 0     | Project Setup            | Chuẩn hóa repo, môi trường, cách quản lý experiment | Repo chạy được, config, README |
| 1     | Data Understanding & EDA | Hiểu VinDr-CXR và chọn finding                      | Dataset specification          |
| 2     | Data Pipeline            | Tạo pipeline train/val/test + bbox/mask             | Dataset/DataLoader hoàn chỉnh  |
| 3     | Classification Baselines | Train ResNet, DenseNet, ViT                         | Baseline classification        |
| 4     | Grad-CAM & Localization  | Đánh giá vùng mô hình tập trung                     | Kết quả RQ1                    |
| 5     | Location-Aware Learning  | Xây loss mới + ablation                             | Kết quả RQ2                    |
| 6     | Final Analysis           | Tổng hợp RQ1/RQ2 + error analysis                   | Kết luận nghiên cứu            |
| 7     | Prototype & Delivery     | Demo hệ thống + report + release code               | Sản phẩm ĐATH hoàn chỉnh       |

---

# Phase 0 — Project Setup

Mục tiêu của phase này là để từ đầu project đã **reproducible**, tránh tới cuối mới gom notebook/code lại.

### Task 0.1 — Khởi tạo GitHub Repository

|                          | Nội dung                                                                                                                                         |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Tên công việc**        | Khởi tạo GitHub repository                                                                                                                       |
| **Mục tiêu công việc**   | Tạo nơi quản lý source code, experiment và tài liệu của toàn bộ ĐATH                                                                             |
| **Điều kiện hoàn thành** | Repo được tạo; có `.gitignore`, `README.md`, `requirements.txt` hoặc `environment.yml`; hai thành viên clone và chạy được project                |
| **Mô tả chi tiết**       | Tạo repository; thiết lập branch `main` và branch phát triển; thống nhất cách commit; không upload dataset/model weight lớn trực tiếp lên GitHub |

### Task 0.2 — Thiết kế cấu trúc source code

|                          | Nội dung                                                                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên công việc**        | Thiết kế project structure                                                                                                                        |
| **Mục tiêu công việc**   | Tách data, model, training, XAI và evaluation thành các module độc lập                                                                            |
| **Điều kiện hoàn thành** | Có cấu trúc thư mục thống nhất và một script test import thành công                                                                               |
| **Mô tả chi tiết**       | Nên tránh toàn bộ project nằm trong Jupyter Notebook. Notebook chủ yếu dùng EDA/visualization; training và evaluation nên chạy bằng Python script |

Tui đề xuất:

```text
cxr-location-aware-xai/
│
├── README.md
├── requirements.txt
├── configs/
│   ├── resnet50.yaml
│   ├── densenet121.yaml
│   └── vit.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── annotations.py
│   │
│   ├── models/
│   │   ├── resnet.py
│   │   ├── densenet.py
│   │   └── vit.py
│   │
│   ├── losses/
│   │   └── location_aware_loss.py
│   │
│   ├── xai/
│   │   ├── gradcam.py
│   │   └── vit_gradcam.py
│   │
│   ├── metrics/
│   │   ├── classification.py
│   │   └── localization.py
│   │
│   └── utils/
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── generate_gradcam.py
│   └── evaluate_localization.py
│
├── experiments/
│
├── app/
│   └── streamlit_app.py
│
└── reports/
```

### Task 0.3 — Chuẩn hóa môi trường

|                          | Nội dung                                                                                                                    |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| **Tên công việc**        | Thiết lập development environment                                                                                           |
| **Mục tiêu công việc**   | Hai thành viên chạy cùng phiên bản thư viện và CUDA/PyTorch                                                                 |
| **Điều kiện hoàn thành** | Clone repo → install dependencies → chạy smoke test thành công                                                              |
| **Mô tả chi tiết**       | Ghi rõ Python, PyTorch, torchvision/timm, OpenCV, pydicom, scikit-learn, matplotlib và thư viện Grad-CAM; lưu seed mặc định |

---

# Phase 1 — Data Understanding & EDA

Đây là phase **cần làm cẩn thận nhất trước khi train model**, vì proposal chưa cố định các finding cuối cùng mà yêu cầu chọn sau khi khảo sát phân bố dữ liệu. 

### Task 1.1 — Khảo sát cấu trúc VinDr-CXR

|                          | Nội dung                                                                                                               |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **Tên công việc**        | Phân tích cấu trúc VinDr-CXR                                                                                           |
| **Mục tiêu công việc**   | Hiểu ảnh, annotation và mối quan hệ giữa image-level label và bounding box                                             |
| **Điều kiện hoàn thành** | Có tài liệu mô tả schema dataset và script đọc annotation                                                              |
| **Mô tả chi tiết**       | Kiểm tra DICOM, `image_id`, `class_name`, `class_id`, tọa độ bounding box, số bác sĩ annotation, local/global findings |

### Task 1.2 — Thống kê phân bố lớp

|                          | Nội dung                                                                                                                           |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Tên công việc**        | EDA phân bố bệnh lý                                                                                                                |
| **Mục tiêu công việc**   | Xác định finding nào đủ dữ liệu cho classification và localization                                                                 |
| **Điều kiện hoàn thành** | Có bảng số ảnh positive/negative, số bounding box và tỷ lệ lớp                                                                     |
| **Mô tả chi tiết**       | Với từng local finding, tính số ảnh positive, số bbox, số bbox trung bình/ảnh, tỷ lệ positive/negative; trực quan hóa distribution |

### Task 1.3 — Chọn các target findings

|                          | Nội dung                                                                                                                                                                            |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên công việc**        | Chốt danh sách bệnh lý mục tiêu                                                                                                                                                     |
| **Mục tiêu công việc**   | Chọn tập finding đủ dữ liệu và có bbox phục vụ đồng thời classification/localization                                                                                                |
| **Điều kiện hoàn thành** | Có danh sách chính thức khoảng 3–5 finding và lý do chọn                                                                                                                            |
| **Mô tả chi tiết**       | Ưu tiên local findings có đủ mẫu dương, annotation ổn định và liên quan rõ vùng ngực/phổi; không chọn Pneumonia làm lớp localization chính vì proposal xác định đây là global label |

### Task 1.4 — Xây dựng annotation rule

|                          | Nội dung                                                                                                             |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| **Tên công việc**        | Quy định cách tổng hợp annotation                                                                                    |
| **Mục tiêu công việc**   | Xử lý trường hợp nhiều bác sĩ annotation cùng ảnh                                                                    |
| **Điều kiện hoàn thành** | Rule được ghi thành tài liệu và implement thành code                                                                 |
| **Mô tả chi tiết**       | Chốt majority vote/image-level rule; quy tắc hợp nhất bounding box; không được thay đổi rule sau khi xem test result |

### Task 1.5 — Dataset sanity check

|                          | Nội dung                                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------------------- |
| **Tên công việc**        | Kiểm tra trực quan annotation                                                                           |
| **Mục tiêu công việc**   | Bảo đảm tọa độ bounding box và label được xử lý đúng                                                    |
| **Điều kiện hoàn thành** | Random ít nhất 50–100 ảnh được visualize và không phát hiện lỗi hệ thống                                |
| **Mô tả chi tiết**       | Vẽ bbox lên ảnh CXR, kiểm tra resize, coordinate transform, multiple boxes và các trường hợp bất thường |

### Milestone Phase 1

Phải có một file kiểu:

```text
dataset_specification.md
```

trả lời được:

```text
Dataset nào?
Finding nào?
Bao nhiêu mẫu?
Label được tạo thế nào?
BBox xử lý thế nào?
Train/val/test thế nào?
```

**Chưa hoàn thành milestone này thì chưa nên train model.**

---

# Phase 2 — Data Pipeline

### Task 2.1 — DICOM preprocessing

|                          | Nội dung                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------- |
| **Tên công việc**        | Xây pipeline xử lý ảnh DICOM                                                                      |
| **Mục tiêu công việc**   | Chuyển CXR thành input chuẩn cho model                                                            |
| **Điều kiện hoàn thành** | Một ảnh từ DICOM → tensor đúng kích thước và intensity                                            |
| **Mô tả chi tiết**       | Đọc DICOM → normalize → resize 224×224 → chuyển grayscale thành input phù hợp pretrained backbone |

### Task 2.2 — Bounding-box transformation

|                          | Nội dung                                                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| **Tên công việc**        | Đồng bộ bounding box với ảnh                                                                                        |
| **Mục tiêu công việc**   | BBox giữ đúng vị trí sau resize/augmentation                                                                        |
| **Điều kiện hoàn thành** | Visual inspection chứng minh bbox sau transform vẫn đúng                                                            |
| **Mô tả chi tiết**       | Scale `x_min`, `y_min`, `x_max`, `y_max`; nếu augmentation có geometric transform thì bbox phải transform đồng thời |

### Task 2.3 — Bounding box → location mask

|                          | Nội dung                                                                                                        |
| ------------------------ | --------------------------------------------------------------------------------------------------------------- |
| **Tên công việc**        | Tạo location mask                                                                                               |
| **Mục tiêu công việc**   | Chuẩn bị ground truth cho localization loss                                                                     |
| **Điều kiện hoàn thành** | Mỗi image/class positive có binary mask tương ứng                                                               |
| **Mô tả chi tiết**       | Nếu có nhiều bbox cùng class, hợp nhất thành mask; resize mask theo resolution của activation map khi tính loss |

### Task 2.4 — Train/Validation/Test split

|                          | Nội dung                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------ |
| **Tên công việc**        | Xây dựng experimental split                                                                |
| **Mục tiêu công việc**   | Bảo đảm validation dùng tuning và test chỉ dùng cuối cùng                                  |
| **Điều kiện hoàn thành** | Có file split cố định lưu `image_id`; tất cả experiment sử dụng cùng split                 |
| **Mô tả chi tiết**       | Split 15.000 training images thành train/validation; official test giữ riêng; seed cố định |

### Task 2.5 — Dataset/DataLoader

|                          | Nội dung                                                                     |
| ------------------------ | ---------------------------------------------------------------------------- |
| **Tên công việc**        | Hoàn thiện PyTorch Dataset/DataLoader                                        |
| **Mục tiêu công việc**   | Một pipeline phục vụ được cả baseline và location-aware training             |
| **Điều kiện hoàn thành** | Batch trả được `image`, `label`, `bbox/mask`, `image_id`                     |
| **Mô tả chi tiết**       | Multi-label output vector, class mask, augmentation và metadata phải đồng bộ |

---

# Phase 3 — Classification Baselines

Phase này tương ứng nhóm **B1–B3** trong proposal. 

### Task 3.1 — Implement classification framework

|                          | Nội dung                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------ |
| **Tên công việc**        | Xây dựng training framework chung                                                          |
| **Mục tiêu công việc**   | Các backbone được train theo cùng protocol                                                 |
| **Điều kiện hoàn thành** | Có `train.py` chạy model/config khác nhau mà không sửa code                                |
| **Mô tả chi tiết**       | Weighted BCE, optimizer, scheduler, checkpoint, early stopping, validation metrics và seed |

### Task 3.2 — ResNet-50 baseline

|                          | Nội dung                                                                 |
| ------------------------ | ------------------------------------------------------------------------ |
| **Tên công việc**        | Train ResNet-50 baseline                                                 |
| **Mục tiêu công việc**   | Tạo CNN baseline đầu tiên                                                |
| **Điều kiện hoàn thành** | Có checkpoint tốt nhất + test metrics + log                              |
| **Mô tả chi tiết**       | Pretrained ImageNet, classifier head multi-label, `L_cls = Weighted BCE` |

### Task 3.3 — DenseNet-121 baseline

Tương tự ResNet nhưng mục tiêu là tạo baseline CNN thứ hai để so sánh kiến trúc.

**Điều kiện hoàn thành:** checkpoint + metrics + config + log đầy đủ.

### Task 3.4 — ViT baseline

|                          | Nội dung                                                           |
| ------------------------ | ------------------------------------------------------------------ |
| **Tên công việc**        | Train Vision Transformer baseline                                  |
| **Mục tiêu công việc**   | Tạo đối chứng CNN vs Transformer                                   |
| **Điều kiện hoàn thành** | ViT chạy cùng dataset split/protocol với CNN và có metrics         |
| **Mô tả chi tiết**       | ViT-B/16 hoặc model phù hợp tài nguyên; sigmoid output multi-label |

### Task 3.5 — Classification evaluation

|                          | Nội dung                                                        |
| ------------------------ | --------------------------------------------------------------- |
| **Tên công việc**        | Đánh giá classification                                         |
| **Mục tiêu công việc**   | So sánh khách quan B1/B2/B3                                     |
| **Điều kiện hoàn thành** | Có bảng AUROC, AUPRC, Precision, Recall, F1 theo class và macro |
| **Mô tả chi tiết**       | Threshold phải chọn từ validation set rồi cố định cho test      |

### Milestone Phase 3

Có bảng dạng:

| Model       | AUROC | AUPRC | Precision | Recall | F1 |
| ----------- | ----: | ----: | --------: | -----: | -: |
| ResNet50    |       |       |           |        |    |
| DenseNet121 |       |       |           |        |    |
| ViT         |       |       |           |        |    |

Đây là **Classification Baseline v1.0**.

---

# Phase 4 — Grad-CAM & Localization Evaluation

Đây là phase tạo dữ liệu để trả lời:

> **RQ1: Does better classification imply better localization?** 

### Task 4.1 — Grad-CAM cho CNN

|                          | Nội dung                                                                                  |
| ------------------------ | ----------------------------------------------------------------------------------------- |
| **Tên công việc**        | Implement Grad-CAM                                                                        |
| **Mục tiêu công việc**   | Sinh class-specific heatmap cho ResNet/DenseNet                                           |
| **Điều kiện hoàn thành** | Một ảnh + target class → heatmap normalized [0,1]                                         |
| **Mô tả chi tiết**       | Hook feature map block convolution cuối; tính gradient; weighted activation; ReLU; resize |

### Task 4.2 — Grad-CAM cho ViT

|                          | Nội dung                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------ |
| **Tên công việc**        | Implement ViT Grad-CAM                                                               |
| **Mục tiêu công việc**   | Cho phép đánh giá localization của Transformer                                       |
| **Điều kiện hoàn thành** | Patch token reshape chính xác thành spatial map và tạo heatmap ổn định               |
| **Mô tả chi tiết**       | Loại CLS token; reshape patch tokens; ghi rõ target layer để bảo đảm reproducibility |

### Task 4.3 — Pointing Game

|                          | Nội dung                                                       |
| ------------------------ | -------------------------------------------------------------- |
| **Tên công việc**        | Implement Pointing Game / Hit Rate                             |
| **Mục tiêu công việc**   | Đánh giá điểm activation mạnh nhất có nằm trong bbox hay không |
| **Điều kiện hoàn thành** | Metric được unit-test bằng các trường hợp giả lập              |
| **Mô tả chi tiết**       | Tìm `argmax(heatmap)` → kiểm tra thuộc bất kỳ GT bbox nào      |

### Task 4.4 — IoU evaluation

|                          | Nội dung                                                 |
| ------------------------ | -------------------------------------------------------- |
| **Tên công việc**        | Implement Grad-CAM IoU                                   |
| **Mục tiêu công việc**   | Định lượng mức độ overlap heatmap-bbox                   |
| **Điều kiện hoàn thành** | Threshold được chọn trên validation và cố định trên test |
| **Mô tả chi tiết**       | Heatmap → binary mask → IoU với union ground-truth bbox  |

### Task 4.5 — Energy Inside Box

|                          | Nội dung                                         |
| ------------------------ | ------------------------------------------------ |
| **Tên công việc**        | Implement Energy Inside Box                      |
| **Mục tiêu công việc**   | Đánh giá localization không phụ thuộc threshold  |
| **Điều kiện hoàn thành** | Có kết quả EIB cho từng architecture × finding   |
| **Mô tả chi tiết**       | Tính tỷ lệ tổng activation nằm bên trong GT mask |

### Task 4.6 — RQ1 analysis

|                          | Nội dung                                                                                        |
| ------------------------ | ----------------------------------------------------------------------------------------------- |
| **Tên công việc**        | Phân tích Classification vs Localization                                                        |
| **Mục tiêu công việc**   | Trả lời RQ1                                                                                     |
| **Điều kiện hoàn thành** | Có bảng ranking và phân tích correlation                                                        |
| **Mô tả chi tiết**       | So thứ hạng AUROC/F1 với IoU/Hit Rate/EIB; bổ sung Spearman rank correlation nếu đủ observation |

Đầu ra quan trọng:

```text
architecture × finding
        ↓
classification performance
        VS
localization performance
```

Ví dụ:

| Finding | Model    | AUROC | F1 | IoU | Hit Rate | EIB |
| ------- | -------- | ----: | -: | --: | -------: | --: |
| A       | ResNet   |       |    |     |          |     |
| A       | DenseNet |       |    |     |          |     |
| A       | ViT      |       |    |     |          |     |

Nếu:

```text
DenseNet AUROC > ResNet
nhưng
DenseNet IoU < ResNet
```

thì đó chính là một bằng chứng cho RQ1.

---

# Phase 5 — Location-Aware Learning

Đây là **phần nghiên cứu quan trọng nhất** và trực tiếp hiện thực hóa góp ý thứ hai của thầy.

### Task 5.1 — Spatial activation extraction

|                          | Nội dung                                                                                                               |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **Tên công việc**        | Tạo differentiable spatial activation                                                                                  |
| **Mục tiêu công việc**   | Có activation map dùng trực tiếp trong training                                                                        |
| **Điều kiện hoàn thành** | Activation map giữ computational graph và backprop được                                                                |
| **Mô tả chi tiết**       | Không dùng Grad-CAM trực tiếp trong loss; lấy spatial activation từ feature map/patch feature như proposal đã xác định |

### Task 5.2 — Implement Location-Aware Loss

|                          | Nội dung                                                                          |
| ------------------------ | --------------------------------------------------------------------------------- |
| **Tên công việc**        | Xây dựng `L_loc`                                                                  |
| **Mục tiêu công việc**   | Penalize activation nằm ngoài bbox                                                |
| **Điều kiện hoàn thành** | Unit test chứng minh loss thấp khi activation nằm trong bbox và cao khi nằm ngoài |
| **Mô tả chi tiết**       | Implement công thức inside-box energy trong proposal                              |

Cốt lõi:

$$
L_{loc}
=
1-
\frac{\sum(A_c\odot G_c)}
{\sum A_c+\epsilon}
$$

và:

$$
L_{total}=L_{cls}+\lambda L_{loc}
$$

### Task 5.3 — Lambda search

|                          | Nội dung                                                     |
| ------------------------ | ------------------------------------------------------------ |
| **Tên công việc**        | Tuning λ                                                     |
| **Mục tiêu công việc**   | Xác định trade-off classification/localization               |
| **Điều kiện hoàn thành** | Có bảng validation cho nhiều λ và chọn λ chính thức          |
| **Mô tả chi tiết**       | Ví dụ thử λ ∈ {0, 0.1, 0.25, 0.5, 1.0}; không tune trên test |

### Task 5.4 — Train Location-Aware ResNet

Nhóm:

```text
L1
ResNet50
L_cls + λL_loc
```

**Điều kiện hoàn thành:** checkpoint + classification metrics + localization metrics.

### Task 5.5 — Train Location-Aware DenseNet

Nhóm:

```text
L2
DenseNet121
L_cls + λL_loc
```

Cùng protocol với baseline.

### Task 5.6 — Location-Aware ViT

Đây nên để **optional / stretch goal**.

Chỉ làm khi spatial representation của ViT đã ổn định và còn đủ thời gian.

### Task 5.7 — Ablation Study

|                          | Nội dung                                                                               |
| ------------------------ | -------------------------------------------------------------------------------------- |
| **Tên công việc**        | Baseline vs Location-Aware ablation                                                    |
| **Mục tiêu công việc**   | Chứng minh tác động thực sự của `L_loc`                                                |
| **Điều kiện hoàn thành** | Có paired comparison cho cùng backbone                                                 |
| **Mô tả chi tiết**       | ResNet `L_cls` vs ResNet `L_cls+λL_loc`; DenseNet tương tự; giữ nguyên các yếu tố khác |

---

# Phase 6 — Final Research Analysis

### Task 6.1 — Trả lời RQ1

**Tên công việc:** Classification–Localization Relationship
**Mục tiêu:** Xác định classification tốt hơn có đồng nghĩa localization tốt hơn không.
**Điều kiện hoàn thành:** Có kết luận dựa trên quantitative results, không dựa vào vài ảnh minh họa.
**Mô tả:** So architecture/class/seed bằng AUROC/F1 versus IoU/Hit Rate/EIB.

---

### Task 6.2 — Trả lời RQ2

**Tên công việc:** Location-Aware Learning Effectiveness
**Mục tiêu:** Kiểm chứng tác động của location-aware loss.
**Điều kiện hoàn thành:** Có bảng before/after và phân tích trade-off.
**Mô tả:** Trả lời:

```text
Localization tăng bao nhiêu?
Classification thay đổi bao nhiêu?
Finding nào hưởng lợi?
Finding nào không?
```

### Task 6.3 — Error analysis

|                          | Nội dung                                                                                           |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| **Tên công việc**        | False Positive / False Negative analysis                                                           |
| **Mục tiêu công việc**   | Hiểu failure mode                                                                                  |
| **Điều kiện hoàn thành** | Có representative case cho TP, FP, FN                                                              |
| **Mô tả chi tiết**       | Hiển thị ảnh + GT bbox + probability + Grad-CAM; phân tích artifact, vùng ngoài phổi, bbox lớn/nhỏ |

### Task 6.4 — Multi-seed validation

|                          | Nội dung                                                                                |
| ------------------------ | --------------------------------------------------------------------------------------- |
| **Tên công việc**        | Repeated experiments                                                                    |
| **Mục tiêu công việc**   | Giảm phụ thuộc vào một lần train                                                        |
| **Điều kiện hoàn thành** | Model quan trọng chạy khoảng 3 seeds nếu tài nguyên cho phép                            |
| **Mô tả chi tiết**       | Báo cáo `mean ± std`; tối thiểu ưu tiên baseline tốt nhất và location-aware counterpart |

### Task 6.5 — Chọn final model

Điều kiện không nhất thiết là:

> model có AUROC cao nhất.

Mà nên xem đồng thời:

```text
Classification
+
Localization
+
Stability
+
Computation
```

---

# Phase 7 — Prototype & Delivery

### Task 7.1 — Inference pipeline

|                          | Nội dung                                                                        |
| ------------------------ | ------------------------------------------------------------------------------- |
| **Tên công việc**        | Final inference pipeline                                                        |
| **Mục tiêu công việc**   | Từ ảnh đầu vào → prediction → Grad-CAM                                          |
| **Điều kiện hoàn thành** | Một function/API chạy end-to-end                                                |
| **Mô tả chi tiết**       | Load checkpoint, preprocessing, inference, sigmoid probability, target Grad-CAM |

### Task 7.2 — Streamlit prototype

|                          | Nội dung                                                                       |
| ------------------------ | ------------------------------------------------------------------------------ |
| **Tên công việc**        | Xây dựng prototype                                                             |
| **Mục tiêu công việc**   | Minh họa hệ thống hỗ trợ phân tích CXR                                         |
| **Điều kiện hoàn thành** | Upload ảnh → prediction → chọn finding → Grad-CAM overlay                      |
| **Mô tả chi tiết**       | UI đơn giản; thể hiện probability; heatmap; disclaimer “research support only” |

### Task 7.3 — Hoàn thiện GitHub README

README cuối project nên bao gồm:

```text
Project overview
Research Questions
Dataset
Selected Findings
Methodology
Architecture
Location-Aware Loss
Experiments
Results
Grad-CAM examples
Installation
Training
Evaluation
Demo
Limitations
References
```

### Task 7.4 — Hoàn thiện báo cáo

**Mục tiêu:** chuyển experiment thành luận điểm nghiên cứu, không chỉ kể lại quá trình code.

Nên tổ chức phần kết quả quanh:

```text
RQ1
↓
Does better classification imply better localization?

RQ2
↓
Does location-aware training improve localization?
```

### Task 7.5 — Chuẩn bị defense

|                          | Nội dung                                                                                                                                                                      |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên công việc**        | Chuẩn bị slide & defense                                                                                                                                                      |
| **Mục tiêu công việc**   | Trình bày logic nghiên cứu rõ ràng                                                                                                                                            |
| **Điều kiện hoàn thành** | Slide, demo và bộ câu hỏi phản biện hoàn thiện                                                                                                                                |
| **Mô tả chi tiết**       | Đặc biệt chuẩn bị giải thích tại sao Grad-CAM ≠ segmentation, tại sao bbox dùng được cho weak localization supervision, tại sao classification cao chưa chắc localization tốt |

---

# Dependency quan trọng

Tui sẽ đi project theo dependency này:

```text
Repository
   ↓
Dataset EDA
   ↓
Chọn Findings
   ↓
Annotation Rule
   ↓
Data Pipeline
   ↓
────────────────────────
          ↓
Classification Baselines
          ↓
Grad-CAM
          ↓
Localization Metrics
          ↓
RQ1
────────────────────────
          ↓
Spatial Activation
          ↓
Location-Aware Loss
          ↓
λ Ablation
          ↓
RQ2
────────────────────────
          ↓
Final Analysis
          ↓
Prototype
          ↓
Report + Defense
```

Có một nguyên tắc rất quan trọng:

> **Không nên làm location-aware loss trước khi baseline + Grad-CAM + localization evaluation chạy ổn.**

Nếu không, khi kết quả location-aware có vấn đề, mình sẽ rất khó biết lỗi nằm ở data, Grad-CAM, metric hay loss.

---

# Phân loại task theo độ ưu tiên

Tui sẽ đặt label GitHub Issues như sau:

| Priority               | Ý nghĩa                                         |
| ---------------------- | ----------------------------------------------- |
| **P0 – Critical**      | Không hoàn thành thì project không thể tiếp tục |
| **P1 – Core Research** | Cần để trả lời RQ1/RQ2                          |
| **P2 – Enhancement**   | Làm project mạnh hơn                            |
| **P3 – Optional**      | Có thời gian mới làm                            |

Ví dụ:

```text
[P0] VinDr-CXR Annotation Parser
[P0] Select Target Findings
[P0] Dataset Split

[P1] ResNet50 Baseline
[P1] DenseNet121 Baseline
[P1] Grad-CAM Evaluation
[P1] Location-Aware Loss
[P1] RQ1 Analysis
[P1] RQ2 Analysis

[P2] ViT Baseline
[P2] Multi-seed Experiments
[P2] Streamlit Prototype

[P3] Location-Aware ViT
[P3] CLAHE Ablation
[P3] External Validation
```

Trong đó **Location-Aware ViT, CLAHE, external validation** không nên để trở thành thứ làm chậm phần nghiên cứu cốt lõi. Proposal cũng đã xem một số thành phần này là tùy tài nguyên/thời gian. 

---

# GitHub repo nên đặt tên gì?

Tên repo nên có 3 đặc điểm:

**Chest X-ray + Location/XAI + dễ hiểu.**

Tui thấy các option này ổn:

| Repo name                         | Nhận xét                                        |
| --------------------------------- | ----------------------------------------------- |
| `cxr-location-aware-xai`          | **Tui đề xuất nhất**                            |
| `cxr-localization-xai`            | Ngắn, dễ hiểu                                   |
| `cxr-explainable-diagnosis`       | Thiên về application                            |
| `cxr-classification-localization` | Rất rõ mục tiêu nghiên cứu                      |
| `location-aware-cxr`              | Ngắn, research-like                             |
| `explainable-cxr`                 | Rất gọn nhưng chưa thể hiện location-aware loss |
| `cxr-gradcam-localization`        | Nhấn Grad-CAM                                   |
| `cxr-diagnostic-ai`               | Hơi rộng                                        |
| `cxr-cnn-vit-xai`                 | Mô tả technology nhưng hơi giống tên môn học    |
| `localize-cxr`                    | Ngắn và đẹp nhưng ít mô tả                      |

### Tui chọn:

```text
cxr-location-aware-xai
```

Tên GitHub:

> **CXR Location-Aware XAI**

Description:

> **Classification and location-aware explainable AI for chest X-ray analysis using CNNs, Vision Transformers and Grad-CAM.**

Nếu muốn thiên hướng academic hơn nữa:

```text
classification-vs-localization-cxr
```

Tên này rất hay về mặt **paper/research**, vì nó gắn trực tiếp với RQ1.

Nhưng xét toàn bộ đồ án, `cxr-location-aware-xai` vẫn cân bằng nhất: ngắn, có CXR, có đóng góp location-aware, có XAI, và không phụ thuộc riêng vào ResNet/DenseNet/ViT.

---

## Nếu đưa tất cả vào GitHub Project

Tui đề xuất board:

```text
Backlog
    ↓
Ready
    ↓
In Progress
    ↓
Experiment Running
    ↓
Review
    ↓
Done
```

Mỗi GitHub Issue nên viết đúng format mà bạn yêu cầu:

```text
Task:
Train ResNet-50 Classification Baseline

Objective:
Xây dựng CNN baseline phục vụ so sánh classification
và localization.

Definition of Done:
- Training chạy hoàn chỉnh.
- Best checkpoint được lưu.
- AUROC/AUPRC/F1 được tính.
- Experiment config được lưu.
- Result được ghi vào bảng tổng hợp.

Details:
- ImageNet pretrained ResNet-50.
- Multi-label sigmoid output.
- Weighted BCE.
- Shared train/val/test split.
- Early stopping theo validation AUROC.
- Lưu random seed, learning rate, batch size.
```

Nếu quản lý theo kiểu này thì tới cuối đồ án, **GitHub Issues gần như trở thành nhật ký thực nghiệm**, cực kỳ hữu ích khi viết báo cáo và khi thầy hỏi “cấu hình này khác cấu hình kia ở đâu”.

Tóm lại, đường critical path của ĐATH nên là **EDA → chốt finding → pipeline → baseline → Grad-CAM/localization → RQ1 → location-aware loss → RQ2 → prototype/report**. Prototype đẹp nhưng không nên chiếm thời gian trước khi hai RQ có kết quả, vì giá trị nghiên cứu chính của đề tài hiện nằm ở **classification vs localization và location-aware learning**, không phải giao diện. 

Tui cũng có thể giúp bạn duy trì tiến độ theo các phase này, ví dụ nhắc review backlog ĐATH mỗi tuần.