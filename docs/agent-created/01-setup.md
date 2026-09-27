Được. Với cách bạn muốn vận hành, tui sẽ **đổi hẳn roadmap theo hướng “local development → GitHub → Colab execution → Drive checkpoint → local result”**. Cách này hợp lý hơn việc chia notebook riêng cho từng model, vì proposal cần ResNet-50, DenseNet-121 và ViT chạy trên **cùng một experimental protocol** để so sánh công bằng. 

Điểm quan trọng nhất là: **chỉ có một source code chung**. Khi chuyển ResNet → DenseNet → ViT, mình **không sửa training code**, chỉ thay file config hoặc tham số model.

## 1. Workflow kỹ thuật mới

Tui đề xuất coi 4 nơi có vai trò khác nhau:

| Nơi              | Vai trò                                                                               |
| ---------------- | ------------------------------------------------------------------------------------- |
| **Máy cá nhân**  | Viết code Python, Git, kiểm tra nhanh, lưu kết quả performance cuối cùng, làm báo cáo |
| **GitHub**       | Source code chính thức, configs, scripts, docs; không chứa dataset/checkpoint lớn     |
| **Google Colab** | GPU runtime; clone repo và gọi các script Python                                      |
| **Google Drive** | Dataset/archive dùng lại và checkpoint của từng experiment                            |

Luồng sẽ là:

```text
Máy cá nhân
Viết/chỉnh Python
      ↓
git push
      ↓
GitHub Repository
      ↓
Colab: git clone / git pull
      ↓
Mount Google Drive
      ↓
Chuẩn bị data trên /content/
      ↓
python scripts/train.py --config ...
      ↓
Checkpoint → Google Drive
      ↓
Evaluate model
      ↓
CSV / JSON / PNG / Grad-CAM
      ↓
Download về máy
```

Và workflow model là tuần tự:

```text
ResNet-50
   ↓
Train → checkpoint → evaluate → tải result
   ↓
DenseNet-121
   ↓
Train → checkpoint → evaluate → tải result
   ↓
ViT
   ↓
Train → checkpoint → evaluate → tải result
```

Không cần giữ 3 model đồng thời trong RAM/GPU.

---

# PHASE 0 — Chuẩn hóa kiến trúc project

Đây là phase nên làm đầu tiên, vì nếu source được thiết kế đúng thì toàn bộ Colab về sau rất nhẹ.

| Tên công việc                          | Mục tiêu công việc                      | Điều kiện hoàn thành                      | Mô tả chi tiết                                                                                         |
| -------------------------------------- | --------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **0.1 Khởi tạo GitHub repository**     | Tạo source chính thức của ĐATH          | Clone được repo trên máy và Colab         | Tạo repo `cxr-location-aware-xai`; README, `.gitignore`, requirements; không commit dataset/checkpoint |
| **0.2 Thiết kế source structure**      | Chia code thành module tái sử dụng      | `python` import được toàn bộ module       | Tách `data`, `models`, `losses`, `xai`, `metrics`, `scripts`, `configs`                                |
| **0.3 Xây cơ chế config**              | Chuyển model mà không sửa source        | Có config riêng cho ResNet, DenseNet, ViT | YAML chứa model name, image size, batch size, LR, epochs, loss, seed, output path                      |
| **0.4 Chuẩn hóa command-line scripts** | Cho Colab gọi code bằng command         | Các script chạy được từ terminal          | Tối thiểu có `train.py`, `evaluate.py`, `generate_gradcam.py`, `evaluate_localization.py`              |
| **0.5 Local smoke test**               | Kiểm tra source trên máy trước khi push | Chạy được vài ảnh/fake batch bằng CPU     | Không cần train thật; mục tiêu là bắt syntax error, import error, dimension mismatch                   |

### Cấu trúc repo nên chuyển thành

```text
cxr-location-aware-xai/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── resnet50_baseline.yaml
│   ├── densenet121_baseline.yaml
│   ├── vit_baseline.yaml
│   ├── resnet50_location.yaml
│   └── densenet121_location.yaml
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   ├── annotations.py
│   │   └── transforms.py
│   │
│   ├── models/
│   │   ├── factory.py
│   │   ├── resnet.py
│   │   ├── densenet.py
│   │   └── vit.py
│   │
│   ├── losses/
│   │   ├── classification_loss.py
│   │   └── location_aware_loss.py
│   │
│   ├── metrics/
│   │   ├── classification.py
│   │   └── localization.py
│   │
│   ├── xai/
│   │   ├── gradcam.py
│   │   └── vit_gradcam.py
│   │
│   └── utils/
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate_gradcam.py
│   ├── evaluate_localization.py
│   └── export_results.py
│
├── notebooks/
│   ├── 00_colab_runner.ipynb
│   └── 01_eda.ipynb
│
├── docs/
├── tests/
└── app/
```

Điểm khác với roadmap cũ là **notebook không chứa logic training chính**. Notebook chỉ gọi script.

---

# PHASE 1 — Thiết lập workflow Local ↔ GitHub ↔ Colab ↔ Drive

| Tên công việc                  | Mục tiêu công việc                        | Điều kiện hoàn thành                                  | Mô tả chi tiết                                                                              |
| ------------------------------ | ----------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **1.1 Thiết lập Python local** | Có thể phát triển và test source trên máy | `python scripts/...` chạy được                        | Cài môi trường Python phục vụ coding và smoke test; không bắt buộc máy phải đủ GPU để train |
| **1.2 Thiết lập Git workflow** | Đồng bộ source máy ↔ GitHub               | Push/pull thành công                                  | Máy là nơi chỉnh code chính; sau mỗi thay đổi ổn định thì commit + push                     |
| **1.3 Tạo Colab Runner**       | Biến Colab thành môi trường execution     | Notebook clone repo và chạy được script               | Chỉ cần một `00_colab_runner.ipynb`, không tạo notebook ResNet/DenseNet/ViT riêng           |
| **1.4 Kết nối Google Drive**   | Lưu tài nguyên persistent                 | Colab ghi checkpoint vào Drive thành công             | Mount `/content/drive`; tạo thư mục cố định cho data/checkpoints                            |
| **1.5 Test full round-trip**   | Kiểm tra toàn bộ workflow                 | Local push → Colab pull → script chạy → Drive có file | Chạy một experiment giả/rất ngắn trước khi dùng GPU train thật                              |

Drive tui đề xuất:

```text
MyDrive/
└── DATH_CXR/
    ├── datasets/
    │   └── vindr_cxr/
    │
    ├── checkpoints/
    │   ├── resnet50_baseline/
    │   ├── densenet121_baseline/
    │   ├── vit_baseline/
    │   ├── resnet50_location/
    │   └── densenet121_location/
    │
    └── run_logs/
```

---

# PHASE 2 — Data & EDA

Phase này vẫn phải hoàn thành trước khi train, vì proposal quy định số finding cuối cùng chỉ được quyết định sau EDA; đồng thời cùng một local finding phải phục vụ được classification và localization. 

| Tên công việc                  | Mục tiêu công việc                | Điều kiện hoàn thành                    | Mô tả chi tiết                                       |
| ------------------------------ | --------------------------------- | --------------------------------------- | ---------------------------------------------------- |
| **2.1 Đọc VinDr-CXR**          | Hiểu cấu trúc DICOM và annotation | Script đọc được image + metadata + bbox | Xử lý DICOM, CSV annotation, class name, tọa độ      |
| **2.2 EDA class distribution** | Biết mức mất cân bằng và số bbox  | Có CSV + biểu đồ thống kê               | Tính positive images, bbox, negative, tỷ lệ class    |
| **2.3 Chọn target findings**   | Chốt lớp nghiên cứu               | Có danh sách chính thức 3–5 finding     | Ưu tiên local finding có bbox đủ nhiều               |
| **2.4 Annotation rule**        | Chốt cách xử lý nhiều bác sĩ      | Rule được lưu trong docs/code           | Xác định cách tổng hợp image-level label và bbox     |
| **2.5 Dataset split**          | Cố định protocol thực nghiệm      | Có `train.csv`, `val.csv`, `test.csv`   | Split chỉ tạo một lần và dùng cho tất cả model       |
| **2.6 Data pipeline**          | Tạo input chung cho mọi model     | DataLoader trả đúng image/label/mask    | Resize, normalize, augmentation, bbox transformation |
| **2.7 Data sanity check**      | Phát hiện lỗi annotation/pipeline | Bbox trên ảnh hiển thị chính xác        | Visualize một tập mẫu trước khi train                |

### Dataset trên Colab

Tui không khuyến khích train trực tiếp bằng hàng nghìn file DICOM đọc liên tục từ Drive vì Drive I/O có thể chậm.

Nên:

```text
Drive
dataset/archive
   ↓
Colab session bắt đầu
   ↓
copy/extract → /content/data/
   ↓
training đọc /content/data/
```

Checkpoint thì ngược lại:

```text
training
   ↓
best checkpoint
   ↓
Drive
```

Như vậy Colab reset không mất model, nhưng training vẫn đọc dữ liệu từ ổ local nhanh hơn.

---

# PHASE 3 — Xây framework train/evaluate chung

Đây là phase quan trọng hơn việc “code ResNet” riêng. Nếu framework tốt thì ba model chỉ khác config.

| Tên công việc              | Mục tiêu công việc               | Điều kiện hoàn thành                                          | Mô tả chi tiết                                  |
| -------------------------- | -------------------------------- | ------------------------------------------------------------- | ----------------------------------------------- |
| **3.1 Model Factory**      | Tạo model theo tên config        | `build_model("resnet50")`, `"densenet121"`, `"vit"` hoạt động | Các architecture dùng cùng interface            |
| **3.2 Training Engine**    | Dùng một trainer cho mọi model   | Train được ít nhất ResNet/DenseNet bằng cùng script           | Forward, loss, backward, optimizer, validation  |
| **3.3 Checkpoint Manager** | Không mất model khi Colab ngắt   | `best.pt` và `last.pt` nằm trên Drive                         | Lưu epoch, model state, optimizer state, config |
| **3.4 Resume Training**    | Tiếp tục experiment bị gián đoạn | Có thể resume từ `last.pt`                                    | Cực kỳ cần thiết với Colab                      |
| **3.5 Evaluation Engine**  | Chuẩn hóa đánh giá               | Một script xuất cùng format cho mọi model                     | AUROC, AUPRC, Precision, Recall, F1             |
| **3.6 Result Exporter**    | Tạo output dễ tải về máy         | Tạo được ZIP/CSV/JSON/PNG                                     | Performance cuối không nằm lẫn trong notebook   |

Ví dụ training script **không được viết riêng như**:

```text
train_resnet.py
train_densenet.py
train_vit.py
```

Mà chỉ nên có:

```bash
python scripts/train.py --config configs/resnet50_baseline.yaml
```

Sau đó đổi model:

```bash
python scripts/train.py --config configs/densenet121_baseline.yaml
```

và:

```bash
python scripts/train.py --config configs/vit_baseline.yaml
```

Đây mới là setup phù hợp để chứng minh các model được chạy theo cùng protocol.

---

# PHASE 4 — Chạy baseline tuần tự

Proposal yêu cầu so sánh ResNet-50, DenseNet-121 và ViT trên cùng protocol. 

Không nên coi đây là ba project nhỏ. Nó là **ba experiment dùng chung infrastructure**.

| Tên công việc                      | Mục tiêu công việc             | Điều kiện hoàn thành                       | Mô tả chi tiết                                 |
| ---------------------------------- | ------------------------------ | ------------------------------------------ | ---------------------------------------------- |
| **4.1 ResNet-50 Baseline**         | Tạo baseline CNN đầu tiên      | Checkpoint Drive + performance local       | Train → evaluate → export → tải kết quả về máy |
| **4.2 Validate ResNet experiment** | Xác nhận pipeline đáng tin cậy | Metrics và prediction không có lỗi rõ ràng | Chỉ khi ResNet chạy ổn mới chuyển architecture |
| **4.3 DenseNet-121 Baseline**      | Baseline CNN thứ hai           | Checkpoint Drive + performance local       | Đổi config, không sửa trainer                  |
| **4.4 ViT Baseline**               | Baseline Transformer           | Checkpoint Drive + performance local       | Đổi model config; giữ split/protocol           |
| **4.5 Baseline comparison**        | Tổng hợp B1/B2/B3              | Có bảng performance chung                  | AUROC/AUPRC/F1 theo từng finding và macro      |

Thứ tự tui khuyên:

```text
ResNet-50
     ↓
kiểm tra toàn pipeline
     ↓
DenseNet-121
     ↓
ViT
```

ResNet là model nên chạy đầu tiên vì nó đóng vai trò **debug baseline**.

---

# PHASE 5 — Grad-CAM & Localization

Sau khi cả baseline classification chạy xong mới bắt đầu đánh giá “model nhìn vào đâu”. Đây là phần phục vụ trực tiếp RQ1: **Does better classification imply better localization?** 

| Tên công việc             | Mục tiêu công việc            | Điều kiện hoàn thành                  | Mô tả chi tiết                                      |
| ------------------------- | ----------------------------- | ------------------------------------- | --------------------------------------------------- |
| **5.1 CNN Grad-CAM**      | Sinh heatmap ResNet/DenseNet  | Heatmap đúng class và đúng kích thước | Load checkpoint trực tiếp từ Drive                  |
| **5.2 ViT Grad-CAM**      | Sinh localization map cho ViT | Patch tokens reshape đúng             | Xác định target layer, loại CLS token               |
| **5.3 Pointing Game**     | Đánh giá hit trong bbox       | Có Hit Rate/model/class               | Không cần threshold                                 |
| **5.4 IoU**               | Đánh giá overlap              | Có IoU/model/class                    | Threshold chỉ tune trên validation                  |
| **5.5 Energy Inside Box** | Đánh giá activation energy    | Có EIB/model/class                    | Không phụ thuộc threshold                           |
| **5.6 RQ1 Result Export** | Trả lời RQ1                   | Có bảng classification + localization | So ranking và correlation giữa hai loại performance |

Ở phase này **không cần train lại model**. Chỉ load từng checkpoint:

```text
Drive/resnet50_baseline/best.pt
        ↓
Grad-CAM + localization evaluation
        ↓
result_resnet50_localization.csv
        ↓
download máy

Drive/densenet121_baseline/best.pt
        ↓
...

Drive/vit_baseline/best.pt
        ↓
...
```

---

# PHASE 6 — Location-Aware Loss

Proposal hiện tại tách rõ baseline `L_cls` và mô hình `L_cls + λL_loc`, đây là thí nghiệm dùng để trả lời RQ2. 

| Tên công việc                   | Mục tiêu công việc                     | Điều kiện hoàn thành                              | Mô tả chi tiết                                                       |
| ------------------------------- | -------------------------------------- | ------------------------------------------------- | -------------------------------------------------------------------- |
| **6.1 Spatial activation map**  | Tạo activation khả vi                  | Backprop qua activation thành công                | Không đưa Grad-CAM trực tiếp vào training loss                       |
| **6.2 Implement `L_loc`**       | Khuyến khích activation nằm trong bbox | Unit test loss hoạt động đúng                     | Implement inside-box energy loss                                     |
| **6.3 Implement `L_total`**     | Kết hợp classification/localization    | `L_cls + λL_loc` train ổn định                    | Log riêng từng thành phần loss                                       |
| **6.4 Lambda ablation**         | Chọn λ hợp lý                          | Có validation result của nhiều λ                  | Chỉ chọn trên validation                                             |
| **6.5 ResNet Location-Aware**   | Đo hiệu quả `L_loc`                    | Checkpoint + classification + localization result | Chạy đầu tiên vì đã có ResNet baseline                               |
| **6.6 DenseNet Location-Aware** | Kiểm chứng trên CNN thứ hai            | Có đầy đủ paired result                           | Chỉ đổi architecture                                                 |
| **6.7 ViT Location-Aware**      | Mở rộng sang Transformer               | Optional                                          | Chỉ làm khi thời gian và spatial map cho phép                        |
| **6.8 RQ2 analysis**            | Baseline vs Location-Aware             | Có bảng before/after                              | Xác định localization tăng bao nhiêu và classification đổi bao nhiêu |

Cách chạy vẫn hoàn toàn giống:

```bash
python scripts/train.py \
    --config configs/resnet50_location.yaml
```

xong ResNet thì:

```bash
python scripts/train.py \
    --config configs/densenet121_location.yaml
```

Không đổi source.

---

# PHASE 7 — Tổng hợp result trên máy cá nhân

Đây là điểm tui sẽ đổi mạnh so với roadmap cũ: **máy cá nhân mới là nơi tổng hợp kết quả cuối**, chứ không để Colab trở thành nơi lưu toàn bộ lịch sử nghiên cứu.

| Tên công việc                    | Mục tiêu công việc                      | Điều kiện hoàn thành                | Mô tả chi tiết                               |
| -------------------------------- | --------------------------------------- | ----------------------------------- | -------------------------------------------- |
| **7.1 Chuẩn hóa export package** | Mỗi experiment có output giống nhau     | Có ZIP cho mỗi run                  | Chứa config, metrics, predictions, figures   |
| **7.2 Download performance**     | Lưu kết quả nghiên cứu local            | Máy có folder đầy đủ cho từng model | Không nhất thiết tải checkpoint lớn về       |
| **7.3 Master Results Table**     | Tổng hợp toàn bộ experiment             | Có một CSV/Excel tổng               | architecture × loss × class × seed × metrics |
| **7.4 RQ1 Analysis**             | Kết luận classification vs localization | Có bảng/biểu đồ/kết luận            | AUROC/F1 vs IoU/Pointing Game/EIB            |
| **7.5 RQ2 Analysis**             | Kết luận tác động location-aware loss   | Có paired comparison                | Baseline vs location-aware cùng backbone     |
| **7.6 Error Analysis**           | Phân tích TP/FP/FN                      | Có case study representative        | Ảnh + bbox + confidence + Grad-CAM           |

Máy local có thể tổ chức:

```text
DATH/
├── source/
│   └── cxr-location-aware-xai/
│
├── results/
│   ├── resnet50_baseline/
│   ├── densenet121_baseline/
│   ├── vit_baseline/
│   ├── resnet50_location/
│   └── densenet121_location/
│
├── analysis/
│   ├── master_results.csv
│   ├── rq1/
│   └── rq2/
│
└── report/
```

---

# PHASE 8 — Prototype + Báo cáo

Chỉ đến đây mới ưu tiên UI/demo.

| Tên công việc                 | Mục tiêu công việc         | Điều kiện hoàn thành                       | Mô tả chi tiết                           |
| ----------------------------- | -------------------------- | ------------------------------------------ | ---------------------------------------- |
| **8.1 Chọn final checkpoint** | Chọn model demo            | Có tiêu chí classification + localization  | Không nhất thiết là model AUROC cao nhất |
| **8.2 Inference script**      | Chạy 1 ảnh end-to-end      | Image → prediction → heatmap               | Dùng đúng preprocessing khi training     |
| **8.3 Prototype**             | Minh họa hệ thống          | Upload CXR → result + Grad-CAM             | Có thể dùng Streamlit                    |
| **8.4 Báo cáo RQ1/RQ2**       | Hoàn thiện phần nghiên cứu | Kết luận có số liệu hỗ trợ                 | Bám đúng hai câu hỏi proposal            |
| **8.5 GitHub cleanup**        | Tạo repo có thể bàn giao   | README + configs + instructions hoàn chỉnh | Người khác clone repo hiểu cách chạy     |
| **8.6 Defense package**       | Chuẩn bị bảo vệ            | Slide + demo + bảng kết quả                | Giải thích được pipeline và ablation     |

---

## 2. Một Colab notebook duy nhất là đủ

Tui sẽ thiết kế Colab theo dạng **runner**, ví dụ session ResNet:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone <github-repo>
%cd cxr-location-aware-xai
!pip install -r requirements.txt
```

Chuẩn bị dataset:

```bash
!python scripts/prepare_data.py \
    --source /content/drive/MyDrive/DATH_CXR/datasets/vindr_cxr \
    --output /content/data/vindr_cxr
```

Train:

```bash
!python scripts/train.py \
    --config configs/resnet50_baseline.yaml \
    --checkpoint-dir /content/drive/MyDrive/DATH_CXR/checkpoints/resnet50_baseline
```

Evaluate:

```bash
!python scripts/evaluate.py \
    --config configs/resnet50_baseline.yaml \
    --checkpoint /content/drive/MyDrive/DATH_CXR/checkpoints/resnet50_baseline/best.pt \
    --output-dir /content/results/resnet50_baseline
```

Export:

```bash
!python scripts/export_results.py \
    --input /content/results/resnet50_baseline \
    --output /content/resnet50_baseline_results.zip
```

Tải ZIP về máy.

Lần sau mở Colab, thay vì clone lại nếu folder còn tồn tại thì có thể `git pull`; nếu runtime đã reset thì clone mới. Sau đó chỉ thay:

```text
resnet50_baseline.yaml
        ↓
densenet121_baseline.yaml
        ↓
vit_baseline.yaml
```

---

## 3. “Đơn vị công việc” mới nên là một Experiment

Tui nghĩ đây là thay đổi quan trọng nhất trong cách quản lý task của bạn.

Thay vì:

```text
Code ResNet
Code DenseNet
Code ViT
```

nên coi mỗi run là:

```text
Experiment
│
├── config.yaml
├── checkpoint
├── training history
├── classification metrics
├── predictions
├── localization metrics
└── figures
```

Ví dụ định danh:

```text
B1_resnet50_cls_seed42
B2_densenet121_cls_seed42
B3_vit_cls_seed42

L1_resnet50_loc_lambda025_seed42
L2_densenet121_loc_lambda025_seed42
```

Như vậy sau này khi thầy hỏi:

> “Kết quả này chạy với lambda bao nhiêu?”

hoặc:

> “DenseNet này dùng learning rate nào?”

mình truy lại được ngay, thay vì cố nhớ từ Colab cũ.

---

## 4. Thứ tự triển khai thực tế tui đề xuất

Đường đi mới của project sẽ là:

```text
PHASE 0
Repository + Source Architecture
            ↓
PHASE 1
Local ↔ GitHub ↔ Colab ↔ Drive
            ↓
PHASE 2
EDA + Findings + Data Pipeline
            ↓
PHASE 3
Generic Training/Evaluation Framework
            ↓
PHASE 4
ResNet50 baseline
      ↓
DenseNet121 baseline
      ↓
ViT baseline
            ↓
PHASE 5
Grad-CAM + IoU + Pointing Game + EIB
            ↓
RQ1
            ↓
PHASE 6
Location-aware loss
      ↓
ResNet50 location-aware
      ↓
DenseNet121 location-aware
            ↓
RQ2
            ↓
PHASE 7
Download + Final Analysis trên PC
            ↓
PHASE 8
Prototype + Report + Defense
```

Cách này cũng phù hợp với logic proposal: classification baseline → localization evaluation → location-aware loss → phân tích RQ1/RQ2 → prototype. 

**Repo `cxr-location-aware-xai` vẫn rất phù hợp**, tui chưa thấy cần đổi. Điểm cần thay là kiến trúc vận hành: **GitHub là source of truth cho code, Drive là source of truth cho checkpoint, máy cá nhân là source of truth cho kết quả nghiên cứu/báo cáo, còn Colab chỉ là compute environment.**

Bước kế tiếp nên làm luôn **Phase 0 + Phase 1**, cụ thể là chốt cấu trúc repo và thiết kế chính xác `config.yaml`, `train.py`, `evaluate.py`, `00_colab_runner.ipynb`. Đây sẽ là bộ khung mà tất cả task phía sau dựa vào.