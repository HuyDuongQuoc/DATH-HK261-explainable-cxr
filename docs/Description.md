Mình đánh giá theo `codebase-reader`: dựa trên code, output notebook và “điều kiện hoàn thành” trong kế hoạch, không chỉ dựa vào việc folder/file đã tồn tại.

## 1. Dự án đang ở giai đoạn nào?

Nếu chọn một mốc duy nhất theo [Plan.md](/D:/Junior/ĐATH/main/docs/Plan.md:107), dự án đang ở:

> **Đầu Phase 2 — Data & EDA**, đồng thời còn thiếu một số nền móng quan trọng của Phase 0 và Phase 1.

Nói dễ hiểu hơn: dự án đã qua bước “chỉ có ý tưởng”, đã tải và khảo sát dữ liệu thật, đã dựng thử workflow local–GitHub–Colab–Drive; nhưng chưa có pipeline ML để huấn luyện model.

| Phase | Trạng thái | Bằng chứng |
|---|---|---|
| Phase 0 — Kiến trúc project | Đang làm dở | Có repo, README, requirements và folder skeleton; nhưng config, model, dataset, training, evaluation hầu hết còn rỗng |
| Phase 1 — Local/GitHub/Colab/Drive | Có prototype | Colab Runner đã mount Drive, clone/pull repo và gọi smoke test; chưa có output notebook chứng minh full round-trip đã chạy hoàn chỉnh |
| Phase 2 — Data & EDA | Đang thực hiện | Đã tải dữ liệu thật, đọc CSV, thống kê lớp/bác sĩ và vẽ bbox |
| Phase 3 — Train/evaluate framework | Chưa bắt đầu | Không có `train.py`, model factory, trainer, checkpoint manager hay evaluation engine |
| Phase 4–8 | Chưa bắt đầu | Chưa có baseline, Grad-CAM, localization metrics, location-aware loss, kết quả hay prototype |

Không có phase nào hoàn tất toàn bộ Definition of Done.

### Những phần đã làm được

- Repo Git đã được thiết lập, có `.gitignore`, dependencies và cấu trúc thư mục ban đầu.
- Đã có script tải bản VinBigData/VinDr-CXR resize 1024px từ Kaggle tại [prepare_data.py](/D:/Junior/ĐATH/main/scripts/prepare_data.py:11).
- Dữ liệu local hiện có:
  - 15.000 ảnh train.
  - 3.000 ảnh test.
  - 67.914 dòng annotation.
  - 15 class, gồm cả `No finding`.
- [01_eda.ipynb](/D:/Junior/ĐATH/main/notebooks/01_eda.ipynb) đã chạy trên dữ liệu thật:
  - Đọc `train.csv`.
  - Vẽ phân bố class.
  - Thống kê annotation theo bác sĩ.
  - Hiển thị ảnh và bounding box.
- [00_colab_runner.ipynb](/D:/Junior/ĐATH/main/notebooks/00_colab_runner.ipynb:56) đã mô tả workflow mount Drive → clone/pull GitHub → chạy smoke test → kiểm tra checkpoint → tải metrics.
- [smoke_test.py](/D:/Junior/ĐATH/main/scripts/smoke_test.py:7) đã kiểm tra được khả năng nhận command-line argument, tạo folder, ghi checkpoint giả và JSON kết quả.

### Những gì khiến Phase 0–2 chưa hoàn thành

- [default.yaml](/D:/Junior/ĐATH/main/configs/default.yaml) đang rỗng; chưa có config riêng cho ResNet, DenseNet và ViT.
- [dataset.py](/D:/Junior/ĐATH/main/src/data/dataset.py) và [resnet.py](/D:/Junior/ĐATH/main/src/models/resnet.py) đang rỗng.
- Chưa có `losses/`, `metrics/`, model factory, preprocessing hay annotation parser.
- Chưa có `train.py`, `evaluate.py`, `generate_gradcam.py`, `evaluate_localization.py`, `export_results.py`.
- Smoke test hiện chỉ kiểm tra ghi file; AUROC `0.85` và F1 `0.80` là số hard-code, không phải kết quả mô hình.
- Chưa chọn chính thức 3–5 target findings.
- Chưa có annotation rule để xử lý nhiều bác sĩ.
- Chưa có `train.csv`, `val.csv`, `test.csv` split cố định.
- Chưa có `Dataset`/`DataLoader` trả về image, label và mask.
- Chưa có test tự động; thư mục `tests/` và `app/` đang rỗng.

## 2. Mô tả dự án hiện tại

Đây là một dự án nghiên cứu Explainable AI trên ảnh X-quang ngực. Mục tiêu cuối cùng là xây dựng mô hình phân loại đa nhãn các bất thường phổi/ngực, đồng thời đánh giá xem mô hình có thực sự tập trung vào đúng vùng tổn thương được bác sĩ đánh dấu bằng bounding box hay không.

Dự án dự kiến so sánh ba backbone:

- ResNet-50.
- DenseNet-121.
- Vision Transformer.

Có hai câu hỏi nghiên cứu chính:

1. Mô hình có khả năng phân loại tốt hơn có đồng nghĩa với khả năng định vị tổn thương tốt hơn không?
2. Việc thêm location-aware loss vào quá trình huấn luyện có giúp activation tập trung đúng bounding box hơn mà không làm giảm đáng kể chất lượng phân loại không?

Pipeline nghiên cứu dự kiến sử dụng:

- AUROC, AUPRC, Precision, Recall và F1 để đánh giá classification.
- Grad-CAM để sinh heatmap giải thích.
- Pointing Game, IoU và Energy Inside Box để đánh giá localization.
- Loss tổng `L_cls + λL_loc` để thử nghiệm location-aware learning.

Ở thời điểm hiện tại, codebase mới hiện thực hóa phần chuẩn bị môi trường, tải dữ liệu và EDA cơ bản. Chưa có model có thể train, chưa có checkpoint thật và chưa có kết quả nghiên cứu.

## 3. Cấu trúc codebase

```text
main/
├── configs/
│   └── default.yaml            # Placeholder, đang rỗng
├── data/
│   ├── raw/                    # Dataset Kaggle đã tải
│   ├── interim/                # Chưa sử dụng
│   └── processed/              # Chưa sử dụng
├── docs/
│   ├── Plan.md                 # Roadmap Phase 0–8
│   ├── Proposal.md             # Milestone và câu hỏi nghiên cứu
│   └── Experiment.md           # Yêu cầu GPU/checkpoint/tqdm
├── notebooks/
│   ├── 00_colab_runner.ipynb   # Workflow GitHub–Colab–Drive
│   └── 01_eda.ipynb            # EDA đang phát triển
├── scripts/
│   ├── prepare_data.py         # Download dataset từ Kaggle
│   └── smoke_test.py           # Kiểm tra ghi checkpoint/result giả
├── src/
│   ├── data/                   # Chưa có Dataset/DataLoader
│   ├── models/                 # Chưa có model implementation
│   ├── training/               # Chưa có trainer
│   ├── evaluation/             # Chưa có metrics/evaluator
│   ├── xai/                    # Chưa có Grad-CAM
│   └── utils/                  # Chưa có utility
├── outputs/                    # Test checkpoint/result local
├── app/                        # Rỗng
└── tests/                      # Rỗng
```

Các folder `src/` hiện chủ yếu đóng vai trò khung kiến trúc, chưa phải implementation thực tế.

## 4. Tech stack dự kiến

- Ngôn ngữ: Python.
- Deep learning: PyTorch, torchvision, timm.
- Metrics: torchmetrics, scikit-learn.
- Xử lý ảnh: OpenCV, Pillow, Albumentations.
- Ảnh y khoa: pydicom.
- XAI: `grad-cam`.
- Data/EDA: pandas, NumPy, SciPy, Matplotlib, Seaborn.
- Config: YAML, dotenv.
- Experiment tracking: TensorBoard, tqdm.
- Testing: pytest.
- Prototype tương lai: Streamlit.
- Compute: Google Colab GPU.
- Persistent storage: Google Drive.
- Dataset acquisition: Kaggle API.

Lưu ý: dependency hiện chưa được pin version và `python-dotenv` bị khai báo hai lần trong [requirements.txt](/D:/Junior/ĐATH/main/requirements.txt).

## 5. Luồng hoạt động

Luồng thực sự đã có:

```text
Kaggle
  ↓
scripts/prepare_data.py
  ↓
data/raw/train.csv + ảnh JPG
  ↓
notebooks/01_eda.ipynb
  ↓
Phân bố class + thống kê bác sĩ + ảnh có bbox
```

Luồng Colab đã được dựng ở mức smoke test:

```text
Local → GitHub → Colab
                    ↓
             Mount Google Drive
                    ↓
          scripts/smoke_test.py
                    ↓
       Checkpoint text trên Drive
       + metrics JSON trong Colab
```

Luồng mục tiêu nhưng chưa được triển khai:

```text
Dataset/DataLoader
        ↓
Model Factory
        ↓
Training Engine
        ↓
Checkpoint trên Drive
        ↓
Classification Evaluation
        ↓
Grad-CAM
        ↓
Localization Evaluation
        ↓
RQ1 / RQ2 Analysis
        ↓
Streamlit Prototype
```

## 6. Cách chạy phần hiện có

Theo [README.md](/D:/Junior/ĐATH/main/README.md:4), cần tạo môi trường và cài dependencies. Trên PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Tạo `.env` từ `.env.example`, điền Kaggle username và API key, sau đó tải dữ liệu:

```powershell
python scripts/prepare_data.py
```

Mở EDA:

```powershell
jupyter lab notebooks/01_eda.ipynb
```

Smoke test có thể gọi như sau dựa trên CLI hiện tại:

```powershell
python scripts/smoke_test.py `
  --checkpoint-dir outputs/test_checkpoint `
  --result-dir outputs/test_results
```

Hiện chưa có lệnh chạy web, train hoặc evaluate vì các entrypoint tương ứng chưa tồn tại.

## 7. Điểm rủi ro và chưa nhất quán

- Nhánh local đang chậm hơn `origin/main` hai commit.
- [Plan.md](/D:/Junior/ĐATH/main/docs/Plan.md) chưa được Git track.
- Notebook EDA có thay đổi local chưa commit.
- Trong notebook, biến được định nghĩa là `TRAIN_DIR` nhưng cell cuối gọi `TRAIN_DIRECTORY` tại [01_eda.ipynb](/D:/Junior/ĐATH/main/notebooks/01_eda.ipynb:400). Output ảnh đang lưu có thể là output cũ từ trước khi cell bị sửa.
- Kế hoạch nói đến pipeline DICOM, nhưng [prepare_data.py](/D:/Junior/ĐATH/main/scripts/prepare_data.py:11) đang tải bản JPG 1024px đã resize. Nếu dùng JPG thì cần cập nhật protocol; nếu cần DICOM thì data pipeline hiện tại chưa đáp ứng.
- `train.csv` có cột index không tên, khi đọc bằng PowerShell trở thành header tự sinh; trong pandas nó xuất hiện dưới dạng `Unnamed: 0`.
- Có hai virtual environment `.venv` và `.venv-1`, dễ gây khác biệt dependency.
- Colab notebook chưa lưu execution output, vì vậy code đã tồn tại nhưng chưa có bằng chứng trong repo rằng workflow Drive chạy end-to-end.
- README còn rất sơ lược và chưa mô tả research questions, dataset schema, selected findings hay kiến trúc dự kiến.

## 8. File nên đọc tiếp

Theo thứ tự giá trị:

1. [Plan.md](/D:/Junior/ĐATH/main/docs/Plan.md:1) — roadmap chính Phase 0–8.
2. [01_eda.ipynb](/D:/Junior/ĐATH/main/notebooks/01_eda.ipynb) — công việc đang được phát triển.
3. [00_colab_runner.ipynb](/D:/Junior/ĐATH/main/notebooks/00_colab_runner.ipynb) — cách vận hành Colab/Drive.
4. [prepare_data.py](/D:/Junior/ĐATH/main/scripts/prepare_data.py) — nguồn và vị trí dataset.
5. [smoke_test.py](/D:/Junior/ĐATH/main/scripts/smoke_test.py) — workflow giả lập hiện có.
6. [Proposal.md](/D:/Junior/ĐATH/main/docs/Proposal.md) — RQ1, RQ2 và tiêu chí thực nghiệm.

Bước hợp lý tiếp theo là hoàn tất các phần còn thiếu của Phase 0–1, đồng thời kết thúc milestone EDA: sửa notebook, chốt target findings, viết annotation rule và cố định dataset split. Chưa nên bắt đầu ResNet baseline trước khi bốn đầu việc này hoàn thành.
