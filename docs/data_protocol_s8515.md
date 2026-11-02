# Data protocol `vindr_s8515_exttest_v1`

Protocol này giữ nguyên 2.250 validation IDs của split 70/15/15 cũ, chuyển 2.250 internal-test IDs cũ vào train, và dùng 3.000 ảnh có ground truth trong `data/raw/test.csv` làm external test độc lập.

## Trạng thái máy đã kiểm chứng

Chạy tất cả lệnh từ repository root `D:\Junior\ĐATH\main` bằng PowerShell.

```powershell
.\.venv-1\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Môi trường đã kiểm chứng: Python 3.13, PyTorch `2.7.0+cu126`, CUDA khả dụng.

Corpus local hiện có 15.000 processed PNG, 15.000 NPZ masks và 3.000 external-test JPG. Kiểm tra lại trước khi chạy:

```powershell
@(Get-ChildItem data/processed/consensus_v1_224/images -File).Count
@(Get-ChildItem data/processed/consensus_v1_224/masks -File).Count
@(Get-ChildItem data/raw/test -Filter *.jpg -File).Count

Test-Path data/processed/manifest.csv
Test-Path data/processed/splits/val.csv
Test-Path data/processed/consensus_v1_224/manifest.csv
Test-Path data/raw/test.csv
```

Kết quả phải lần lượt là `15000`, `15000`, `3000` và bốn giá trị `True`.

Không chạy lại `prepare_data.py`, `build_manifest.py` hoặc full `preprocess.py` trên máy này. Ảnh/mask train đã đầy đủ; chỉ cần project split mới.

## Tạo split 85/15 lần đầu

Bootstrap validation IDs đúng một lần từ validation cũ:

```powershell
python scripts/make_splits.py `
  --config configs/default.yaml `
  --bootstrap-validation-from data/processed/splits/val.csv
```

Kết quả bắt buộc:

```text
train: 12750
val: 2250
overlap: 0
covered images: 15000
```

Sau khi `validation_ids.csv` đã tồn tại, tái tạo assignment bằng:

```powershell
python scripts/make_splits.py --config configs/default.yaml --overwrite
```

Chỉ dùng `--overwrite` khi source manifest, validation-ID hash và protocol ID không thay đổi.

## Project processed manifests

```powershell
python scripts/preprocess.py `
  --config configs/default.yaml `
  --splits-only
```

Nếu manifests đã tồn tại và cần tái tạo có chủ đích:

```powershell
python scripts/preprocess.py `
  --config configs/default.yaml `
  --splits-only `
  --overwrite
```

Lệnh này chỉ đọc `data/processed/consensus_v1_224/manifest.csv`; không xử lý lại ảnh hoặc mask.

## Preprocess external test

Lần đầu:

```powershell
python scripts/preprocess_external_test.py --config configs/default.yaml
```

Tái tạo có chủ đích:

```powershell
python scripts/preprocess_external_test.py --config configs/default.yaml --overwrite
```

Output nặng nằm trong `data/processed/consensus_test_v1_224` và bị Git ignore. Manifest/metadata nhẹ nằm trong `data/protocols/vindr_s8515_exttest_v1`.

## Verify và test

```powershell
python scripts/verify_data_protocol.py --config configs/default.yaml
python -m pytest -q

python scripts/smoke_test.py --config configs/resnet50_baseline.yaml
python scripts/smoke_test.py --config configs/densenet121_baseline.yaml
python scripts/smoke_test.py --config configs/vit_baseline.yaml
```

Verifier phải báo `status: success`, 12.750 train, 2.250 validation, 3.000 external test và overlap bằng 0.

## Train baseline mới

ResNet-50:

```powershell
python scripts/train.py `
  --config configs/resnet50_baseline.yaml `
  --checkpoint-dir outputs/checkpoints/resnet50_s8515_exttest_v1 `
  --result-dir outputs/metrics/resnet50_s8515_exttest_v1
```

DenseNet-121:

```powershell
python scripts/train.py `
  --config configs/densenet121_baseline.yaml `
  --checkpoint-dir outputs/checkpoints/densenet121_s8515_exttest_v1 `
  --result-dir outputs/metrics/densenet121_s8515_exttest_v1
```

ViT-B/16:

```powershell
python scripts/train.py `
  --config configs/vit_baseline.yaml `
  --checkpoint-dir outputs/checkpoints/vit_s8515_exttest_v1 `
  --result-dir outputs/metrics/vit_s8515_exttest_v1
```

Checkpoint protocol cũ bị từ chối vì assignment hash và protocol ID đã thay đổi.

## Validation và final test

Validation ResNet:

```powershell
python scripts/evaluate.py `
  --config configs/resnet50_baseline.yaml `
  --checkpoint outputs/checkpoints/resnet50_s8515_exttest_v1/best.pt `
  --output-dir outputs/metrics/resnet50_s8515_exttest_v1/val `
  --split val `
  --threshold 0.5
```

Trước final test, ghi model name, checkpoint SHA-256 và threshold đã khóa vào `final_test_plan.json`. External test không được dùng để thay đổi model hoặc threshold.

```powershell
python scripts/evaluate.py `
  --config configs/resnet50_baseline.yaml `
  --checkpoint outputs/checkpoints/resnet50_s8515_exttest_v1/best.pt `
  --output-dir outputs/metrics/resnet50_s8515_exttest_v1/final_test `
  --split test `
  --threshold 0.5
```

## Tự stage và push

Codex không commit hoặc push. Người dùng tự kiểm tra và stage bằng danh sách tường minh:

```powershell
git status --short --untracked-files=all
git add .gitignore
git add configs src scripts tests docs notebooks/02_experiment.ipynb
git add data/raw/*.csv
git add data/protocols/vindr_s8515_exttest_v1

git diff --cached --stat
git diff --cached --name-only |
  Select-String -Pattern '\.(jpg|jpeg|png|bmp|tif|tiff|dcm|npz|npy|zip|7z|tar|gz|pt|pth|ckpt)$'
```

Lệnh lọc cuối phải không trả kết quả. Chỉ sau khi tự review, người dùng mới chạy `git commit` và `git push`.
