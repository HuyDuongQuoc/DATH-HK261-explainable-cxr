# CXR Location-Aware XAI

Multi-label classification and explainability on VinDr-CXR chest X-rays. This branch
(`cleanup`) only keeps the code used in the paper:

- **RQ1** — does better classification come with better localization? ResNet-50 and DenseNet-121 baselines.
- **RQ2** — does a location-aware loss `L_cls + λ·L_loc` improve localization, and what happens to classification? ResNet-50, DenseNet-121 and ViT-B/16 with a GAP head (ViT-GAP), λ = 1.

Research plan: [`docs/Proposal.md`](docs/Proposal.md).

## Scope

| Component | Where |
| --- | --- |
| 14-finding labels from three readers, 224×224 preprocessing, soft/union masks | `src/data/`, `scripts/build_manifest.py`, `scripts/preprocess*.py` |
| Fixed train/validation protocol + external test | `data/protocols/vindr_s8515_exttest_v1` |
| ResNet-50, DenseNet-121, ViT-GAP | `src/models/` |
| Weighted BCE (`pos_weight = n_neg / n_pos`), AdamW, AMP, early stopping, resume | `src/losses/classification_loss.py`, `scripts/train.py` |
| Location-aware loss `L_loc` on the class activation map | `src/losses/localization_loss.py` |
| Grad-CAM, occlusion, box occlusion, anatomical prior | `src/xai/` |
| Pointing game, EIB, IoU, image-level bootstrap, paired tests | `src/metrics/` |
| RQ1/RQ2 analysis and reports | `src/analysis/`, `scripts/rq_analysis.py` |
| External-test evaluation on Kaggle | `scripts/kaggle_eval.py`, `scripts/make_eval_bundle.py`, `kaggle/` |

## Data protocol

Protocol `vindr_s8515_exttest_v1`:

- 12,750 training and 2,250 validation images, fixed, from the 15,000 images annotated by three readers.
- 3,000 external test images with consensus labels, disjoint from train/validation, used only for the final evaluation.
- 14 output classes (abnormal findings). `No finding` images are negatives for every class.
- Train/validation labels are the majority vote (2 of 3 readers). Soft masks average the per-reader union of boxes.
- The external test set has one consensus reader and binary masks.
- Split assignments, metadata and checksums are version-controlled in [`data/protocols/vindr_s8515_exttest_v1`](data/protocols/vindr_s8515_exttest_v1). Processed images, masks and checkpoints are not committed.

## Repository layout

```text
configs/        default.yaml + one config per paper run type
data/           annotations, protocol artifacts, local data folders
kaggle/         Kaggle kernel for external-test evaluation
notebooks/      EDA and the Colab runner
scripts/        data preparation, train, evaluate, localization, RQ analysis
src/analysis/   RQ1/RQ2 tables and reports
src/data/       consensus labels, preprocessing, dataset, protocol
src/losses/     weighted BCE and L_loc
src/metrics/    classification and localization metrics, bootstrap statistics
src/models/     model factory, CAM forward pass, ViT-GAP
src/training/   epoch engine
src/xai/        Grad-CAM, occlusion, priors, localization evaluation
tests/          unit tests
```

## Setup

`requirements.txt` pins PyTorch `2.7.0+cu126` and torchvision `0.22.0+cu126` (CUDA 12.6).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On another CUDA version, install a matching PyTorch first, then:

```bash
pip install -r requirements-common.txt
```

## Data preparation

1. Copy `.env.example` to `.env`, fill in the Kaggle credentials, then download
   `sunghyunjun/vinbigdata-1024-jpg-dataset` into `data/raw`:

   ```bash
   python scripts/prepare_data.py
   ```

2. Materialize the committed protocol (no new split is created):

   ```bash
   python scripts/build_manifest.py --config configs/default.yaml
   python scripts/preprocess.py --config configs/default.yaml --overwrite
   python scripts/preprocess_external_test.py --config configs/default.yaml --overwrite
   python scripts/verify_data_protocol.py --config configs/default.yaml
   ```

Only use `scripts/make_splits.py --overwrite` to change the protocol on purpose; it changes
the split checksums.

## Training

The configs reproduce the paper runs (seeds 42 and 43):

| Config | Model | λ | Recipe |
| --- | --- | --- | --- |
| `resnet50_baseline.yaml` / `resnet50_location.yaml` | ResNet-50 | 0 / 1 | lr 1e-4, wd 1e-4, 12 epochs, patience 3 |
| `densenet121_baseline.yaml` / `densenet121_location.yaml` | DenseNet-121 | 0 / 1 | same as ResNet-50 |
| `vit_gap_baseline.yaml` / `vit_gap_location.yaml` | ViT-GAP | 0 / 1 | lr 2e-5, wd 0.05, 30 epochs, patience 5 |

All runs use batch size 16, AdamW, AMP, no augmentation, and select the checkpoint on
validation macro-AUROC. Per-class F1 thresholds are tuned on validation and stored in the checkpoint.

```bash
python scripts/train.py \
  --config configs/resnet50_location.yaml \
  --checkpoint-dir outputs/runs/r50_l1_s42/ckpt \
  --result-dir outputs/runs/r50_l1_s42/res
```

Add `--resume <checkpoint-dir>/last.pt` to resume. Set the seed with `seed:` in the config.

ViT-GAP (`src/models/vit_gap.py`) keeps the ImageNet ViT-B/16 backbone but classifies
from the mean of per-patch logits instead of the CLS token. Its 14×14 CAM therefore
works with `L_loc`, Grad-CAM and the metrics exactly like a CNN.

## Evaluation

Classification on the external test set (uses the validation-tuned thresholds):

```bash
python scripts/evaluate.py \
  --config configs/resnet50_location.yaml \
  --checkpoint outputs/runs/r50_l1_s42/ckpt/best.pt \
  --output-dir outputs/runs/r50_l1_s42/cls_test --split test
```

Localization on the external test set:

```bash
python scripts/evaluate_localization.py \
  --config configs/resnet50_location.yaml \
  --checkpoint outputs/runs/r50_l1_s42/ckpt/best.pt \
  --output-dir outputs/runs/r50_l1_s42/loc_test_occlusion \
  --split test --test-only --iou-threshold 0.30 \
  --method occlusion --box-occlusion
```

- `--method gradcam` uses the last feature map. With GAP + linear this equals the CAM
  that `L_loc` optimizes, so it is **not independent of the loss**; RQ2 conclusions rest on
  `occlusion` and box occlusion.
- `--box-occlusion` records the logit drop when hiding the inside (`drop_inside`) or the
  outside (`drop_outside`) of the box.
- The anatomical prior (mean training union mask per finding) is cached in
  `outputs/priors/train_union_prior.npz`; records include `prior_eib`, `eib_minus_prior`.
- `box_area` is the chance level of EIB/pointing; `iou_mean` averages IoU over thresholds.

To evaluate on Kaggle instead, build a bundle with `scripts/make_eval_bundle.py`, upload it
as a private dataset and run `kaggle/kernel_eval.py` (see the docstrings of both scripts).

## RQ analysis

```bash
python scripts/rq_analysis.py outputs/modal_runs/volume --output-dir outputs/rq --exclude 'vitgap_*'
python scripts/verify_metrics.py outputs/modal_runs/volume
```

- RQ1 uses `--rq1-models` (default `resnet50,densenet121`); RQ2 uses every model in `--models`.
- `vitgap_*` runs were trained with the CNN recipe and did not converge; exclude them.
- Outputs: `rq1_{gradcam,occlusion}.md`, `rq2.md` and the CSV tables behind them.
- `verify_metrics.py` recomputes AUROC/AP from the predictions and checks the localization records.

## Tests

```bash
python -m pytest -q
```

## Reproducibility notes

- Checkpoints are bound to the model name, the order of the 14 findings, the protocol ID
  and the split-assignment hash.
- The external test set is never used for model selection or threshold tuning.
- `requirements-common.txt` does not pin every package; store a package snapshot with results.
