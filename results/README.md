# results/

Results of the paper runs on the VinDr-CXR external test set (3,000 images).

| Path | Content |
| --- | --- |
| [`rq/rq1_occlusion.md`](rq/rq1_occlusion.md), [`rq/rq1_gradcam.md`](rq/rq1_gradcam.md) | RQ1 reports: ResNet-50 and DenseNet-121 baselines (λ=0), architecture table, within-finding Spearman, L_loc intervention |
| [`rq/rq2.md`](rq/rq2.md) | RQ2 report: paired comparison λ=1 vs baseline for ResNet-50, DenseNet-121 and ViT-GAP (classification, Grad-CAM, occlusion, box occlusion) with image-level bootstrap CIs |
| `rq/*.csv` | Tables behind the reports (`rq1_*_long.csv` is the run x finding table) |
| [`verify_metrics.txt`](verify_metrics.txt) | Independent re-check of the saved metrics (`scripts/verify_metrics.py`) |
| `runs/<run>/` | Per-run summaries: training config and history, best validation metrics, test classification metrics, localization summaries and per-finding tables |
| [`full_lambda_grid.csv`](full_lambda_grid.csv) | Earlier λ grid on ResNet-50 (validation used to choose λ = 1; one seed) |

## Runs

| Run | Model | λ | Seeds |
| --- | --- | --- | --- |
| `r50_base_s*`, `r50_l1_s*` | ResNet-50 | 0, 1 | 42, 43 |
| `d121_base_s*`, `d121_l1_s*` | DenseNet-121 | 0, 1 | 42, 43 |
| `vit_gap_ft_base_s*`, `vit_gap_ft_l1_s*` | ViT-B/16 GAP head | 0, 1 | 42, 43 |

The configs that produced them are in `configs/` (see the top-level README). The reports
are regenerated with:

```bash
python scripts/rq_analysis.py <runs-folder> --output-dir results/rq --exclude 'vitgap_*'
```

Per-pair localization records and prediction files (~3 MB per run) and checkpoints are
not committed; `rq_analysis.py` needs them to recompute the paired tests.

## Reading notes

- Last-layer Grad-CAM equals the CAM that L_loc optimizes, so RQ2 localization claims rest
  on occlusion and box occlusion (`drop_inside_norm`).
- `drop_outside` hides everything outside the box, which is far out of distribution; an
  increase does not mean worse localization.
- The image-independent anatomical prior has EIB ≈ 0.147, above every baseline model.
- ViT-GAP baselines have many empty Grad-CAM maps (all-negative CAM → EIB 0), which lowers
  their Grad-CAM EIB below chance; occlusion is not affected.
- With two CNNs and two seeds, RQ1 uses 4 runs per finding, so the within-finding test is weak.
