# RQ1 — classification vs localization (test, gradcam)

## By architecture (baselines λ=0, macro over findings)

| arch | n_seeds | auroc_mean | auroc_std | auprc_mean | auprc_std | pointing_hit_rate_mean | pointing_hit_rate_std | eib_mean | eib_std | eib_over_area_mean | eib_over_area_std | eib_minus_prior_mean | eib_minus_prior_std | iou_mean_mean | iou_mean_std | auroc_rank | auprc_rank | pointing_hit_rate_rank | eib_rank | eib_over_area_rank | eib_minus_prior_rank | iou_mean_rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DenseNet-121 | 2 | 0.8771 | 0.0042 | 0.3291 | 0.0095 | 0.1716 | 0.0044 | 0.0854 | 0.0004 | 2.0778 | 0.041 | -0.0617 | 0.0004 | 0.0889 | 0.0032 | 1.0 | 1.0 | 2.0 | 2.0 | 2.0 | 2.0 | 1.0 |
| ResNet-50 | 2 | 0.8626 | 0.0176 | 0.3083 | 0.0278 | 0.2014 | 0.0192 | 0.1114 | 0.0069 | 2.6908 | 0.168 | -0.0357 | 0.0069 | 0.0836 | 0.006 | 2.0 | 2.0 | 1.0 | 1.0 | 1.0 | 1.0 | 2.0 |

Notes:

- The anatomical prior (mean training union mask, image-independent) has EIB ≈ 0.147. `eib_minus_prior` < 0 means the explanation is less concentrated in the box than this fixed location map.
- Grad-CAM pointing uses each model's native grid (CNN 7×7, ViT 14×14), so it is not fully comparable between CNN and ViT; occlusion uses the same 7×7 grid for every model.

## Spearman correlation

`within_finding_spearman`: Spearman across baseline runs within each finding, averaged over findings (bootstrap CI over findings). `pooled_spearman` pools all run × finding pairs and is therefore dominated by finding difficulty.

| cls_metric | loc_metric | within_finding_spearman | within_ci_low | within_ci_high | n_findings | runs_per_finding | pooled_spearman | n_units |
|---|---|---|---|---|---|---|---|---|
| auroc | pointing_hit_rate | -0.072 | -0.268 | 0.158 | 14 | 4 | 0.647 | 56 |
| auroc | eib | -0.357 | -0.571 | -0.143 | 14 | 4 | 0.693 | 56 |
| auroc | eib_over_area | -0.4 | -0.586 | -0.214 | 14 | 4 | 0.555 | 56 |
| auroc | eib_minus_prior | -0.357 | -0.571 | -0.171 | 14 | 4 | 0.258 | 56 |
| auroc | iou_mean | -0.157 | -0.414 | 0.114 | 14 | 4 | 0.728 | 56 |
| auprc | pointing_hit_rate | -0.171 | -0.468 | 0.124 | 14 | 4 | 0.516 | 56 |
| auprc | eib | -0.371 | -0.571 | -0.171 | 14 | 4 | 0.509 | 56 |
| auprc | eib_over_area | -0.357 | -0.557 | -0.129 | 14 | 4 | 0.387 | 56 |
| auprc | eib_minus_prior | -0.371 | -0.571 | -0.157 | 14 | 4 | -0.206 | 56 |
| auprc | iou_mean | -0.243 | -0.543 | 0.072 | 14 | 4 | 0.542 | 56 |

## Intervention with L_loc (λ>0 minus the same-architecture, same-seed baseline)

| arch | seed | lambda | run | baseline_run | delta_auroc | delta_auprc | delta_pointing_hit_rate | delta_eib | delta_eib_over_area | delta_eib_minus_prior | delta_iou_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DenseNet-121 | 42 | 1.0 | d121_l1_s42 | d121_base_s42 | 0.0159 | 0.0484 | 0.236 | 0.163 | 4.9595 | 0.163 | 0.0503 |
| DenseNet-121 | 43 | 1.0 | d121_l1_s43 | d121_base_s43 | 0.0144 | 0.0281 | 0.2129 | 0.1483 | 4.3794 | 0.1483 | 0.0524 |
| ResNet-50 | 42 | 1.0 | r50_l1_s42 | r50_base_s42 | 0.0104 | 0.0316 | 0.1976 | 0.1551 | 5.2425 | 0.1551 | 0.0395 |
| ResNet-50 | 43 | 1.0 | r50_l1_s43 | r50_base_s43 | 0.0073 | 0.0373 | 0.2165 | 0.1616 | 5.3071 | 0.1616 | 0.0437 |
