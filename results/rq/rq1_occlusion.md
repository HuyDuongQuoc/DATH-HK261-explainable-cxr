# RQ1 — classification vs localization (test, occlusion)

## By architecture (baselines λ=0, macro over findings)

| arch | n_seeds | auroc_mean | auroc_std | auprc_mean | auprc_std | pointing_hit_rate_mean | pointing_hit_rate_std | eib_mean | eib_std | eib_over_area_mean | eib_over_area_std | eib_minus_prior_mean | eib_minus_prior_std | iou_mean_mean | iou_mean_std | auroc_rank | auprc_rank | pointing_hit_rate_rank | eib_rank | eib_over_area_rank | eib_minus_prior_rank | iou_mean_rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DenseNet-121 | 2 | 0.8771 | 0.0042 | 0.3291 | 0.0095 | 0.2316 | 0.0405 | 0.1094 | 0.0091 | 2.9728 | 0.2839 | -0.0377 | 0.0091 | 0.0908 | 0.0116 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| ResNet-50 | 2 | 0.8626 | 0.0176 | 0.3083 | 0.0278 | 0.2099 | 0.0127 | 0.1036 | 0.005 | 2.6349 | 0.0285 | -0.0435 | 0.005 | 0.074 | 0.0072 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |

Notes:

- The anatomical prior (mean training union mask, image-independent) has EIB ≈ 0.147. `eib_minus_prior` < 0 means the explanation is less concentrated in the box than this fixed location map.

## Spearman correlation

`within_finding_spearman`: Spearman across baseline runs within each finding, averaged over findings (bootstrap CI over findings). `pooled_spearman` pools all run × finding pairs and is therefore dominated by finding difficulty.

| cls_metric | loc_metric | within_finding_spearman | within_ci_low | within_ci_high | n_findings | runs_per_finding | pooled_spearman | n_units |
|---|---|---|---|---|---|---|---|---|
| auroc | pointing_hit_rate | -0.045 | -0.377 | 0.292 | 14 | 4 | 0.623 | 56 |
| auroc | eib | 0.014 | -0.286 | 0.3 | 14 | 4 | 0.671 | 56 |
| auroc | eib_over_area | 0.071 | -0.229 | 0.415 | 14 | 4 | 0.581 | 56 |
| auroc | eib_minus_prior | 0.014 | -0.286 | 0.3 | 14 | 4 | 0.355 | 56 |
| auroc | iou_mean | 0.043 | -0.3 | 0.429 | 14 | 4 | 0.639 | 56 |
| auprc | pointing_hit_rate | -0.008 | -0.305 | 0.321 | 14 | 4 | 0.54 | 56 |
| auprc | eib | -0.043 | -0.314 | 0.243 | 14 | 4 | 0.576 | 56 |
| auprc | eib_over_area | 0.257 | 0.029 | 0.457 | 14 | 4 | 0.538 | 56 |
| auprc | eib_minus_prior | -0.043 | -0.314 | 0.229 | 14 | 4 | -0.059 | 56 |
| auprc | iou_mean | -0.0 | -0.271 | 0.286 | 14 | 4 | 0.528 | 56 |

## Intervention with L_loc (λ>0 minus the same-architecture, same-seed baseline)

| arch | seed | lambda | run | baseline_run | delta_auroc | delta_auprc | delta_pointing_hit_rate | delta_eib | delta_eib_over_area | delta_eib_minus_prior | delta_iou_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DenseNet-121 | 42 | 1.0 | d121_l1_s42 | d121_base_s42 | 0.0159 | 0.0484 | 0.0647 | 0.0312 | 0.8788 | 0.0312 | 0.0291 |
| DenseNet-121 | 43 | 1.0 | d121_l1_s43 | d121_base_s43 | 0.0144 | 0.0281 | 0.0135 | 0.0088 | 0.3769 | 0.0088 | 0.008 |
| ResNet-50 | 42 | 1.0 | r50_l1_s42 | r50_base_s42 | 0.0104 | 0.0316 | 0.1011 | 0.0424 | 1.439 | 0.0424 | 0.0293 |
| ResNet-50 | 43 | 1.0 | r50_l1_s43 | r50_base_s43 | 0.0073 | 0.0373 | 0.078 | 0.0287 | 0.7454 | 0.0287 | 0.0296 |
