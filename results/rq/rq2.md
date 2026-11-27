# RQ2 — effect of the location-aware loss (external test)

## Paired comparison (B = λ>0, A = same-architecture, same-seed baseline; image-level bootstrap CI)

Last-layer Grad-CAM equals the CAM that L_loc optimizes, so localization conclusions should also rest on occlusion and box occlusion (`drop_inside`, `drop_outside`). `eib_minus_prior` compares against the image-independent anatomical prior.

Reading notes:

- `drop_*` are raw logit drops; L_loc changes the logit scale, so compare with `drop_*_norm` (divided by the run's own logit std within each finding).
- `drop_outside` hides the whole image outside the box, which is far out of distribution; an increase does not mean worse localization. The main behavioural evidence is `drop_inside(_norm)`.
- The prior is the same for both runs, so Δ`eib_minus_prior` always equals Δ`eib`; the column is only informative in its a/b values (negative = worse than the average location map).

| arch | seed | lambda | method | metric | a | b | diff | ci_low | ci_high | p_two_sided |
|---|---|---|---|---|---|---|---|---|---|---|
| DenseNet-121 | 42 | 1.0 | classification | macro_auroc | 0.8801 | 0.896 | 0.0159 | 0.0103 | 0.0217 | 0.0 |
| DenseNet-121 | 42 | 1.0 | gradcam | pointing_hit | 0.1685 | 0.4044 | 0.236 | 0.2095 | 0.2658 | 0.0 |
| DenseNet-121 | 42 | 1.0 | gradcam | eib | 0.0851 | 0.2481 | 0.163 | 0.156 | 0.1699 | 0.0 |
| DenseNet-121 | 42 | 1.0 | gradcam | eib_over_area | 2.1068 | 7.0663 | 4.9595 | 4.6573 | 5.2745 | 0.0 |
| DenseNet-121 | 42 | 1.0 | gradcam | eib_minus_prior | -0.062 | 0.101 | 0.163 | 0.156 | 0.1699 | 0.0 |
| DenseNet-121 | 42 | 1.0 | gradcam | iou_mean | 0.0912 | 0.1415 | 0.0503 | 0.0438 | 0.0574 | 0.0 |
| DenseNet-121 | 42 | 1.0 | occlusion | pointing_hit | 0.203 | 0.2676 | 0.0647 | 0.04 | 0.0923 | 0.0 |
| DenseNet-121 | 42 | 1.0 | occlusion | eib | 0.103 | 0.1342 | 0.0312 | 0.0247 | 0.038 | 0.0 |
| DenseNet-121 | 42 | 1.0 | occlusion | eib_over_area | 2.7721 | 3.6509 | 0.8788 | 0.6841 | 1.0854 | 0.0 |
| DenseNet-121 | 42 | 1.0 | occlusion | eib_minus_prior | -0.0441 | -0.0129 | 0.0312 | 0.0247 | 0.038 | 0.0 |
| DenseNet-121 | 42 | 1.0 | occlusion | iou_mean | 0.0826 | 0.1117 | 0.0291 | 0.0238 | 0.0352 | 0.0 |
| DenseNet-121 | 42 | 1.0 | occlusion | drop_inside | 0.4037 | 1.5817 | 1.1779 | 1.0532 | 1.3291 | 0.0 |
| DenseNet-121 | 42 | 1.0 | occlusion | drop_outside | 0.0283 | 0.628 | 0.5998 | 0.4436 | 0.7636 | 0.0 |
| DenseNet-121 | 42 | 1.0 | occlusion | drop_inside_norm | 0.1354 | 0.4653 | 0.33 | 0.2982 | 0.3628 | 0.0 |
| DenseNet-121 | 42 | 1.0 | occlusion | drop_outside_norm | 0.0399 | 0.2323 | 0.1924 | 0.1478 | 0.2357 | 0.0 |
| DenseNet-121 | 43 | 1.0 | classification | macro_auroc | 0.8741 | 0.8885 | 0.0144 | 0.0053 | 0.0231 | 0.004 |
| DenseNet-121 | 43 | 1.0 | gradcam | pointing_hit | 0.1746 | 0.3876 | 0.2129 | 0.1836 | 0.2425 | 0.0 |
| DenseNet-121 | 43 | 1.0 | gradcam | eib | 0.0857 | 0.234 | 0.1483 | 0.1417 | 0.1547 | 0.0 |
| DenseNet-121 | 43 | 1.0 | gradcam | eib_over_area | 2.0488 | 6.4282 | 4.3794 | 4.119 | 4.6188 | 0.0 |
| DenseNet-121 | 43 | 1.0 | gradcam | eib_minus_prior | -0.0614 | 0.0869 | 0.1483 | 0.1417 | 0.1547 | 0.0 |
| DenseNet-121 | 43 | 1.0 | gradcam | iou_mean | 0.0866 | 0.139 | 0.0524 | 0.0465 | 0.0584 | 0.0 |
| DenseNet-121 | 43 | 1.0 | occlusion | pointing_hit | 0.2603 | 0.2738 | 0.0135 | -0.0089 | 0.035 | 0.224 |
| DenseNet-121 | 43 | 1.0 | occlusion | eib | 0.1158 | 0.1246 | 0.0088 | 0.0044 | 0.0134 | 0.0 |
| DenseNet-121 | 43 | 1.0 | occlusion | eib_over_area | 3.1735 | 3.5504 | 0.3769 | 0.2153 | 0.5526 | 0.0 |
| DenseNet-121 | 43 | 1.0 | occlusion | eib_minus_prior | -0.0313 | -0.0225 | 0.0088 | 0.0044 | 0.0134 | 0.0 |
| DenseNet-121 | 43 | 1.0 | occlusion | iou_mean | 0.099 | 0.1069 | 0.008 | 0.0032 | 0.0126 | 0.0 |
| DenseNet-121 | 43 | 1.0 | occlusion | drop_inside | 0.6448 | 1.3834 | 0.7386 | 0.6445 | 0.8431 | 0.0 |
| DenseNet-121 | 43 | 1.0 | occlusion | drop_outside | -0.5819 | 1.3154 | 1.8974 | 1.7407 | 2.0464 | 0.0 |
| DenseNet-121 | 43 | 1.0 | occlusion | drop_inside_norm | 0.1751 | 0.4709 | 0.2958 | 0.2664 | 0.3278 | 0.0 |
| DenseNet-121 | 43 | 1.0 | occlusion | drop_outside_norm | -0.163 | 0.4759 | 0.6389 | 0.5943 | 0.6822 | 0.0 |
| ResNet-50 | 42 | 1.0 | classification | macro_auroc | 0.8502 | 0.8606 | 0.0104 | -0.0001 | 0.0204 | 0.056 |
| ResNet-50 | 42 | 1.0 | gradcam | pointing_hit | 0.215 | 0.4126 | 0.1976 | 0.1741 | 0.2238 | 0.0 |
| ResNet-50 | 42 | 1.0 | gradcam | eib | 0.1162 | 0.2713 | 0.1551 | 0.1471 | 0.1638 | 0.0 |
| ResNet-50 | 42 | 1.0 | gradcam | eib_over_area | 2.8095 | 8.052 | 5.2425 | 4.9112 | 5.5831 | 0.0 |
| ResNet-50 | 42 | 1.0 | gradcam | eib_minus_prior | -0.0309 | 0.1243 | 0.1551 | 0.1471 | 0.1638 | 0.0 |
| ResNet-50 | 42 | 1.0 | gradcam | iou_mean | 0.0878 | 0.1273 | 0.0395 | 0.0325 | 0.0461 | 0.0 |
| ResNet-50 | 42 | 1.0 | occlusion | pointing_hit | 0.2189 | 0.3201 | 0.1011 | 0.0757 | 0.1266 | 0.0 |
| ResNet-50 | 42 | 1.0 | occlusion | eib | 0.1071 | 0.1495 | 0.0424 | 0.037 | 0.0488 | 0.0 |
| ResNet-50 | 42 | 1.0 | occlusion | eib_over_area | 2.6147 | 4.0537 | 1.439 | 1.2598 | 1.6294 | 0.0 |
| ResNet-50 | 42 | 1.0 | occlusion | eib_minus_prior | -0.04 | 0.0025 | 0.0424 | 0.037 | 0.0488 | 0.0 |
| ResNet-50 | 42 | 1.0 | occlusion | iou_mean | 0.079 | 0.1083 | 0.0293 | 0.0241 | 0.035 | 0.0 |
| ResNet-50 | 42 | 1.0 | occlusion | drop_inside | 0.6335 | 1.8471 | 1.2136 | 1.0801 | 1.3527 | 0.0 |
| ResNet-50 | 42 | 1.0 | occlusion | drop_outside | -0.1104 | -0.9959 | -0.8855 | -1.1172 | -0.6457 | 0.0 |
| ResNet-50 | 42 | 1.0 | occlusion | drop_inside_norm | 0.2099 | 0.4091 | 0.1992 | 0.1681 | 0.2325 | 0.0 |
| ResNet-50 | 42 | 1.0 | occlusion | drop_outside_norm | -0.0228 | -0.2501 | -0.2274 | -0.2882 | -0.1679 | 0.0 |
| ResNet-50 | 43 | 1.0 | classification | macro_auroc | 0.8751 | 0.8824 | 0.0073 | -0.0006 | 0.0147 | 0.072 |
| ResNet-50 | 43 | 1.0 | gradcam | pointing_hit | 0.1878 | 0.4043 | 0.2165 | 0.1863 | 0.2454 | 0.0 |
| ResNet-50 | 43 | 1.0 | gradcam | eib | 0.1065 | 0.2681 | 0.1616 | 0.1539 | 0.1696 | 0.0 |
| ResNet-50 | 43 | 1.0 | gradcam | eib_over_area | 2.572 | 7.8791 | 5.3071 | 5.0017 | 5.6544 | 0.0 |
| ResNet-50 | 43 | 1.0 | gradcam | eib_minus_prior | -0.0406 | 0.121 | 0.1616 | 0.1539 | 0.1696 | 0.0 |
| ResNet-50 | 43 | 1.0 | gradcam | iou_mean | 0.0794 | 0.123 | 0.0437 | 0.0379 | 0.049 | 0.0 |
| ResNet-50 | 43 | 1.0 | occlusion | pointing_hit | 0.201 | 0.279 | 0.078 | 0.0551 | 0.1012 | 0.0 |
| ResNet-50 | 43 | 1.0 | occlusion | eib | 0.1001 | 0.1287 | 0.0287 | 0.0239 | 0.0338 | 0.0 |
| ResNet-50 | 43 | 1.0 | occlusion | eib_over_area | 2.655 | 3.4005 | 0.7454 | 0.5499 | 0.9296 | 0.0 |
| ResNet-50 | 43 | 1.0 | occlusion | eib_minus_prior | -0.047 | -0.0183 | 0.0287 | 0.0239 | 0.0338 | 0.0 |
| ResNet-50 | 43 | 1.0 | occlusion | iou_mean | 0.0689 | 0.0985 | 0.0296 | 0.0246 | 0.0348 | 0.0 |
| ResNet-50 | 43 | 1.0 | occlusion | drop_inside | 0.7789 | 2.0054 | 1.2264 | 1.1077 | 1.365 | 0.0 |
| ResNet-50 | 43 | 1.0 | occlusion | drop_outside | -1.7376 | -0.2677 | 1.4699 | 1.2376 | 1.7103 | 0.0 |
| ResNet-50 | 43 | 1.0 | occlusion | drop_inside_norm | 0.1705 | 0.3908 | 0.2203 | 0.1953 | 0.2477 | 0.0 |
| ResNet-50 | 43 | 1.0 | occlusion | drop_outside_norm | -0.4067 | -0.0752 | 0.3315 | 0.2842 | 0.3804 | 0.0 |
| ViT-B/16 (GAP) | 42 | 1.0 | classification | macro_auroc | 0.7826 | 0.7999 | 0.0173 | 0.0065 | 0.0285 | 0.006 |
| ViT-B/16 (GAP) | 42 | 1.0 | gradcam | pointing_hit | 0.0766 | 0.2914 | 0.2148 | 0.1916 | 0.2361 | 0.0 |
| ViT-B/16 (GAP) | 42 | 1.0 | gradcam | eib | 0.0387 | 0.1916 | 0.1529 | 0.1446 | 0.1614 | 0.0 |
| ViT-B/16 (GAP) | 42 | 1.0 | gradcam | eib_over_area | 0.8132 | 5.7671 | 4.9539 | 4.6371 | 5.2926 | 0.0 |
| ViT-B/16 (GAP) | 42 | 1.0 | gradcam | eib_minus_prior | -0.1084 | 0.0445 | 0.1529 | 0.1446 | 0.1614 | 0.0 |
| ViT-B/16 (GAP) | 42 | 1.0 | gradcam | iou_mean | 0.0314 | 0.1283 | 0.097 | 0.0917 | 0.1027 | 0.0 |
| ViT-B/16 (GAP) | 42 | 1.0 | occlusion | pointing_hit | 0.1553 | 0.2124 | 0.0571 | 0.0404 | 0.0757 | 0.0 |
| ViT-B/16 (GAP) | 42 | 1.0 | occlusion | eib | 0.0685 | 0.0994 | 0.0309 | 0.0257 | 0.0362 | 0.0 |
| ViT-B/16 (GAP) | 42 | 1.0 | occlusion | eib_over_area | 1.925 | 2.5728 | 0.6478 | 0.4724 | 0.8161 | 0.0 |
| ViT-B/16 (GAP) | 42 | 1.0 | occlusion | eib_minus_prior | -0.0785 | -0.0477 | 0.0309 | 0.0257 | 0.0362 | 0.0 |
| ViT-B/16 (GAP) | 42 | 1.0 | occlusion | iou_mean | 0.0484 | 0.0667 | 0.0183 | 0.014 | 0.023 | 0.0 |
| ViT-B/16 (GAP) | 42 | 1.0 | occlusion | drop_inside | 0.2528 | 0.4814 | 0.2286 | 0.1495 | 0.3061 | 0.0 |
| ViT-B/16 (GAP) | 42 | 1.0 | occlusion | drop_outside | 0.1761 | 0.1706 | -0.0055 | -0.1212 | 0.1142 | 0.884 |
| ViT-B/16 (GAP) | 42 | 1.0 | occlusion | drop_inside_norm | 0.0982 | 0.1597 | 0.0615 | 0.0351 | 0.0864 | 0.0 |
| ViT-B/16 (GAP) | 42 | 1.0 | occlusion | drop_outside_norm | 0.0904 | 0.0521 | -0.0383 | -0.0799 | 0.0018 | 0.062 |
| ViT-B/16 (GAP) | 43 | 1.0 | classification | macro_auroc | 0.8104 | 0.8576 | 0.0472 | 0.0343 | 0.0619 | 0.0 |
| ViT-B/16 (GAP) | 43 | 1.0 | gradcam | pointing_hit | 0.0677 | 0.3421 | 0.2744 | 0.254 | 0.2982 | 0.0 |
| ViT-B/16 (GAP) | 43 | 1.0 | gradcam | eib | 0.0352 | 0.2116 | 0.1764 | 0.1688 | 0.1847 | 0.0 |
| ViT-B/16 (GAP) | 43 | 1.0 | gradcam | eib_over_area | 0.7694 | 6.2727 | 5.5033 | 5.1725 | 5.8377 | 0.0 |
| ViT-B/16 (GAP) | 43 | 1.0 | gradcam | eib_minus_prior | -0.1119 | 0.0645 | 0.1764 | 0.1688 | 0.1847 | 0.0 |
| ViT-B/16 (GAP) | 43 | 1.0 | gradcam | iou_mean | 0.0284 | 0.1608 | 0.1324 | 0.1266 | 0.1384 | 0.0 |
| ViT-B/16 (GAP) | 43 | 1.0 | occlusion | pointing_hit | 0.2033 | 0.2611 | 0.0578 | 0.0303 | 0.0853 | 0.0 |
| ViT-B/16 (GAP) | 43 | 1.0 | occlusion | eib | 0.103 | 0.1317 | 0.0287 | 0.0207 | 0.0358 | 0.0 |
| ViT-B/16 (GAP) | 43 | 1.0 | occlusion | eib_over_area | 2.9618 | 3.722 | 0.7603 | 0.5168 | 0.9955 | 0.0 |
| ViT-B/16 (GAP) | 43 | 1.0 | occlusion | eib_minus_prior | -0.0441 | -0.0154 | 0.0287 | 0.0207 | 0.0358 | 0.0 |
| ViT-B/16 (GAP) | 43 | 1.0 | occlusion | iou_mean | 0.0642 | 0.0839 | 0.0197 | 0.0145 | 0.0247 | 0.0 |
| ViT-B/16 (GAP) | 43 | 1.0 | occlusion | drop_inside | 0.4066 | 0.9262 | 0.5195 | 0.4378 | 0.6012 | 0.0 |
| ViT-B/16 (GAP) | 43 | 1.0 | occlusion | drop_outside | 0.4429 | 2.178 | 1.7351 | 1.5905 | 1.8699 | 0.0 |
| ViT-B/16 (GAP) | 43 | 1.0 | occlusion | drop_inside_norm | 0.1312 | 0.3361 | 0.2049 | 0.1784 | 0.2315 | 0.0 |
| ViT-B/16 (GAP) | 43 | 1.0 | occlusion | drop_outside_norm | 0.1336 | 0.7441 | 0.6105 | 0.5627 | 0.6569 | 0.0 |

## By finding (mean over pairs, Grad-CAM)

| finding | delta_auroc | delta_auprc | delta_pointing_hit_rate | delta_eib | delta_eib_minus_prior |
|---|---|---|---|---|---|
| Cardiomegaly | 0.0252 | 0.0873 | 0.4628 | 0.5751 | 0.5751 |
| Aortic enlargement | 0.0283 | 0.0472 | 0.6674 | 0.5017 | 0.5017 |
| Infiltration | 0.0145 | 0.0031 | 0.2586 | 0.1807 | 0.1807 |
| Consolidation | 0.0184 | 0.0274 | 0.2378 | 0.1572 | 0.1572 |
| ILD | -0.0007 | 0.0121 | 0.2368 | 0.1451 | 0.1451 |
| Pleural effusion | 0.0293 | 0.0598 | 0.2402 | 0.1387 | 0.1387 |
| Pulmonary fibrosis | 0.0285 | 0.0096 | 0.1997 | 0.1156 | 0.1156 |
| Lung Opacity | 0.0312 | 0.0141 | 0.1766 | 0.0911 | 0.0911 |
| Atelectasis | 0.0078 | 0.0405 | 0.1628 | 0.0773 | 0.0773 |
| Nodule/Mass | 0.0469 | 0.053 | 0.143 | 0.0721 | 0.0721 |
| Other lesion | 0.01 | 0.0142 | 0.1259 | 0.0694 | 0.0694 |
| Pleural thickening | 0.0297 | 0.043 | 0.0957 | 0.0394 | 0.0394 |
| Calcification | -0.0289 | 0.0076 | 0.0644 | 0.0362 | 0.0362 |
| Pneumothorax | 0.0223 | 0.0334 | 0.0833 | 0.0342 | 0.0342 |
