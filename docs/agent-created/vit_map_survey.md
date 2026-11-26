# Khảo sát: chuyển output của ViT thành heatmap (saliency map)

Mục tiêu: chọn và biện minh cách tạo map cho ViT trong RQ1/RQ2, và có số liệu tham chiếu
từ tài liệu. Số liệu chỉ được ghi khi đã đọc được từ nguồn; mức độ kiểm chứng ghi ở cột
"Nguồn".

## 1. Setup hiện tại của nhóm

| Model | Map | Ghi chú |
|---|---|---|
| ViT-B/16 (CLS) | Grad-CAM tại `ln_1` của block cuối, trên 196 patch token (bỏ CLS), lưới 14×14 | Head chỉ đọc CLS nên gradient ở output block cuối của patch token = 0; phải lấy ở trong block (qua attention). Gần với cách Chefer et al. 2021 mô tả cho Grad-CAM baseline. |
| ViT-B/16 (GAP) | CAM trực tiếp: head áp lên từng patch token, logit = mean | Tương đương CAM của CNN → dùng được L_loc. Grad-CAM lớp cuối ≡ CAM nên đánh giá bằng nó bị vòng lặp. |
| Mọi model | Occlusion (patch 32, stride 32, lưới 7×7) + box-occlusion | Không phụ thuộc kiến trúc → thước đo chính cho so sánh chéo kiến trúc. |

## 2. Các họ phương pháp

| Phương pháp | Cơ chế | Theo lớp? | Ưu | Nhược |
|---|---|---|---|---|
| Raw attention (CLS→patch, block cuối) | Lấy hàng CLS của ma trận attention | Không | Rẻ, không cần gradient | Chỉ 1 layer, bỏ qua residual/MLP; không phân biệt lớp |
| Attention Rollout (Abnar & Zuidema 2020) | Nhân dồn (A + I) qua các layer | Không | Gộp thông tin mọi layer | Không theo lớp → vô dụng cho multi-label 14 finding |
| Gradient Attention Rollout | Rollout với attention có trọng số gradient | Có | Theo lớp, cài đặt phổ biến (vit-explain) | Heuristic, kết quả khác nhau theo cách gộp head |
| Grad-CAM trên token | Gradient logit theo activation token, reshape lưới | Có | Giống pipeline CNN | Phải chọn vị trí hook; Chefer 2021 xếp hạng thấp |
| Transformer Attribution / TransLRP (Chefer, Gur, Wolf, CVPR 2021) | LRP qua attention + gradient, bỏ phần âm | Có | Tốt nhất trên ImageNet-seg và perturbation | Cần cài LRP cho từng layer (phức tạp) |
| Generic attention explainability (Chefer et al., ICCV 2021) | Ā = mean_heads((∇A ⊙ A)⁺), cập nhật R ← R + Ā·R | Có | Gần bằng TransLRP, không cần LRP, dễ cài | Vẫn dựa vào attention |
| ViT-CX (Xie et al., IJCAI 2023) | Mask từ patch embedding, chấm điểm theo hiệu ứng nhân quả lên output | Có | Faithful hơn attention-based theo tác giả | Tốn nhiều forward (giống occlusion/RISE) |
| Occlusion / RISE | Perturbation, model-agnostic | Có | Công bằng giữa CNN và ViT | Chậm, map thô |
| Thiết kế sẵn (ViT-GAP/CAM, class token riêng — CLARiTy 2025) | Map là một phần của kiến trúc | Có | Map "trung thực" theo cấu trúc, tối ưu được bằng loss | Đổi kiến trúc so với ViT chuẩn |

Vấn đề liên quan: **artifact token** (Darcet et al., *Vision Transformers Need Registers*,
ICLR 2024). ViT lớn, train đủ lâu (cả supervised như DeiT-III) sinh vài token chuẩn rất cao
ở vùng nền ít thông tin, gây đốm sai trên attention map; khắc phục bằng register token
(hoặc sửa activation không cần train, arXiv 2506.08010). Cần kiểm tra trên ViT-B/16 của
nhóm trước khi kết luận về chất lượng map của ViT.

## 3. Số liệu tham chiếu

### 3.1 Chefer et al. — ViT-base, ImageNet

Perturbation AUC (%). Negative: cao hơn tốt; Positive: thấp hơn tốt.
Nguồn: bảng 2 bản ICCV 2021 (đọc nguyên văn); trùng với bảng 1 bản CVPR 2021.

| Test | Rollout | Raw attn | Grad-CAM | Partial LRP | TransLRP (CVPR'21) | Generic (ICCV'21) |
|---|---|---|---|---|---|---|
| Negative, predicted ↑ | 53.10 | 45.55 | 41.52 | 50.49 | 54.16 | **54.61** |
| Negative, target ↑ | – | – | 42.02 | 50.49 | 55.04 | **55.67** |
| Positive, predicted ↓ | 20.05 | 23.99 | 34.06 | 19.64 | **17.03** | 17.32 |
| Positive, target ↓ | – | – | 33.56 | 19.64 | **16.04** | 16.72 |

ImageNet-Segmentation (%). Nguồn: bản CVPR 2021, bảng 2 — đọc qua bản tóm tắt trích xuất,
**chưa đối chiếu nguyên văn**, cần kiểm lại PDF trước khi trích dẫn.

| Metric | Rollout | Raw attn | Grad-CAM | LRP | Partial LRP | TransLRP |
|---|---|---|---|---|---|---|
| Pixel acc. | 73.54 | 67.84 | 64.44 | 51.09 | 76.31 | **79.70** |
| mAP | 84.76 | 80.24 | 71.60 | 55.68 | 84.67 | **86.03** |
| mIoU | 55.42 | 46.37 | 40.82 | 32.89 | 57.94 | **61.95** |

Grad-CAM cho ViT trong bài: dùng attention map của layer cuối (phần token ứng với CLS) làm
"feature map", vì gradient ở output cuối chỉ khác 0 tại CLS. Trên ImageNet, Grad-CAM xếp
gần cuối ở cả hai benchmark.

### 3.2 Komorowski, Baniecki, Biecek — CVPRW (XAI4CV) 2023, chest X-ray

ViT-base/16 trên COVID-QU-Ex (33.920 ảnh, 3 lớp, accuracy 0.954). 300 ảnh test, metric
Quantus, **không đánh giá theo box**. Nguồn: arXiv HTML, nguyên văn.

| Phương pháp | Faithfulness ↑ | Sensitivity ↓ | Complexity ↓ |
|---|---|---|---|
| LIME | 0.07 ± 0.13 | 0.21 ± 0.06 | 0.09 ± 0.04 |
| Attention (gộp head = max) | 0.10 ± 0.08 | 0.07 ± 0.03 | 0.16 ± 0.12 |
| TransLRP (Chefer 2021) | **0.16 ± 0.18** | **0.06 ± 0.03** | 0.21 ± 0.16 |

Không có Grad-CAM. Độ lệch chuẩn lớn so với khoảng cách giữa các phương pháp, không có
kiểm định.

### 3.3 Barekatain & Glocker — arXiv 2510.12021 (2025, preprint)

ViT-B, DeiT-tiny, DINO-S, Swin-tiny. Ảnh tế bào máu và siêu âm vú (**không có CXR**).
So Grad-CAM với Gradient Attention Rollout bằng insertion/deletion AUC, 1 lần chạy.
Ví dụ ViT-B: deletion ↓ Grad-CAM 0.60 vs Rollout 0.42 (PBC); 0.63 vs 0.61 (siêu âm).
Tác giả kết luận Grad-CAM tốt hơn, nhưng số liệu không nhất quán (Rollout thắng ở ViT-B
PBC và Swin). Chỉ nên dùng như ví dụ rằng xếp hạng phụ thuộc dataset/metric.

### 3.4 CLARiTy — arXiv 2512.16700 (ViT-S/16, class token riêng cho mỗi bệnh)

Map = attention class-token → patch, trung bình p block cuối, nhân với mask giải phẫu.
Có kết quả transfer sang **VinDr-CXR test 3.000 ảnh** (6 finding), metric = tỉ lệ ảnh dương
có box dự đoán đạt IoU ≥ T (box rút từ map bằng ngưỡng hysteresis đã fit). Nguồn: arXiv
HTML bảng 4, nguyên văn.

| VinDr-CXR, IoU ≥ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|
| Hossain et al. 2024 (in-domain) | 0.457 | 0.354 | 0.247 | 0.145 | 0.060 |
| CLARiTy 224, NIH-calibrated | 0.683 | 0.514 | 0.407 | 0.323 | 0.189 |
| CLARiTy 512, NIH-calibrated | 0.749 | 0.567 | 0.451 | 0.358 | 0.248 |

Macro AUC trên NIH ChestX-ray14: 0.808 (224) / 0.825 (512). Bài không so với Grad-CAM hay
rollout, và metric là box-accuracy chứ không phải EIB/pointing → **không so trực tiếp được**
với số của nhóm, chỉ cho thấy ViT có map thiết kế sẵn có thể localize tốt trên VinDr.

### 3.5 Saporta et al., Nature Machine Intelligence 2022 (CNN, CheXpert/CheXlocalize)

Benchmark saliency (Grad-CAM, Grad-CAM++, Integrated Gradients) trên CNN so với người;
khuyến nghị báo cả mIoU và pointing (hit rate) vì hai metric có thể mâu thuẫn. Không đọc
được bản đầy đủ (medRxiv chặn, Nature yêu cầu đăng nhập) nên **không ghi số**; cần tải PDF
thủ công nếu muốn trích số liệu.

## 4. Đối chiếu với kết quả của nhóm

- Tài liệu (Chefer 2021) xếp Grad-CAM gần cuối trên ViT. Trong kết quả của nhóm, ViT cũng
  xếp cuối với Grad-CAM, **nhưng** xếp cuối cả với occlusion (EIB 0.091/0.086 so với CNN
  0.104–0.109), nên kết luận "ViT localize kém hơn" không chỉ do Grad-CAM.
- Không có bài nào trong khảo sát báo cáo EIB/pointing theo box cho ViT trên VinDr-CXR với
  Grad-CAM/Chefer → một bảng so sánh như vậy của nhóm là đóng góp mới (dù nhỏ).
- Các bài về ViT y tế đều nhấn mạnh xếp hạng phụ thuộc metric (Komorowski; Barekatain), phù
  hợp với lựa chọn của nhóm: báo nhiều metric và dùng occlusion làm thước đo chính.

## 5. Đề xuất

1. Thêm **Generic attention explainability (Chefer ICCV 2021)** cho ViT-CLS: theo lớp, gần
   tốt nhất trong các benchmark trên, không cần LRP, cài bằng hook attention + gradient
   (~50 dòng). Ưu tiên hơn TransLRP vì chênh lệch nhỏ mà cài đơn giản hơn nhiều.
2. Thêm **Attention Rollout** chỉ như baseline tham chiếu (không theo lớp, nên ghi rõ).
3. Chạy trên 4 run ViT baseline (ViT-CLS s42/s43, ViT-GAP s42/s43), cùng metric
   (EIB, pointing, IoU, prior) trên test → hàng robustness trong RQ1. Không cần train lại.
4. Kiểm tra artifact token: in phân phối norm patch token ở các block giữa của ViT-B/16 và
   xem vài heatmap; nếu có đốm nền chuẩn cao thì ghi vào phần hạn chế.
5. Không dùng các phương pháp này cho RQ2: L_loc tối ưu CAM của ViT-GAP, không tối ưu
   attention.

## Nguồn

- Chefer, Gur, Wolf. Transformer Interpretability Beyond Attention Visualization. CVPR 2021. https://arxiv.org/abs/2012.09838
- Chefer, Gur, Wolf. Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers. ICCV 2021. https://arxiv.org/abs/2103.15679
- Abnar, Zuidema. Quantifying Attention Flow in Transformers. ACL 2020.
- Komorowski, Baniecki, Biecek. Towards Evaluating Explanations of Vision Transformers for Medical Imaging. CVPRW 2023. https://arxiv.org/abs/2304.06133
- Barekatain, Glocker. Evaluating the Explainability of Vision Transformers in Medical Imaging. arXiv 2510.12021. https://arxiv.org/abs/2510.12021
- CLARiTy: A Vision Transformer for Multi-Label Classification and Weakly-Supervised Localization of Chest X-ray Pathologies. arXiv 2512.16700. https://arxiv.org/abs/2512.16700
- Xie et al. ViT-CX: Causal Explanation of Vision Transformers. IJCAI 2023. https://arxiv.org/abs/2211.03064
- Darcet, Oquab, Mairal, Bojanowski. Vision Transformers Need Registers. ICLR 2024. https://arxiv.org/abs/2309.16588
- Vision Transformers Don't Need Trained Registers. arXiv 2506.08010. https://arxiv.org/abs/2506.08010
- Saporta et al. Benchmarking saliency methods for chest X-ray interpretation. Nature Machine Intelligence 2022. https://www.nature.com/articles/s42256-022-00536-x
