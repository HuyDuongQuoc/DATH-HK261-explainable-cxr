
# BÁO CÁO TIẾN ĐỘ ĐỒ ÁN TỔNG HỢP

**Nội dung: Hoàn thành Data Pipeline và Classification Baselines (Phase 2–3)**

Dạ thưa thầy, tiếp nối lần báo cáo trước về quá trình khảo sát dữ liệu và xác định phương pháp xử lý annotation, trong giai đoạn vừa qua nhóm đã tập trung triển khai hai phần chính: **xây dựng Data Pipeline (Phase 2)** và **huấn luyện, đánh giá ba mô hình Classification Baseline (Phase 3)**.

Ở **Phase 2**, nhóm đã hoàn thành quá trình tiền xử lý bộ dữ liệu VinDr-CXR gồm **15.000 ảnh X-quang có ground truth**. Các ảnh được resize về kích thước **224×224** và chuyển thành **3 kênh** để phù hợp với các mô hình pretrained trên ImageNet.

Đối với annotation, nhóm sử dụng **Majority Vote** để xác định nhãn phân loại của **14 finding**, với điều kiện ít nhất **2 trong 3 bác sĩ đồng thuận**. Đồng thời, nhóm áp dụng **Soft-pixel Consensus** để tổng hợp các bounding box thành **location mask**, thể hiện mức độ đồng thuận của các bác sĩ tại từng pixel. Những mask này sẽ được sử dụng cho việc đánh giá localization và nghiên cứu location-aware loss ở các giai đoạn tiếp theo.

Bộ dữ liệu sau xử lý được chia thành ba tập bằng phương pháp **Multilabel Stratification** với seed cố định:

- **Training:** 10.500 ảnh (70%)
- **Validation:** 2.250 ảnh (15%)
- **Internal Test:** 2.250 ảnh (15%)

Nhóm cũng đã hoàn thiện **PyTorch Dataset và DataLoader**, bảo đảm cả ba backbone sử dụng cùng dữ liệu, nhãn và quy trình tiền xử lý. Hiện tại, **Phase 2 đã hoàn thành** và có thể phục vụ các thực nghiệm tiếp theo mà không cần xây dựng lại bộ dữ liệu.

---

## 1. Xây dựng và huấn luyện Classification Baselines

Sau khi hoàn tất Data Pipeline, nhóm triển khai framework huấn luyện chung cho ba kiến trúc:

- **ResNet-50:** mô hình CNN sử dụng residual connections
- **DenseNet-121:** mô hình CNN sử dụng dense connections
- **ViT-B/16:** mô hình Vision Transformer xử lý ảnh thông qua các patch tokens

Cả ba mô hình đều sử dụng **pretrained weights từ ImageNet** và được điều chỉnh lớp đầu ra thành **14 logits**, tương ứng với bài toán **multi-label classification**.

Nhóm sử dụng **Weighted Binary Cross-Entropy** để giảm ảnh hưởng của sự mất cân bằng lớp, optimizer **AdamW** với learning rate **0.0001**, batch size **8** và tối đa **30 epochs**. Early stopping được thực hiện dựa trên **Macro-AUROC của tập validation** với patience bằng **5**.

Quá trình huấn luyện được thực hiện trên **Google Colab sử dụng GPU Tesla T4** và **mixed precision FP16** nhằm tiết kiệm bộ nhớ và tăng tốc độ xử lý. Framework đã hỗ trợ lưu checkpoint tốt nhất, tiếp tục huấn luyện khi bị gián đoạn và xuất kết quả đánh giá. Nhóm cũng đã thực hiện kiểm thử kỹ thuật, xác nhận các mô hình hoạt động đúng, có thể thực hiện forward/backward và lưu, khôi phục checkpoint.

Kết quả là **cả ba baseline đã hoàn thành huấn luyện, có đầy đủ artifact và đã được đánh giá trên tập Internal Test**. Các hyperparameter đã được cố định trước khi đánh giá test để bảo đảm tính nhất quán của thực nghiệm.

---

## 2. Kết quả thực nghiệm tổng quát

Kết quả trên **2.250 ảnh Internal Test** được tổng hợp như sau:

|Metric|ResNet-50|DenseNet-121|ViT-B/16|
|---|---|---|---|
|Macro-AUROC|0.9401|**0.9476**|0.8458|
|Macro-AUPRC|0.4470|**0.4842**|0.2015|
|Macro-Precision|**0.2532**|0.2350|0.1185|
|Macro-Recall|0.7831|**0.8510**|0.7250|
|Macro-F1|**0.3564**|0.3434|0.1881|

Từ bảng kết quả trên, có thể thấy **DenseNet-121 đạt hiệu năng phân loại tốt nhất xét theo Macro-AUROC và Macro-AUPRC**, lần lượt là **0.9476** và **0.4842**. ResNet-50 có kết quả khá gần với DenseNet-121 và đạt **Macro-Precision** cùng **Macro-F1** cao hơn tại threshold **0.5**. Trong khi đó, **ViT-B/16** có hiệu năng thấp hơn đáng kể so với hai mô hình CNN trong cùng điều kiện thực nghiệm.

**[Chèn Hình/Sơ đồ 1: “Internal-test macro metrics của ba baseline”]**  
_(Biểu đồ cột so sánh Macro AUROC, Macro AUPRC, Macro Precision, Macro Recall và Macro F1 của ResNet-50, DenseNet-121 và ViT-B/16.)_

Biểu đồ macro metrics cho thấy xu hướng này khá rõ: DenseNet-121 dẫn đầu về **AUROC, AUPRC và Recall**, trong khi ResNet-50 nhỉnh hơn ở **Precision và F1-score**. Điều này cho thấy việc lựa chọn mô hình không thể chỉ dựa vào một metric duy nhất, mà cần cân nhắc đồng thời khả năng phân biệt, khả năng phát hiện mẫu dương và độ cân bằng giữa precision–recall.

---

## 3. Phân tích ROC và Precision–Recall

Để đánh giá sâu hơn ở mức tổng quát, nhóm sử dụng thêm **micro-average ROC curve** và **micro-average Precision–Recall curve**.

**[Chèn Hình/Sơ đồ 2: “Internal-test ROC và Precision–Recall comparison”]**  
_(Biểu đồ gồm hai phần: bên trái là micro-average ROC curve, bên phải là micro-average Precision–Recall curve của ba baseline.)_

Khi quan sát **đường ROC**, DenseNet-121 đạt **AUC = 0.962**, ResNet-50 đạt **AUC = 0.959** và ViT-B/16 đạt **AUC = 0.885**. Hai mô hình CNN có đường ROC nằm gần góc trên bên trái, cho thấy khả năng phân biệt mẫu dương và mẫu âm tương đối tốt.

Tuy nhiên, khi xét sang **Precision–Recall curve**, sự khác biệt giữa các mô hình thể hiện rõ hơn. DenseNet-121 đạt **Average Precision = 0.587**, ResNet-50 đạt **0.573**, trong khi ViT-B/16 chỉ đạt **0.296**.

Đường baseline của Precision–Recall nằm ở mức khoảng **0.042**, phản ánh tỷ lệ nhãn dương thấp trong dữ liệu đánh giá. Vì vậy, nhóm nhận thấy rằng **AUROC cao chưa đủ để kết luận mô hình phát hiện tốt tất cả các finding**, đặc biệt với các lớp hiếm. Trong bối cảnh dữ liệu mất cân bằng, **AUPRC** và **F1-score** mang nhiều ý nghĩa thực tế hơn khi đánh giá khả năng dự đoán của mô hình.

---

## 4. Phân tích kết quả theo từng finding

Nhóm cũng đánh giá **AUROC, AUPRC và F1-score riêng cho từng finding** thay vì chỉ dựa vào macro-average.

**[Chèn Hình/Sơ đồ 3: “Internal-test performance theo từng finding”]**  
_(Heatmap gồm ba phần: AUROC, AUPRC và F1 @ threshold 0.5 cho 14 finding trên ba mô hình ResNet-50, DenseNet-121 và ViT-B/16.)_

Kết quả cho thấy một số finding như **Aortic enlargement, Cardiomegaly** và **Pleural effusion** có hiệu năng khá tốt trên hai mô hình CNN.

Ví dụ, với **DenseNet-121**:

- **Cardiomegaly** đạt **AUROC 0.977** và **AUPRC 0.883**
- **Aortic enlargement** đạt **AUROC 0.969** và **AUPRC 0.858**
- **Pleural effusion** đạt **AUROC 0.991** và **AUPRC 0.906**

Trong khi đó, các lớp có ít mẫu dương như **Atelectasis** và **Pneumothorax** vẫn là những trường hợp khó. Chẳng hạn, với **Atelectasis**, DenseNet-121 đạt **AUROC 0.923** nhưng **AUPRC chỉ 0.266** và **F1-score chỉ 0.135**. Điều này cho thấy mô hình vẫn có khả năng xếp hạng mẫu dương cao hơn mẫu âm tương đối tốt, nhưng khi chuyển sang quyết định nhị phân tại threshold **0.5**, hiệu năng còn hạn chế.

Ngoài ra, quan sát heatmap cũng cho thấy:

- Hai mô hình **CNN** nhìn chung vượt trội hơn **ViT-B/16** trên hầu hết các finding.
- Ở một số finding hiếm, **AUROC** vẫn cao nhưng **AUPRC** và **F1** lại thấp, cho thấy khoảng cách giữa khả năng xếp hạng và khả năng dự đoán dương chính xác.
- Các finding phổ biến hoặc có tín hiệu rõ ràng trên ảnh X-quang thường cho kết quả tốt hơn các finding hiếm hoặc khó phân biệt.

Từ đó, nhóm nhận định rằng **mất cân bằng lớp là một trong những yếu tố quan trọng ảnh hưởng đến chất lượng dự đoán**, và việc phân tích theo từng finding là cần thiết để tránh đánh giá quá lạc quan chỉ dựa trên macro-average.

---

## 5. So sánh hiệu quả tính toán

Bên cạnh hiệu năng phân loại, nhóm cũng theo dõi **chi phí đánh giá và tài nguyên sử dụng** của từng mô hình.

**[Chèn Hình/Sơ đồ 4: “Internal-test evaluation efficiency”]**  
_(Ba biểu đồ cột nhỏ so sánh Throughput, Total Evaluation Time và Peak GPU Memory của ba mô hình.)_

Kết quả cho thấy:

- **ViT-B/16** có **throughput cao nhất**, khoảng **246 ảnh/giây**
- **ResNet-50** đạt khoảng **194 ảnh/giây**
- **DenseNet-121** đạt khoảng **163 ảnh/giây**

Về **thời gian đánh giá toàn bộ internal test**:

- ResNet-50: khoảng **11.58 giây**
- DenseNet-121: khoảng **13.79 giây**
- ViT-B/16: khoảng **9.14 giây**

Tuy nhiên, về **bộ nhớ GPU cực đại**:

- ResNet-50 sử dụng khoảng **0.43 GB**
- DenseNet-121 sử dụng khoảng **0.16 GB**
- ViT-B/16 sử dụng khoảng **1.49 GB**

Như vậy, có thể thấy rằng **ViT-B/16 có tốc độ đánh giá nhanh nhất nhưng tiêu tốn bộ nhớ GPU lớn nhất**, trong khi **DenseNet-121 đạt hiệu năng phân loại tốt nhất nhưng throughput thấp hơn**, còn **ResNet-50** là lựa chọn tương đối cân bằng giữa hiệu năng và chi phí tính toán.

---

## 6. Nhận xét tổng thể và giới hạn hiện tại

Sau khi hoàn thành Phase 3, nhóm rút ra ba nhận xét chính.

Thứ nhất, **hai kiến trúc CNN là ResNet-50 và DenseNet-121 đang đạt hiệu năng phân loại tốt hơn ViT-B/16** trên bộ dữ liệu và thiết lập thực nghiệm hiện tại.

Thứ hai, **hiệu năng thay đổi tùy theo metric và từng finding**. DenseNet-121 nổi bật ở AUROC, AUPRC và Recall, nhưng ResNet-50 lại có Precision và F1 tốt hơn ở threshold 0.5. Điều này cho thấy việc chọn mô hình cần dựa trên **nhiều chỉ số** thay vì chỉ nhìn vào một giá trị tổng hợp.

Thứ ba, các kết quả hiện tại mới phản ánh **classification performance**, tức khả năng dự đoán nhãn bất thường từ ảnh X-quang. Chúng **chưa cho biết mô hình có thực sự tập trung vào đúng vùng bất thường do bác sĩ đánh dấu hay không**. Đây chính là động lực để nhóm chuyển sang giai đoạn đánh giá localization bằng Grad-CAM.

Ngoài ra, thực nghiệm hiện tại vẫn có một số giới hạn:

- mới sử dụng **một random seed**
- chưa thực hiện **external validation**
- threshold đang cố định ở **0.5**
- kết quả mới dừng ở **classification**, chưa đánh giá **localization quality**

Do đó, bộ kết quả này được xem là **baseline chính thức** cho phần thực nghiệm tiếp theo, chưa phải kết luận cuối cùng cho toàn bộ câu hỏi nghiên cứu.

---

## 7. Định hướng công việc tiếp theo

Trong giai đoạn tiếp theo, nhóm sẽ chuyển sang **Phase 4 — Grad-CAM & Localization Evaluation**, tập trung trả lời câu hỏi nghiên cứu thứ nhất:

**RQ1: Does better classification imply better localization?**

Cụ thể, nhóm dự kiến thực hiện các công việc sau:

- Triển khai **Grad-CAM** cho ResNet-50 và DenseNet-121
- Triển khai **Grad-CAM** cho ViT-B/16, trong đó cần xử lý patch tokens và chuyển về biểu diễn không gian phù hợp
- Đánh giá khả năng định vị bằng ba chỉ số:
    - **Pointing Game**
    - **Intersection over Union (IoU)**
    - **Energy Inside Box (EIB)**
- So sánh thứ hạng mô hình theo **classification metrics** với **localization metrics** để kiểm tra xem mô hình phân loại tốt hơn có thực sự định vị tốt hơn hay không

Các checkpoint baseline hiện tại sẽ được giữ nguyên và sử dụng cho quá trình đánh giá Grad-CAM. Nhóm sẽ sử dụng **validation set** để cố định các lựa chọn đánh giá cần thiết trước khi áp dụng trên **internal test**.

Sau khi hoàn thành Phase 4, nhóm sẽ tiếp tục **Phase 5**, xây dựng **Location-Aware Loss** nhằm khuyến khích mô hình tập trung activation vào vùng bounding box của bác sĩ. Khi đó, nhóm sẽ so sánh mô hình baseline với mô hình được huấn luyện bổ sung location-aware loss để trả lời câu hỏi nghiên cứu thứ hai.

