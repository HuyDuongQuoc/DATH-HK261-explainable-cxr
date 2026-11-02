# `vindr_s8515_exttest_v1`

- Train: 12.750 ảnh từ corpus annotation ba reader.
- Validation: 2.250 ảnh, giữ nguyên IDs của validation protocol 70/15/15 cũ.
- External test: 3.000 ảnh consensus ground truth, không overlap train/validation.
- External test chỉ dùng cho final evaluation.
- Ảnh, masks, archives và checkpoints không nằm trong thư mục này và không được Git theo dõi.

Xem `docs/data_protocol_s8515.md` để biết lệnh tái tạo, kiểm chứng, train và evaluate.
