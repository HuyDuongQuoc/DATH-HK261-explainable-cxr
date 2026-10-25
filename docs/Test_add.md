# Protocol test hiện tại

Project dùng **internal test 2.250 ảnh** được tách từ manifest 15.000 ảnh có ground truth theo protocol 70/15/15, seed 42.

## Quy tắc sử dụng

- Train gồm 10.500 ảnh; validation và test mỗi tập 2.250 ảnh.
- Validation dùng cho early stopping, chọn model, hyperparameter và threshold.
- Internal test chỉ được chạy sau khi toàn bộ lựa chọn đã khóa.
- Mọi backbone và loss variant phải dùng cùng `assignment_sha256`.
- Checkpoint từ split 85/15 cũ không được resume hoặc dùng làm kết quả chính thức.

## `data/raw/test`

Thư mục `data/raw/test` chứa 3.000 ảnh official test dạng JPG nhưng workspace không có ground truth tương ứng. Các ảnh này không được đưa vào manifest, preprocessing, training, validation hoặc test metrics của protocol hiện tại.

Nếu sau này bổ sung `image_labels_test.csv` và `annotations_test.csv` từ PhysioNet, đó phải là một protocol external evaluation riêng; không thay thế âm thầm internal test và không áp dụng majority vote ba reader của training set.

## Đánh giá

Sau khi chọn best checkpoint bằng validation, chạy `scripts/evaluate.py --split test`. Kết quả phải gồm metrics, predictions và metadata có split fingerprint để truy xuất nguồn gốc.
