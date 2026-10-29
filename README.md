# MedSSL-BHSD

Framework huấn luyện và đánh giá phân đoạn CT BHSD: PICK, Mean Teacher, supervised, CPS, BCP và LeFeD. Model/method chỉ ghép được khi đáp ứng capability tương ứng.

## Cài đặt

Dùng môi trường Python với cặp PyTorch/torchvision phù hợp CUDA, sau đó cài dependency:

```bash
python -m pip install -r requirements.txt
# Nếu cần W&B, export, dashboard hoặc công cụ phân tích:
python -m pip install -r requirements-tools.txt
```

Requirements là danh mục dependency có cận dưới, chưa phải lockfile hay ma trận tương thích đã kiểm chứng. Kaggle controller cần thêm `kaggle` và `requests`. Ghi lại phiên bản thực tế của môi trường dùng cho thí nghiệm.

## Chạy nhanh

Chạy từ repository root; thay đường dẫn dữ liệu và output theo máy của bạn:

```bash
python train.py --dataset bhsd_multiclass --method mean_teacher --fold 0 \
  --data.root_path /path/to/BHSD_fold0 \
  --experiment.exp_dir /path/to/experiments \
  --experiment.name mt_fold0

python evaluate.py --dataset bhsd_multiclass \
  --root_path /path/to/BHSD_fold0 \
  --checkpoint /path/to/best_model.pth --patch_size 32 192 192
```

Xem [hướng dẫn training](docs/guides/training.md) trước khi chọn batch, config và DDP; [checkpoint/resume](docs/guides/checkpoint-resume.md) để tiếp tục run. Dữ liệu H5 và split cần được chuẩn bị sẵn; repo không tự tải dataset bằng các lệnh trên.

## Tài liệu và kiểm chứng

[Chỉ mục tài liệu](docs/README.md) phân biệt hướng dẫn hiện hành, kiến trúc, phân tích run và hồ sơ lịch sử. [Giao thức benchmark](docs/guides/benchmark-protocol.md) mô tả các điều kiện cần thống nhất trước khi so với nnU-Net.

```bash
python -m unittest discover -s tests -v
```

Một số test cần dependency tùy chọn, IPC hoặc môi trường GPU; đọc kết quả pass/fail/skip cụ thể. Kết quả test cũ trong archive không chứng nhận checkout hiện tại.
