# Việc còn cần kiểm chứng

Cập nhật ngày 29/10/2026 trong đợt tổ chức tài liệu. Đây là danh sách giới hạn đã thấy khi đối chiếu source; không sao chép toàn bộ finding cũ thành bug đang mở. Thứ tự thể hiện mức ảnh hưởng đến khả năng tái lập và đánh giá.

| Ưu tiên | Công việc | Bằng chứng để đóng |
|---|---|---|
| P1 | Nghiệm thu epoch-boundary resume, legacy fallback và metadata khi đi qua validation/final save | Test thực thi cho các nguồn checkpoint; xác minh completed epoch, split/batch compatibility, thiếu file và lịch sử resume; smoke DDP ở môi trường đích |
| P1 | Chốt checkpoint-selection metric cho thí nghiệm mới | Protocol định trước; so legacy mDice với positive Dice/support; không đổi tiêu chí hồi tố rồi báo như recipe cũ |
| P1 | Xác minh preprocessing/split giữa BHSD H5 và baseline nnU-Net | Commit/hash source ngoài repo, plans/fingerprint, manifest, patient IDs và split hashes; kiểm tra nhiều ca phù hợp |
| P1 | Quyết định grid benchmark và triển khai kiểm chứng nếu cần NIfTI gốc | Inverse transform có metadata đủ, prediction/GT cùng grid, regression theo ca và report protocol |
| P2 | Nghiệm thu local/cloud resume và dashboard theo môi trường thực | Kiểm tra artifact upload/download hoàn tất, run identity, lịch sử step, SDK/UI và CUDA/NCCL; không chỉ mock/offline |
| P2 | Profile chi phí trước khi tăng batch hoặc giãn validation | Đo train/val/render/upload/save, peak VRAM/RAM, throughput; ablation chất lượng và labeled exposure |
| P2 | Đóng băng môi trường thực nghiệm | Phiên bản Python/PyTorch/torchvision/CUDA và dependency từ môi trường đã chạy thành công; requirements hiện chưa là lockfile |

Exact resume giữa epoch với sampler cursor, worker/prefetch và RNG từng rank là mở rộng riêng, chưa được hứa bởi policy epoch-boundary. Các audit lịch sử có thể chứa finding đã sửa; cần tái hiện trên revision hiện tại trước khi thêm vào backlog này.
