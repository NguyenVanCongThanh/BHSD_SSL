# Audit lưu giữ run Kaggle — 29/10/2026

## Dọn local sau khi người dùng yêu cầu

Đã xóa thư mục riêng của `kpu7le0o`, `lgvtexrb`, `kaggle_runs/mean_teacher/dry_run_user`, ba symlink W&B hỏng và log W&B rỗng. Trước khi xóa, đã sao lưu ba thư mục vào `experiment_data/imports/cleanup/20261029T120023Z/removed-local-runs.tar.gz` (12.359 byte), đọc lại archive và kiểm chứng SHA-256 của cả 9 file thường. Inventory/hash và danh sách đã xóa nằm trong manifest.json cùng thư mục. Collection lịch sử vẫn giữ dữ liệu so sánh của hai run. Không xóa run/artifact cloud hoặc các run có bằng chứng sự cố. Nội dung audit bên dưới mô tả trạng thái trước lần dọn này.

Phạm vi: dữ liệu trong workspace, không truy cập hoặc thay đổi cloud. Không xóa file. Đối chiếu snapshot, collection, log, báo cáo và SHA-256; trạng thái cloud hiện tại chưa được xác minh.

## Kết luận

Chưa có run local nào đáng xóa toàn bộ ngay. `experiment_data/` chứa 39.680.016 byte (~37,84 MiB; `du` khoảng 39 MiB), chủ yếu log/JSON; không tìm thấy checkpoint `.pt`/`.pth` trong workspace. Có 8 thư mục run W&B, 2 snapshot Kaggle, 2 collection và một bộ export. Mỗi run chỉ còn một snapshot thật; `latest` là symlink, không phải bản sao. Không thấy file trùng SHA-256 lớn hơn 10 KB trong experiment_data, logs và kaggle_runs (bỏ qua alias latest). Kiểm tra này không chứng minh các định dạng JSON/CSV tương đương ngữ nghĩa.

## Quyết định theo run

| Run | Bằng chứng local | Đề xuất |
| --- | --- | --- |
| `donxcjc2` | `logs/kaggle-bhsd-mean-teacher-dual-gpu.log`: run ngày 28/10, train tới 18.280; validation tới 18.250 | Giữ ưu tiên cao. Chưa có snapshot riêng trong experiment_data; nên crawl config/history/artifact và bảo toàn checkpoint trước khi notebook/output bị thay thế. Log không xác nhận trạng thái kết thúc. |
| `9tvq8qld` | Snapshot ~20,54 MiB, export ~6,3 MiB; train 16.250, báo cáo phân tích riêng | Giữ toàn bộ. Cần để so sánh với run mới, phân tích metric và kiểm tra checkpoint. |
| `s9i9nsui` | ~4,65 MiB; train 3.996; log Kaggle lỗi NCCL và source thực thi | Giữ cùng snapshot Kaggle `thnhnguynvncng`. W&B ghi finished không đồng nghĩa train thành công; báo cáo đã xác nhận lỗi launcher. |
| `1ex7ty08`, `fum8iuny`, `a35mtk0i` | Tổng ~5,42 MiB; nhóm run 340–341 bước, events tài nguyên và log đối chiếu | Có thể đưa vào nhóm archive để bớt rối, chưa nên xóa. Được dùng trong báo cáo sự cố 26/10; xóa events sẽ mất bằng chứng so sánh RAM/GPU. |
| `5i9jspgj` | ~44 KiB; history 60 validation tới 12.000 | Giữ làm mốc lịch sử. Tên target_baseline trong công cụ không tự chứng minh đây là supervised baseline hoặc cùng protocol với run mới. |
| `lgvtexrb` | ~14 KiB; 13 validation tới 2.600 | Lưu trữ tham khảo; dung lượng quá nhỏ để đáng xóa. |
| `kpu7le0o` | ~13 KiB; 10 validation tới 2.000 | Ứng viên loại khỏi danh sách làm việc hằng ngày; vẫn nên giữ bản archive vì collection cũ dùng nó làm target_recent. |
| `4e4ciq84` | Log ở gốc repo: nonfinite gradients tại iteration 9 | Run thử lỗi sớm, ứng viên dọn trên giao diện cloud sau khi lưu config/summary cần thiết. Giữ log local vì chưa thấy bản trùng. |
| `tzf027rm` | Collection ghi failed; không có snapshot chi tiết | Ứng viên rà soát dọn cloud, chưa đủ bằng chứng quyết định xóa: cần kiểm tra iteration, lỗi và artifact trước. |

Collection ngày 28/10 liệt kê 17 run. Các ID chỉ có mục danh sách, không có snapshot chi tiết riêng ở local: `5m915z5b`, `8atasfq5`, `bbzkrgd5`, `rlf5s9fm`, `rsbcfm9z`, `netzw73w` và ID dài `BHSD_multiclass_MeanTeacher_nnunet3d_dual_gpu_fold0_mean_teacher_fold0`. Không kết luận các run này vô giá trị chỉ từ state crashed. Nếu muốn dọn cloud, đây là nhóm cần kiểm tra tiếp, cùng `tzf027rm`.

## Những mục dễ nhầm là rác

- `kaggle_runs/` (~136 KiB) là notebook/metadata phục vụ submit, không phải kho kết quả từng run. `dry_run_user/` là ứng viên dọn cấu hình thử, không giúp giải phóng dung lượng đáng kể.
- `wandb/` ở gốc có ba symlink hỏng (`latest-run`, `debug.log`, `debug-internal.log`) và một log rỗng: có thể dọn cache này khi không có tiến trình đang dùng. Đây không phải các snapshot dưới experiment_data.
- Hai collection không tương đương: bản 26/10 có 9 run và target_recent=kpu7le0o; bản 28/10 có 17 run và target_recent=9tvq8qld. Giữ cả hai nếu còn cần tái lập so sánh cũ.
- JSON export và CSV của 9tvq8qld có mục đích gần nhau nhưng chưa kiểm tra tương đương kiểu/null/cột với history snapshot. Chưa đề xuất xóa JSON chỉ vì đã có CSV.
- Log gốc là run 4e4ciq84, log trong logs là donxcjc2: cùng tên file không có nghĩa trùng run.
- Source trong snapshot Kaggle và manifest migration phục vụ truy vết; giữ lại dù có báo cáo lịch sử từng gọi chúng là tàn tích.

## Đề xuất quản lý

Giữ nhóm đang làm việc `donxcjc2`, `9tvq8qld`, mốc `5i9jspgj`; đánh dấu các run còn lại là lịch sử/sự cố trong một inventory. Giữ nguyên đường dẫn snapshot vì báo cáo đang dẫn tới đó. Nếu cần chuyển kho, chuyển cả experiment_data và dùng BHSD_EXPERIMENT_DATA theo hướng dẫn hiện hành, đồng thời kiểm tra các script phân tích có đường dẫn cố định.

Lợi ích dọn local hiện chủ yếu là giảm rối, không phải tiết kiệm ổ đĩa. Nếu mục tiêu là dung lượng checkpoint cloud, cần audit artifact versions và aliases riêng: artifacts.json chỉ là metadata, không chứng minh file đã tải, còn truy cập được hoặc có thể xóa an toàn. Không xóa run hay artifact cloud dựa duy nhất vào audit local này.

Tham chiếu: [hướng dẫn dữ liệu](../guides/experiment-data.md), [sự cố s9i9nsui](run-s9i9nsui-termination-2026-10-26.md), [phân tích 9tvq8qld](run-9tvq8qld-16250-2026-10-28.md). Các khuyến cáo retention rộng hơn trong audit 28/10 là hồ sơ lịch sử; hướng dẫn hiện hành yêu cầu kiểm tra bằng chứng và nội dung trước khi dọn.
