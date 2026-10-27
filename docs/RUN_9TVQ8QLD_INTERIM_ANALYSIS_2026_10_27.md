# Đánh giá tạm thời run 9tvq8qld

Phân tích offline ngày 27/10/2026, dựa trên snapshot trong experiment_data; không truy vấn cloud hay thay đổi run. Manifest lấy lúc 10:49 UTC (17:49 Việt Nam). History có 364 dòng: 351 log train, 13 validation. Train tới iteration 3500, validation đầy đủ mới nhất ở 3250. Không gán mọi giá trị summary cho iteration 3500.

## Kết luận

Mô hình đang học IPH và IVH, nhưng EDH, SAH, SDH gần như chưa được phát hiện. Chưa đủ bằng chứng kết luận hội tụ, overfit hoặc Mean Teacher tốt hơn supervised. Chỉ số chọn checkpoint hiện tại thưởng nhiều cho ca âm tính nên best mDice chưa đại diện tốt cho chất lượng phát hiện tổn thương.

## Đối chiếu setting và phương pháp đã submit

- Source revision `8d750d7d12345e53d07c11450d94d22adda0d6df`, source_dirty=false, trùng HEAD khi phân tích. Config/model_spec của run là căn cứ chính; notebook local chỉ hỗ trợ đối chiếu.
- Fold 0, seed 1337, AMP, DDP 2 GPU; global batch 4 gồm 2 labeled + 2 unlabeled, mỗi GPU 1+1. Actual labeled cases=154. max_samples=2200 là trần, không phải số ca thực đã nạp.
- nnunet3d 7 stages, base filters 32, patch [32,192,192], anisotropic kernels, strided convolution, deep supervision. Sáu trọng số [1,.5,.25,.125,.0625,.03125] được chuẩn hóa; consistency chỉ trên head chính.
- SGD LR .01, momentum .9, weight decay .0001; poly power .7; 30000 iterations. Consistency .05, sigmoid rampup 8000; cosine rampdown từ 19500 về floor .0125 ở 30000. EMA cosine .99→.999.
- Teacher nhận ảnh sau augmentation chung; student thêm Gaussian noise std .02 và intensity scale ±5%. Supervised loss là (CE+Dice)/2 trên labeled. Hybrid consistency là 50% foreground soft Dice + 50% weighted channel MSE trên toàn batch, gồm labeled và unlabeled. BG weight .05 áp dụng cho nhánh MSE; confidence threshold=0 nghĩa là không lọc confidence.
- EMA cập nhật sau optimizer thành công; validation dùng EMA teacher. Chưa có validation student để đo lợi ích EMA.
- Gaussian sliding window bật, stride XY=32/Z=8, AMP bật; TTA và surface metrics tắt. surface_valid_pairs=0 không phải kết quả HD95 bằng 0.
- model.normalization ghi batchnorm nhưng model_spec không truyền trường này; backbone thực dùng InstanceNorm3d. Metadata này gây hiểu nhầm, không chứng minh run dùng BatchNorm.
- Augmentation thực trong source: scale .85–1.15 (p=.2), gamma .7–1.5 (p=.2), blur sigma .5–1 (p=.15), rot/flip, foreground crop p=.33. Crop không cân bằng từng lớp; annotation unlabeled được ẩn trước augmentation. Không thể gọi augmentation này tương đương hoàn toàn nnUNet.
- resolved.normalization/intensity_units=null: snapshot chưa đủ xác minh đơn vị cường độ và độ mạnh thực tế của noise .02.

## Diễn biến validation

Positive macro là trung bình đều Dice trên ca GT dương tính của năm lớp, được tính thêm cho phân tích; không phải tiêu chí chọn checkpoint đang chạy.

| Iteration | mDice đang log | Positive macro | IPH positive | IVH positive |
|---:|---:|---:|---:|---:|
| 500 | .33684 | .00000 | .00000 | .00000 |
| 1000 | .36641 | .06626 | .32973 | .00159 |
| 1500 | .44275 | .07941 | .38747 | .00956 |
| 2000 | .44040 | .08442 | .38897 | .03316 |
| 2500 | .45371 | .09816 | .40755 | .08326 |
| 2750 | .46258 | .10432 | .41519 | .10641 |
| 3000 | .45687 | .11261 | .41740 | .14562 |
| 3250 | .42482 | .12091 | .41706 | .18719 |

Best theo metric hiện tại là iteration 2750. Sau đó mDice giảm 3.78 điểm phần trăm, trong khi positive macro tăng và IVH tiếp tục cải thiện. Chưa đủ cơ sở gọi đây là overfitting.

## Từng lớp ở iteration 3250

| Lớp | Ca dương / 38 | Dice case mean | Dice positive | Precision micro | Recall micro |
|---|---:|---:|---:|---:|---:|
| EDH | 6 | .84211 | 0 | Không xác định* | 0 |
| IPH | 23 | .27875 | .41706 | .29803 | .70345 |
| IVH | 22 | .13469 | .18719 | .20692 | .18157 |
| SAH | 20 | .28955 | .000154 | .10938 | .000034 |
| SDH | 13 | .57899 | .000127 | .62069 | .000073 |

*EDH không có prediction: TP=FP=0. History để trống precision, summary ghi 0; về mặt toán học mẫu số bằng 0, nên không diễn giải là precision đo được bằng 0.

EDH Dice .8421 đến hoàn toàn từ 32/38 ca âm. SAH chỉ có 7 TP trên 205022 voxel GT; SDH chỉ có 18 TP trên 246841 voxel GT. Ba lớp này gần như bị bỏ sót toàn bộ.

IPH có recall .7035 nhưng precision .2980, 14/15 ca âm có FP. IVH có 15/16 ca âm có FP, đồng thời recall chỉ .1816. IVH đang học nhưng vẫn vừa bỏ sót vừa dự đoán nhầm.

## Vấn đề chọn checkpoint

Theo chính policy cả prediction và target rỗng thì Dice=1, dự đoán toàn background sẽ có mDice=(32+15+16+18+25)/(38×5)=.55789, cao hơn best .46258. Đây là baseline tính giải tích, chưa chạy inference. Không có nghĩa baseline phát hiện bệnh tốt hơn; nó cho thấy chỉ số hiện tại có thể thưởng việc không phát hiện tổn thương.

Một voxel FP trên ca âm khiến Dice lớp/ca từ 1 xuống 0; vì vậy mDice có thể giảm mạnh dù Dice trên ca dương tăng. Nên báo cáo positive Dice cùng precision/recall và FP ca âm, đồng thời lưu thêm checkpoint theo positive macro ở run sau. Giữ metric legacy để đối chiếu và ghi rõ thay đổi protocol.

## Loss và giai đoạn lịch học

Iteration 3500: supervised=.516667, raw consistency=.036926, weight=.010271; phần consistency đóng góp khoảng .000379, bằng .0734% supervised. Trung bình các log iteration 2501–3500, tỷ lệ này khoảng .0869%. Scalar loss không trực tiếp đo tỷ lệ gradient, nhưng chưa có bằng chứng consistency đang lấn át supervised.

Run mới đạt 11.67% ngân sách và 43.75% giai đoạn rampup. EMA alpha khoảng .99030, cửa sổ hiệu dụng xấp xỉ 103 updates; chưa phải alpha cố định .999. Rampdown 19500 chưa diễn ra. Chưa có supervised ablation cùng setting để chứng minh lợi ích unlabeled data.

## Ngân sách và bước tiếp theo

Runtime history tới iteration 3500 là 11863 giây (~3.30 giờ), bình quân 3.39 giây/iteration gồm validation/logging. Ngoại suy tuyến tính 30000 bước cần ~28.2 giờ; 12 giờ tương ứng ~12745 bước. Đây là ước lượng, không phải xác nhận giới hạn phiên. Comment notebook “30k ~5h, an toàn trong 12h” không phù hợp tốc độ snapshot. Cần kiểm tra ngân sách phiên thực và bảo đảm lấy được latest checkpoint để resume. Chưa có profiling để quy chi phí cho validation hay visualization.

Tiếp tục theo dõi ở 5000 và 8000 nếu ngân sách cho phép: EDH/SAH/SDH có thoát Dice/recall gần 0, IVH có tiếp tục tăng, FP của IPH/IVH có giảm. Nếu ba lớp vẫn gần 0 qua rampup, ưu tiên kiểm tra phân bố từng lớp trong 154 ca labeled, tỷ lệ crop chứa từng lớp, CE/Dice từng lớp và chuẩn hóa cường độ trước khi chỉnh consistency. Hiện chưa xác nhận nguyên nhân nằm ở dữ liệu, sampling hay loss.

Run cũ s9i9nsui có best .29024, nhưng source/policy/DS weights đã khác; chênh lệch không chứng minh riêng Mean Teacher cải thiện. Chưa đủ cơ sở so thắng nnUNet khi chưa thống nhất preprocessing, không gian metric, TTA, label budget và fold.

## Nguồn

- experiment_data/wandb/9tvq8qld/20261027T104902_174075Z_98f75ebf/{manifest,config,summary,export}.json
- experiment_data/wandb_exports/9tvq8qld/metrics_history.csv
- configs/methods/mean_teacher.yaml; methods/mean_teacher.py; methods/ssl_utils.py; methods/base_method.py; losses/consistency.py; evaluation/metrics.py; core/model_builder.py; models/backbones/nnunet3d.py
- kaggle_runs/mean_teacher/thnhnguynvncng/kaggle_bhsd_mean_teacher_dual_gpu.ipynb

Snapshot không có ảnh prediction/checkpoint mới để kiểm tra mask trực quan hoặc gradient. Không chạy lại training/tests vì đây là phân tích dữ liệu có sẵn.
