# Phân tích run 9tvq8qld ở 16.250 iterations

Ngày phân tích: 28/10/2026. Phân tích offline, không thay đổi code huấn luyện hay run. Nguồn chính là snapshot `experiment_data/wandb/9tvq8qld/20261028T050921_123714Z_840978e3/{history,config,summary,artifacts}.json`, đối chiếu với code metric, crop, Mean Teacher và checkpoint. Tài liệu phân tích 27/10 chỉ có dữ liệu tới 3.500 iterations nên không dùng các kết luận về SAH/SDH trong tài liệu đó cho hiện tại.

## 1. Kết luận có thể khẳng định

- Train có 1.626 bản ghi tới 16.250; validation có 64 bản ghi từ 250 tới **16.000**, không có validation 16.250. Các metric validation trong summary thuộc 16.000. Run hoàn thành 54,17% ngân sách 30.000; việc dừng vì thời gian Kaggle là thông tin người dùng cung cấp, không suy từ metric thành lỗi thuật toán.
- Mô hình đã học IPH, IVH, SAH, SDH; **EDH không có TP trong mọi validation**. Thiếu 13.750 bước có thể làm kết quả chưa đạt mức cuối, nhưng không đủ để giải thích chắc chắn việc EDH hoàn toàn không được phát hiện.
- Chất lượng trên ca có tổn thương tăng mạnh tới khoảng 12–14,5k, rồi tăng chậm/dao động. Không có bằng chứng sụp đổ toàn bộ mô hình, cũng chưa đủ bằng chứng khẳng định overfit.
- Checkpoint selection hiện tại lệch với mục tiêu phân đoạn tổn thương: best legacy mDice ở 2.750, trong khi best positive macro ở 14.500. Đây là vấn đề thực tế cần ưu tiên trước khi đánh giá mô hình.
- Chưa có ablation supervised cùng điều kiện hay validation student trong history để chứng minh lợi ích riêng của unlabeled data/EMA.

## 2. Setting thực của run

Fold 0, seed 1337; 154 ca labeled thực tế; validation 38 ca. `max_samples=2200` là trần cấu hình, không xác nhận số unlabeled thực tế. DDP 2 GPU; global batch 4 = 2 labeled + 2 unlabeled, mỗi GPU 1+1. Backbone nnunet3d, 7 stages, base filters 32, patch 32×192×192, deep supervision. SGD LR .01, momentum .9, weight decay .0001, poly power .7, max_iteration 30.000. Validation dùng EMA teacher, sliding window Gaussian, stride Z=8/XY=32; không TTA, không surface metrics.

Consistency hybrid, hệ số tối đa .05, rampup 8.000, rampdown bắt đầu 19.500 về hệ số .0125 ở 30.000. EMA cosine .99→.999. Student noise std .02, intensity scale ±5%; confidence threshold 0. Supervised là CE+Dice với deep supervision. Consistency áp dụng cả labeled và unlabeled, head chính. Run chưa tới giai đoạn rampdown.

Config ghi normalization=batchnorm nhưng model_spec không truyền trường đó; backbone thực dùng InstanceNorm3d theo code đã đối chiếu trong báo cáo trước. `resolved.normalization/intensity_units=null` nên không khẳng định đơn vị intensity từ metadata này. Source revision của run là `8d750d7d12345e53d07c11450d94d22adda0d6df`; code local có thể đã thay đổi sau run, vì vậy config/history là căn cứ mô tả run, không đồng nhất mọi tính năng hiện tại với source đã chạy.

## 3. Diễn biến validation

Positive macro = trung bình đều Dice positive của 5 lớp. Dice positive của từng lớp chỉ tính trên ca GT có lớp đó; ca dương bị bỏ sót hoàn toàn vẫn tính 0. Đây là metric tính thêm từ history, không phải tiêu chí chọn best của run.

| Iteration | mDice legacy | Positive macro | IPH positive | IVH positive | SAH positive | SDH positive |
|---:|---:|---:|---:|---:|---:|---:|
| 2.750 | .46258 | .10432 | .4152 | .1064 | .00001 | .00000 |
| 5.000 | .30061 | .18759 | .4419 | .3643 | .0321 | .0996 |
| 8.000 | .34046 | .28388 | .5001 | .4272 | .1986 | .2936 |
| 10.000 | .34181 | .30920 | .5161 | .4517 | .2314 | .3467 |
| 12.000 | .35668 | .32665 | .5216 | .4944 | .2671 | .3501 |
| 14.000 | .36147 | .33502 | .5377 | .5042 | .2841 | .3492 |
| 14.500 | — | **.34167** | .5372 | .5224 | .2929 | .3559 |
| 15.000 | .37344 | .33782 | .5455 | .5055 | .2843 | .3538 |
| 16.000 | .36223 | .33542 | .5432 | .5136 | .2810 | .3393 |

EDH positive luôn bằng 0, đã tính vào mọi giá trị positive macro. Từ 2.750 tới 16.000, positive macro tăng **23,11 điểm phần trăm**, trong khi mDice legacy giảm 10,04 điểm. Từ best positive 14.500 tới 16.000 giảm khoảng .625 điểm phần trăm; chưa đủ gọi là overfit. Trung bình positive macro trong (12k,14k] là .33235, trong (14k,16k] là .33792: vẫn cải thiện nhẹ nếu nhìn theo cửa sổ thay vì một mốc.

## 4. Từng lớp tại validation 16.000

| Lớp | GT dương/38 | Dice case mean | Dice positive | Precision micro | Recall micro | FP trên ca âm | FP mL/ca âm trung bình |
|---|---:|---:|---:|---:|---:|---:|---:|
| EDH | 6 | .8421 | **0** | Không xác định | **0** | 0/32 | 0 |
| IPH | 23 | .3814 | .5432 | .5787 | .7636 | 13/15 | 2.651 |
| IVH | 22 | .3237 | .5136 | .6548 | .5841 | 15/16 | .322 |
| SAH | 20 | .1479 | .2810 | .3775 | .2921 | 18/18 | 2.055 |
| SDH | 13 | .1161 | .3393 | .3469 | .4431 | 25/25 | 5.645 |

Precision/recall micro cộng voxel toàn bộ cohort theo từng lớp; Dice positive là trung bình đều theo ca dương. Không tính F1 từ precision/recall micro rồi gọi đó là Dice positive. Dice micro tương ứng EDH/IPH/IVH/SAH/SDH là 0/.6584/.6174/.3293/.3891, khác nhau do cách trọng số ca lớn/nhỏ.

- **EDH:** TP=0, FP=0, FN=73.194 voxel. Mô hình hoàn toàn không xuất nhãn EDH trong validation. Precision có mẫu số TP+FP=0 nên không xác định; summary ghi 0 không biến nó thành một phép đo precision có ý nghĩa. Case Dice .8421 đến hoàn toàn từ 32 ca âm, không biểu thị phát hiện EDH tốt. Chưa biết EDH bị nhầm thành background hay lớp khác nếu không có confusion matrix.
- **IPH:** đang tốt nhất trong 5 lớp theo Dice positive; recall .7636 nhưng precision .5787, còn dư vùng/nhầm lớp. FP 116.062 voxel, FN 49.356; cần giảm FP đồng thời giữ recall.
- **IVH:** precision .6548 cao hơn recall .5841, còn thiếu tổn thương. Có FP ở 15/16 ca âm nhưng thể tích trung bình chỉ .322 mL; tỷ lệ FP-case cao có thể chịu ảnh hưởng các cụm rất nhỏ, cần xem phân bố component trước khi xử lý.
- **SAH:** recall .2921 và precision .3775 đều thấp; FN 145.134, FP 98.768 voxel. Đây là bài toán vừa bỏ sót vừa nhầm, không thể giải chỉ bằng tăng threshold.
- **SDH:** precision .3469, recall .4431; FP 205.917 so với TP 109.366 voxel. FP trung bình 5.645 mL trên ca âm là vấn đề đáng ưu tiên cùng EDH. Không thể giải thích tất cả FP của SDH bằng một vài voxel nhiễu.

Các FP là one-vs-rest: có thể gồm nhầm giữa các loại xuất huyết, không chỉ nhầm background thành tổn thương. Chưa có confusion matrix để phân biệt. Tỷ lệ FP-ca hiện tính bất kỳ FP voxel nào, chưa phải lesion-level detection metric với ngưỡng kích thước lâm sàng. Thể tích mL theo spacing/evaluation grid của pipeline, chưa kiểm chứng lại trên dữ liệu gốc.

HD95/ASD không được tính vì `val_surface_metrics=false`; `surface_valid_pairs=0` không có nghĩa sai số bề mặt bằng 0. Không có uncertainty/calibration, lesion recall hoặc khoảng tin cậy trong snapshot. EDH chỉ có 6 ca dương nên đặc biệt cần kiểm tra từng ca.

## 5. Vì sao best mDice gây hiểu nhầm?

Pipeline cho Dice=1 khi cả GT và prediction cùng rỗng. Một ca âm chỉ cần có FP cũng khiến Dice lớp đó bằng 0, bất kể 1 voxel hay một vùng lớn. Do đó metric này vừa thưởng việc giữ lớp hiếm luôn rỗng vừa phạt mạnh sự xuất hiện FP khi mô hình bắt đầu học lớp đó.

Nếu dự đoán toàn background, mDice legacy trên cohort này sẽ là:

`(32+15+16+18+25)/(38*5) = .55789`.

Đây là baseline tính giải tích từ số ca âm, chưa chạy inference. Nó cao hơn best .46258 của mô hình dù recall mọi lớp bằng 0. Điều này cho thấy mDice legacy đơn lẻ không đủ đánh giá khả năng phân đoạn tổn thương của run này.

Tại 16k, đóng góp của EDH vào mDice là .8421/5=.16842, chiếm khoảng 46,5% mDice .36223, dù không phát hiện EDH nào.

Đề xuất báo cáo đồng thời legacy mDice, macro positive Dice 5 lớp, precision/recall, FP mL và FP-case rate. Lưu thêm best theo positive macro, không âm thầm thay metric legacy hoặc so sánh hai protocol như cùng một chỉ số. Positive macro cũng có điểm yếu là không trực tiếp phạt FP trên ca âm, nên vẫn cần các cột FP. Nếu đặt tiêu chí FP phụ, phải chốt trước và dùng thống nhất cho mọi phương pháp.

## 6. Loss và mức đóng góp của consistency

| Cửa sổ train | Supervised trung bình | Raw consistency trung bình | Weighted consistency / supervised, trung bình từng log |
|---|---:|---:|---:|
| 1–2.000 | .54236 | .08958 | .0164% |
| 2.001–4.000 | .50687 | .06126 | .0947% |
| 6.001–8.000 | .47044 | .05949 | .5682% |
| 8.001–10.000 | .46470 | .06409 | .6901% |
| 12.001–14.000 | .45319 | .06184 | .6829% |
| 14.001–16.250 | .44775 | .06465 | .7225% |

Loss supervised giảm theo cửa sổ; raw consistency dao động quanh .06 sau rampup. Riêng log cuối supervised .30449 thấp hơn trung bình cửa sổ và không nên dùng một batch đó để kết luận xu hướng. Ở 16.250: weighted consistency=.05×.019668=.0009834, total=.3054726, đúng với công thức tổng loss.

Chưa thấy consistency lấn át supervised trên scalar loss. Tuy nhiên tỷ lệ loss không phải tỷ lệ gradient; chưa thể kết luận SSL không có tác dụng hoặc tăng hệ số 10–20 lần là đúng. Không có per-class consistency, teacher confidence, gradient norm hoặc student-vs-teacher validation để xác định cơ chế EDH bị triệt tiêu. Khi teacher luôn không dự đoán EDH, consistency có thể củng cố thiếu sót đó, nhưng đây là giả thuyết cần ablation, không phải nguyên nhân đã chứng minh.

## 7. 30.000 iterations và thời gian Kaggle

Runtime train cuối ~42.948 giây = 11,93 giờ. Tốc độ toàn run khoảng 2,64 giây/iteration; đoạn 12k→16k khoảng 2,44 giây/iteration, gồm chi phí validation/logging trong khoảng đo. Với tốc độ gần đây, 13.750 bước còn lại cần khoảng **9,1–10,1 giờ**, chưa tính khởi tạo/download và biến động phiên mới. Tổng 30k khoảng 21–22 giờ theo các ước lượng này, không phù hợp một phiên khoảng 12 giờ.

Tại 16.250 LR=.005792, còn khoảng 58% LR đầu; consistency rampdown 19.500 chưa bắt đầu. Vì vậy mô hình chưa trải qua toàn bộ lịch tối ưu. Chạy nốt là cần thiết để kết luận kết quả cuối của cấu hình này; không bảo đảm mọi lớp sẽ cải thiện hay EDH tự xuất hiện.

Artifacts snapshot liệt kê `BHSD_multiclass_MeanTeacher_fold0_20261027_072834_fold0_latest:v63` với metadata iteration **16.000**, size 552.980.931 bytes. Đây là bằng chứng có artifact được đăng ký, chưa phải kiểm chứng tải và nội dung tensor. Không thấy artifact checkpoint 16.250 trong danh sách. Checkpoint local có thể mới hơn do save_interval=50 nhưng cần kiểm tra output Kaggle thực tế.

Khi tiếp tục: dùng latest đầy đủ gần cuối, ưu tiên local 16.250 nếu có và kiểm chứng được, nếu không kiểm tra artifact 16.000; không dùng best legacy 2.750 để resume tiến trình. Giữ max_iteration=30.000 và global iteration gốc để LR/rampup/EMA/rampdown tiếp tục đúng. Khôi phục student, EMA teacher, optimizer, AMP scaler và RNG nếu checkpoint/source hỗ trợ; kiểm tra log iteration/LR/weight sau load. Chỉ nạp weights rồi reset iteration là một thí nghiệm fine-tune mới, không phải tiếp tục cùng lịch. Không khẳng định bitwise reproducibility nếu sampler/worker/DDP state không được khôi phục tương ứng.

Giữ source/config như run cũ cho nhánh hoàn tất 30k. Nếu thay sampling/loss/preprocessing, tạo nhánh/run mới và ghi rõ thay đổi. Lưu riêng checkpoint 14.500 nếu artifact còn dùng được để đánh giá; không suy rằng file best hiện tại chứa weights 14.500.

## 8. Thứ tự cải thiện đề xuất

### P0 — Đánh giá đúng và hoàn tất thí nghiệm hiện tại

1. Kiểm chứng và bảo toàn latest checkpoint, resume từ 16k/16,25k tới 30k; theo dõi 20k/25k/30k với cùng validation protocol. Lưu thêm best positive macro mà vẫn giữ best legacy; nếu đổi code khi resume phải ghi rõ phần chỉ thay logging/selection và kiểm chứng không đổi training.
2. Đánh giá checkpoints 2.750, 14.500, 16.000 và sau resume bằng cùng evaluator, ưu tiên EMA weights vì validation hiện dùng teacher. So sánh metric cùng policy, không chỉ đọc trường `best_mDice` kế thừa trong checkpoint.
3. Chọn cohort xem mask cố định có đủ EDH, SAH, SDH, ca âm có FP và tổn thương nhỏ; xuất CT/GT/pred/error, confusion matrix và component size. Bốn ảnh default trong summary không bảo đảm đại diện lớp hiếm; snapshot hiện không chứa mask/ảnh local đủ để xác minh trực quan.

### P1 — Điều tra EDH trước khi thay mô hình

4. Đếm số ca và voxel EDH trong **154 ca labeled**, số crop train thực sự chứa EDH, tỷ lệ bị mất sau augmentation/downsampling và mapping nhãn 0–5. Kiểm tra sáu ca EDH validation, phân bố kích thước và sự tương đồng preprocessing với train. Số EDH validation không suy ra tần suất EDH trong labeled train.
5. Crop hiện chọn foreground voxel từ mọi lớp gộp, xác suất .33, nên lớp nhiều voxel dễ được chọn hơn; không cân bằng từng lớp. Thử sampling theo lớp trên labeled: chọn lớp hiện diện trước rồi chọn voxel/case, giữ một phần crop ngẫu nhiên để học background và kiểm soát FP. Bắt đầu một ablation chỉ đổi sampling; có thể thử foreground probability .5, nhưng đây là giá trị thử nghiệm, không phải tối ưu đã chứng minh. Không dùng annotation unlabeled để cân bằng crop.
6. Theo dõi per-class CE/Dice và tỷ lệ dự đoán EDH; thử overfit một tập labeled nhỏ có đủ năm lớp để phát hiện lỗi dữ liệu/mapping/pipeline. Nếu EDH chưa học nổi ở bài kiểm tra này, chưa nên tăng consistency hay đổi backbone.

### P2 — Cân bằng FP/FN của SAH, SDH, IPH, IVH

7. Sau khi sampling được kiểm soát, thử riêng CE weighting mức vừa phải từ thống kê labeled và giữ Dice; không dùng inverse frequency cực lớn ngay. Tversky/focal là các nhánh thử sau, vì ưu tiên recall có thể làm FP vốn cao của SDH nặng hơn. Không đổi sampling, loss và consistency đồng thời nếu muốn biết điều gì giúp.
8. Thử hậu xử lý theo component và probability trên validation với tiêu chí giữ recall lớp nhỏ; so sánh trước/sau cả Dice positive và FP mL. Không dùng một ngưỡng kích thước chung cho IVH/SAH/SDH, không áp dụng mặc định chỉ giữ component lớn nhất cho tổn thương đa ổ. Chốt lựa chọn trước khi đánh giá test.
9. Rà soát clip/normalization, spacing, resampling nhãn và mức giữ cấu trúc mỏng/nhỏ; xác minh trên dữ liệu thay vì tự đổi HU window vì ảnh hiển thị. Patch depth 32 và XY192 chỉ là một ứng viên cần ablation nếu error map cho thấy thiếu ngữ cảnh; hiện chưa có bằng chứng phải tăng patch hay model.

### P3 — Đo lợi ích SSL và tối ưu ngân sách

10. Baseline supervised cùng backbone, split, seed, labeled set, batch labeled, augmentation, lịch LR và số optimizer updates; báo cáo số lượt thấy labeled để tránh chênh exposure. Bổ sung validation student và EMA. Sau đó so MT hiện tại với consistency=0; nếu teacher/student gần như trùng mới thử tăng consistency vừa phải, ví dụ .05→.1. Confidence mask cần đo tỷ lệ voxel giữ lại theo lớp, vì threshold chung cao có thể loại gần hết lớp hiếm. Mỗi thay đổi là một ablation, chưa có bảo đảm tăng điểm.
11. Profile train/validation/render/upload/checkpoint trước khi tối ưu. Hiện val_interval=250, visualization mỗi validation, stride khá dày, artifact checkpoint khoảng 553 MB mỗi validation; có thể tạo overhead nhưng chưa đo để quy nguyên nhân. Thử validation mỗi 500–1.000 và visualization mỗi 4 lần validation, giữ các checkpoint mốc và bản phục hồi định kỳ. Thay stride/TTA làm đổi protocol inference nên cần benchmark lại; giảm tần suất validation có thể bỏ lỡ best ngắn hạn.
12. Sau khi chọn cấu hình bằng validation, báo cáo nhiều seed/fold hoặc khoảng tin cậy theo ca, đặc biệt EDH chỉ 6 ca dương. Không liên tục chỉnh theo test; không kết luận thắng nnUNet khi preprocessing, ngân sách nhãn, fold và metric chưa thống nhất.

## 9. Điều chưa thể kết luận từ snapshot

Không xác nhận chất lượng mask từng ca, phân bố lớp train thực tế, confusion matrix, gradient, calibration hay nội dung checkpoint vì chưa tải/đánh giá tensor. Không khẳng định đạt 30k sẽ chữa EDH, MT tốt hơn supervised, model overfit hay lỗi nằm chắc chắn ở loss. Các khuyến nghị thay hyperparameter là thí nghiệm cần kiểm chứng, không phải kết quả đã đo.

Bản tính chi tiết và đồ thị dữ liệu có thể tái tạo bằng `python tools/analyze_run_9tvq8qld.py` (chỉ thư viện chuẩn). Không chạy training hay inference trong lần phân tích này.
