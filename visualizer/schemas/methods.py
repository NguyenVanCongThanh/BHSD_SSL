"""
Declarative Method Specifications (MethodSpec) for the Research Method Visualization System.
Demonstrates a truly method-agnostic architecture where Mean Teacher (primary deep-dive)
and CPS, BCP, PICK, LeFeD, Supervised are all expressed via structured declarative schemas.
"""

from typing import Any, Dict


def get_method_specs() -> Dict[str, Dict[str, Any]]:
    return {
        "mean_teacher": {
            "id": "mean_teacher",
            "name": "Mean Teacher (Enhanced 3D SSL)",
            "short_name": "Mean Teacher",
            "badge": "EMA Teacher-Student • Hybrid Consistency",
            "file": "methods/mean_teacher.py",
            "class_name": "MeanTeacherMethod",
            "paper": {
                "title": "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results",
                "authors": "Antti Tarvainen & Harri Valpola",
                "venue": "NeurIPS 2017",
                "adaptation": "Tối ưu hóa cho 3D Brain Hemorrhage CT (BHSD 6 lớp): Nhiễu bất đối xứng trên Student, Hybrid Consistency (50% FG Soft Dice + 50% Class-Balanced MSE với bg_weight=0.05), Sigmoid Ramp-up + Cosine Ramp-down, và Cosine EMA Schedule.",
            },
            "networks": [
                {
                    "id": "student",
                    "name": "Student Network (self.model)",
                    "role": "Trainable Primary Model (DDP/DP wrapped)",
                    "mode": "train()",
                    "grad": True,
                    "update_rule": "SGD with Momentum (0.9) + GradClip (12.0) + AMP GradScaler",
                    "color": "blue",
                },
                {
                    "id": "teacher",
                    "name": "EMA Teacher Network (self.ema_model)",
                    "role": "Detached Role Model & Validation Target",
                    "mode": "eval() + param.detach_()",
                    "grad": False,
                    "update_rule": "EMA: θ'_t = α_t θ'_{t-1} + (1 - α_t) θ_t (only when AMP grad is finite)",
                    "color": "purple",
                },
            ],
            "nodes": [
                {
                    "id": "dataset_sampler",
                    "title": "BHSDDataset & TwoStreamBatchSampler",
                    "subtitle": "Anti-Leakage Crop + [Labeled | Unlabeled]",
                    "category": "data",
                    "x": 40,
                    "y": 195,
                    "w": 225,
                    "h": 95,
                    "code_ref": "anti_leakage_crop",
                    "secondary_code_ref": "two_stream_sampler",
                    "shape_key": "batch_input",
                    "grad_state": "no_grad",
                    "summary": (
                        "Đọc volume .h5, xóa nhãn thật của mẫu Unlabeled (label = 0) TRƯỚC khi gọi RandomCrop "
                        "(33% foreground oversampling chỉ tác động lên mẫu Labeled). Ghép B_L mẫu Labeled và B_U mẫu Unlabeled."
                    ),
                    "details": [
                        "Tầng Shared 3D Transform (CPU): RandomRotFlip -> RandomScaling3D(0.85..1.15, p=0.2) -> GammaTransform(0.7..1.5, p=0.2) -> GaussianBlur3D(0.5..1.0, p=0.15) -> RandomCrop(patch_size) -> ToTensor().",
                        "Cả Student và Teacher đều nhận CÙNG một patch 3D đã căn chỉnh hình học từ bước này (không bị lệch tọa độ voxel).",
                        "Nếu DDP bật (world_size=2): DistributedTwoStreamBatchSampler chia đều global_batch_size cho mỗi GPU.",
                    ],
                },
                {
                    "id": "student_perturb",
                    "title": "Asymmetric Input Perturbation",
                    "subtitle": "X_s = X * scale + noise (Student only)",
                    "category": "transform",
                    "x": 315,
                    "y": 75,
                    "w": 225,
                    "h": 88,
                    "code_ref": "mt_train_step",
                    "shape_key": "student_input",
                    "grad_state": "no_grad",
                    "summary": (
                        "Chỉ làm nhiễu đầu vào của Student bằng biến đổi cường độ (±5%) và nhiễu cộng Gaussian (σ=0.02). "
                        "Teacher nhận trực tiếp volume_batch sạch."
                    ),
                    "details": [
                        "scale = 1.0 + (rand(B, 1, 1, 1, 1) * 2 - 1) * student_intensity_scale (shape: [B,1,1,1,1])",
                        "noise = randn_like(volume_batch) * student_noise_std (shape: [B,1,Z,Y,X])",
                        "Mục đích: Phá vỡ đối xứng Student-Teacher, buộc Student học đặc trưng bất biến với nhiễu xạ CT mà không làm mờ nhãn giả của Teacher.",
                    ],
                },
                {
                    "id": "student_net",
                    "title": "Student Network (self.model)",
                    "subtitle": "nnUNet3D (7 Stages) • train() • AMP FP16",
                    "category": "model_trainable",
                    "x": 590,
                    "y": 65,
                    "w": 240,
                    "h": 105,
                    "code_ref": "nnunet_arch",
                    "secondary_code_ref": "mt_train_step",
                    "shape_key": "student_outputs",
                    "grad_state": "requires_grad",
                    "summary": (
                        "Chạy forward trên toàn bộ batch (B_L + B_U) đã nhiễu. Khi deep_supervision=True, trả về "
                        "dict {'seg': student_logits, 'aux': [ds2, ds3, ds4, ds5, ds6]}."
                    ),
                    "details": [
                        "7 tầng kênh đặc trưng: [32, 64, 128, 256, 320, 320, 320].",
                        "Anisotropic kernels (1,3,3) ở enc1, enc2, dec2, dec1; Strides (1,2,2) ở pool1, pool2, pool5, pool6; Strides (2,2,2) ở pool3, pool4.",
                        "Trục Z chỉ giảm 4 lần (28 -> 7), trục Y và X giảm 64 lần (256 -> 4).",
                    ],
                },
                {
                    "id": "teacher_net",
                    "title": "EMA Teacher (self.ema_model)",
                    "subtitle": "Detached Clone • eval() • torch.no_grad()",
                    "category": "model_ema",
                    "x": 590,
                    "y": 300,
                    "w": 240,
                    "h": 105,
                    "code_ref": "mt_init",
                    "secondary_code_ref": "mt_train_step",
                    "shape_key": "teacher_outputs",
                    "grad_state": "detached",
                    "summary": (
                        "Bản sao deepcopy(raw_net) đã gọi param.detach_() và đặt cố định ở eval(). "
                        "Chạy trong khối with torch.no_grad() trên volume_batch SẠCH."
                    ),
                    "details": [
                        "Vì self.ema_model.training == False, nnUNet3D mặc định chỉ trả về tensor final_conv (bỏ qua 5 nhánh aux) trừ khi truyền return_all_scales=True.",
                        "Trong mean_teacher.yaml, consistency_multiscale=False nên Teacher chỉ xuất teacher_logits (B, 6, Z, Y, X), tiết kiệm VRAM và tốc độ.",
                        "Không bao giờ nhận gradient từ loss.backward().",
                    ],
                },
                {
                    "id": "sup_loss",
                    "title": "Multi-Scale Supervised Loss",
                    "subtitle": "Sliced [:labeled_bs] • 0.5*(CE + FG_Dice) • 6 Scales",
                    "category": "loss_sup",
                    "x": 890,
                    "y": 45,
                    "w": 255,
                    "h": 110,
                    "code_ref": "mt_train_step",
                    "secondary_code_ref": "deep_supervision",
                    "shape_key": "sup_loss_tensors",
                    "grad_state": "requires_grad",
                    "summary": (
                        "Cắt lát [:self.labeled_bs] từ student_logits và 5 đầu ra phụ student_aux. "
                        "Nội suy trilinear các đầu ra phụ về (Z, Y, X) và tính tổng có trọng số chuẩn hóa w̃_k."
                    ),
                    "details": [
                        "Tại mỗi tỉ lệ k: L_stage^(k) = 0.5 * CrossEntropy(aux_lab, lab_targets) + 0.5 * MaskDiceLoss(aux_lab, lab_targets).",
                        "MaskDiceLoss(include_background=False) loại bỏ lớp 0 (Background) khỏi trung bình Dice để tập trung vào 5 lớp xuất huyết.",
                        "Trọng số chuẩn hóa 6 tỉ lệ: [0.5079, 0.2540, 0.1270, 0.0635, 0.0317, 0.0159].",
                    ],
                },
                {
                    "id": "cons_loss",
                    "title": "Hybrid Consistency Loss",
                    "subtitle": "Full Batch [0..B-1] • 0.5*FG_SoftDice + 0.5*Weighted_MSE",
                    "category": "loss_cons",
                    "x": 890,
                    "y": 195,
                    "w": 255,
                    "h": 115,
                    "code_ref": "consistency_loss",
                    "secondary_code_ref": "mt_consistency_schedule",
                    "shape_key": "cons_loss_tensors",
                    "grad_state": "requires_grad",
                    "summary": (
                        "So khớp phân phối xác suất FP32 Softmax của Student và Teacher trên TOÀN BỘ batch (cả Labeled + Unlabeled). "
                        "Nhân với trọng số động w_cons(t) có Sigmoid Ramp-up + Cosine Ramp-down."
                    ),
                    "details": [
                        "Ép kiểu .float() (FP32) trước F.softmax(dim=1) để tránh underflow FP16.",
                        "50% Continuous Foreground Soft Dice trên 5 kênh tổn thương [:, 1:] (bỏ kênh 0 Background).",
                        "50% Class-Balanced MSE trên 6 kênh với class_weights = [0.05, 1, 1, 1, 1, 1] (giảm 20 lần áp lực từ Background).",
                    ],
                },
                {
                    "id": "grad_opt",
                    "title": "GradientHandler & Optimizer",
                    "subtitle": "AMP Unscale • DDP Finite Check • Clip 12.0 • SGD",
                    "category": "optimizer",
                    "x": 890,
                    "y": 345,
                    "w": 255,
                    "h": 95,
                    "code_ref": "gradient_handler",
                    "secondary_code_ref": "lr_scheduler",
                    "shape_key": "optimizer_state",
                    "grad_state": "backward_sink",
                    "summary": (
                        "Lan truyền ngược từ L_total = L_sup + w_cons(t)*L_cons về Student. "
                        "Kiểm tra isfinite trên mọi GPU DDP, clip_grad_norm_(12.0), cập nhật SGD rồi kích hoạt hook EMA."
                    ),
                    "details": [
                        "Nếu bất kỳ GPU nào có grad Inf/NaN (is_finite == 0): Bỏ qua cả scaler.step(optimizer) VÀ bỏ qua after_optimizer_step (bảo vệ EMA Teacher).",
                        "Learning Rate cập nhật mỗi iteration theo hàm đa thức: lr(t) = base_lr * (1 - t/T_max)^poly_power (poly_power = 0.7).",
                    ],
                },
                {
                    "id": "val_sliding",
                    "title": "3D Gaussian Sliding-Window Validation",
                    "subtitle": "Evaluates EMA Teacher • Stride Z=8, XY=32",
                    "category": "validation",
                    "x": 315,
                    "y": 310,
                    "w": 225,
                    "h": 90,
                    "code_ref": "mt_validate",
                    "secondary_code_ref": "ema_update_fn",
                    "shape_key": "validation_state",
                    "grad_state": "no_grad",
                    "summary": (
                        "Định kỳ mỗi val_interval (200 bước), đánh giá trọng số của EMA Teacher (không phải Student) "
                        "bằng cửa sổ trượt 3D có mặt nạ Gaussian (σ = 1/8 patch_size)."
                    ),
                    "details": [
                        "Lưu cả state_dict của Student ('net') và Teacher ('ema_net') vào best_model.pth & checkpoint_latest.pth.",
                        "Khi evaluate.py hoặc load_checkpoint(..., prefer_ema=True) chạy, hệ thống tự động nạp 'ema_net'.",
                    ],
                },
            ],
            "edges": [
                {"id": "e1", "from": "dataset_sampler", "to": "student_perturb", "label": "volume_batch (Clean)", "kind": "data"},
                {"id": "e2", "from": "dataset_sampler", "to": "teacher_net", "label": "volume_batch (Clean [0..B-1])", "kind": "data"},
                {"id": "e3", "from": "student_perturb", "to": "student_net", "label": "student_input (Perturbed)", "kind": "data"},
                {"id": "e4", "from": "dataset_sampler", "to": "sup_loss", "label": "label_batch[:B_L]", "kind": "label"},
                {"id": "e5", "from": "student_net", "to": "sup_loss", "label": "student_logits[:B_L] + aux[:B_L]", "kind": "forward_grad"},
                {"id": "e6", "from": "student_net", "to": "cons_loss", "label": "student_logits [0..B-1]", "kind": "forward_grad"},
                {"id": "e7", "from": "teacher_net", "to": "cons_loss", "label": "teacher_logits [0..B-1] (no_grad)", "kind": "forward_nograd"},
                {"id": "e8", "from": "sup_loss", "to": "grad_opt", "label": "L_supervised", "kind": "loss_grad"},
                {"id": "e9", "from": "cons_loss", "to": "grad_opt", "label": "w_cons(t) * L_consistency", "kind": "loss_grad"},
                {"id": "e10", "from": "grad_opt", "to": "student_net", "label": "∇_θ L_total (SGD Update)", "kind": "backward_grad"},
                {"id": "e11", "from": "student_net", "to": "teacher_net", "label": "EMA: α_t θ' + (1-α_t)θ", "kind": "ema_transfer"},
                {"id": "e12", "from": "teacher_net", "to": "val_sliding", "label": "ema_model weights", "kind": "validation"},
            ],
            "execution_steps": [
                {
                    "step": 1,
                    "title": "Two-Stream Batch Assembly & Anti-Leakage 3D Crop",
                    "code_line": "methods/base_method.py:611 & data/dataset_bhsd.py:218",
                    "active_nodes": ["dataset_sampler"],
                    "active_edges": [],
                    "what_happens": (
                        "DataLoader lấy B_L mẫu từ primary_indices (Labeled) và B_U mẫu từ secondary_indices (Unlabeled). "
                        "Trong BHSDDataset.__getitem__, nhãn của mẫu Unlabeled bị gán bằng 0 TRƯỚC khi qua RandomCrop(foreground_prob=0.33). "
                        "Nhờ đó, 33% crop tập trung vào vùng xuất huyết chỉ áp dụng trên mẫu Labeled, còn mẫu Unlabeled được crop ngẫu nhiên thuần túy."
                    ),
                    "key_formula": "X \\in \\mathbb{R}^{(B_L + B_U) \\times 1 \\times Z \\times Y \\times X}, \\quad Y_{[B_L : B]} \\equiv 0",
                },
                {
                    "step": 2,
                    "title": "Teacher Clean Forward Pass (torch.no_grad + eval mode)",
                    "code_line": "methods/mean_teacher.py:118-123",
                    "active_nodes": ["dataset_sampler", "teacher_net"],
                    "active_edges": ["e2"],
                    "what_happens": (
                        "Volume sạch (chưa cộng nhiễu Gaussian hay scale cường độ) đi qua self.ema_model trong ngữ cảnh torch.no_grad() và AMP autocast. "
                        "Vì consistency_multiscale=False trong cấu hình chuẩn, Teacher chỉ tính đầu ra độ phân giải gốc teacher_logits (B, 6, Z, Y, X)."
                    ),
                    "key_formula": "Z_T = f_{\\theta'}^{\\text{eval}}(X_{\\text{clean}}) \\in \\mathbb{R}^{B \\times 6 \\times Z \\times Y \\times X} \\quad (\\text{no\\_grad})",
                },
                {
                    "step": 3,
                    "title": "Asymmetric Student Perturbation & Multi-Scale Forward",
                    "code_line": "methods/mean_teacher.py:126-135",
                    "active_nodes": ["student_perturb", "student_net"],
                    "active_edges": ["e1", "e3"],
                    "what_happens": (
                        "Khi perturb_student=True, mỗi mẫu trong batch được nhân với hệ số cường độ ngẫu nhiên scale ∈ [0.95, 1.05] "
                        "và cộng nhiễu Gaussian N(0, 0.02²). Mạng Student (ở chế độ train()) chạy forward sinh ra student_logits (tỉ lệ 1x) "
                        "cùng 5 tensor phụ student_aux[0..4] từ các tầng giải mã d2..d6."
                    ),
                    "key_formula": "X_S = X_{\\text{clean}} \\odot s + \\epsilon, \\quad \\{Z_S^{(0)}, Z_S^{(1..5)}\\} = f_{\\theta}^{\\text{train}}(X_S)",
                },
                {
                    "step": 4,
                    "title": "Multi-Scale Supervised Loss on Labeled Slice [:labeled_bs]",
                    "code_line": "methods/mean_teacher.py:138-160",
                    "active_nodes": ["student_net", "sup_loss"],
                    "active_edges": ["e4", "e5"],
                    "what_happens": (
                        "Cắt B_L mẫu đầu tiên từ student_logits và 5 đầu ra phụ student_aux. Các đầu ra phụ có độ phân giải thấp hơn "
                        "(ví dụ 7×8×8 ở ds_conv6) được phóng đại bằng F.interpolate(mode='trilinear') về đúng (Z, Y, X). "
                        "Tính trung bình 0.5*CE + 0.5*MaskDice(loại bỏ Background) tại mỗi tỉ lệ rồi nhân với trọng số chuẩn hóa w̃_k = 2⁻ᵏ / ∑2⁻ʲ."
                    ),
                    "key_formula": "\\mathcal{L}_{\\text{sup}} = \\sum_{k=0}^{5} \\tilde{w}_k \\cdot \\frac{1}{2}\\Big(\\mathcal{L}_{\\text{CE}}(\\uparrow Z_{S, [:B_L]}^{(k)}, Y_{[:B_L]}) + \\mathcal{L}_{\\text{Dice}}^{c \\ge 1}(\\uparrow Z_{S, [:B_L]}^{(k)}, Y_{[:B_L]})\\Big)",
                },
                {
                    "step": 5,
                    "title": "Hybrid Consistency Loss Across Full Batch [0..B-1]",
                    "code_line": "methods/mean_teacher.py:163 & losses/consistency.py:63-107",
                    "active_nodes": ["student_net", "teacher_net", "cons_loss"],
                    "active_edges": ["e6", "e7"],
                    "what_happens": (
                        "Ép cả student_logits và teacher_logits (toàn bộ B_L + B_U mẫu) về FP32 (.float()) rồi lấy Softmax theo chiều kênh. "
                        "Tính 50% Continuous Foreground Soft Dice (chỉ trên 5 lớp xuất huyết 1..5) + 50% Class-Balanced MSE "
                        "(trọng số lớp [0.05, 1, 1, 1, 1, 1] giảm 20 lần ảnh hưởng của Background)."
                    ),
                    "key_formula": "\\mathcal{L}_{\\text{cons}} = 0.5 \\cdot \\mathcal{L}_{\\text{SoftDice}}(P_S^{1..5}, P_T^{1..5}) + 0.5 \\cdot \\frac{\\sum_{c=0}^5 w_c \\cdot \\text{MSE}_c(P_{S,c}, P_{T,c})}{\\sum_{c=0}^5 w_c}",
                },
                {
                    "step": 6,
                    "title": "Dynamic Consistency Weighting & Total Loss Combination",
                    "code_line": "methods/mean_teacher.py:94-110, 175-177",
                    "active_nodes": ["sup_loss", "cons_loss", "grad_opt"],
                    "active_edges": ["e8", "e9"],
                    "what_happens": (
                        "Tính hệ số w_cons(t): tăng theo đường cong Sigmoid trong giai đoạn ramp-up (theo iter nếu rampup > 100, "
                        "theo epoch nếu <= 100), giữ đỉnh ở consistency (0.05), và suy giảm Cosine từ bước consistency_rampdown_start (7800) "
                        "về 25% giá trị đỉnh ở cuối quá trình huấn luyện để chống pseudo-label drift trên các lớp hiếm (IVH/SAH)."
                    ),
                    "key_formula": "\\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{sup}} + w_{\\text{cons}}(t) \\cdot \\mathcal{L}_{\\text{cons}}",
                },
                {
                    "step": 7,
                    "title": "AMP Scaled Backward, DDP Finite Consensus, Clip 12.0 & SGD Step",
                    "code_line": "training/gradient_handler.py:51-88",
                    "active_nodes": ["grad_opt", "student_net"],
                    "active_edges": ["e10"],
                    "what_happens": (
                        "GradScaler nhân scale vào L_total và gọi backward(). Đạo hàm CHỈ chảy ngược về Student (Teacher bị chặn bởi detach_ và no_grad). "
                        "Sau khi unscale_(), các GPU đồng bộ all_reduce(MIN) để kiểm tra không có gradient Inf/NaN. "
                        "Nếu toàn bộ hữu hạn: cắt chuẩn gradient ở max_norm=12.0 và cập nhật trọng số Student bằng SGD."
                    ),
                    "key_formula": "\\text{if } \\bigwedge_{\\text{rank}} \\text{isfinite}(\\nabla_\\theta \\mathcal{L}): \\quad g \\leftarrow \\text{ClipNorm}(\\nabla_\\theta \\mathcal{L}, 12.0), \\quad \\theta_t \\leftarrow \\text{SGD}(\\theta_{t-1}, g, \\text{lr}_t)",
                },
                {
                    "step": 8,
                    "title": "Cosine EMA Teacher Update & Periodic Sliding-Window Validation",
                    "code_line": "methods/mean_teacher.py:191-195 & methods/ssl_utils.py:93-120",
                    "active_nodes": ["student_net", "teacher_net", "val_sliding"],
                    "active_edges": ["e11", "e12"],
                    "what_happens": (
                        "Chỉ khi bước SGD ở Step 7 thành công, hook after_optimizer_step mới được gọi. Hệ số α_t được tính theo lịch Cosine "
                        "từ 0.99 lên 0.999, cập nhật mọi parameter của Teacher: θ'_t = α_t θ'_{t-1} + (1 - α_t) θ_t và copy trực tiếp các buffer. "
                        "Mỗi val_interval bước, EMA Teacher được đem đi đánh giá bằng cửa sổ trượt 3D Gaussian."
                    ),
                    "key_formula": "\\theta'_t \\leftarrow \\alpha_t \\theta'_{t-1} + (1 - \\alpha_t) \\theta_t, \\quad \\alpha_t = 0.99 + (\\alpha_{\\max} - 0.99)\\frac{1 - \\cos(\\pi t / T_{\\max})}{2}",
                },
            ],
            "paper_vs_impl": [
                {
                    "dimension": "Không gian & Đầu ra",
                    "paper": "Phân loại ảnh 2D (CIFAR-10, ImageNet) — đầu ra vector [B, C].",
                    "impl": "Phân đoạn thể tích CT 3D 6 lớp [B, 6, Z, Y, X] kèm 5 đầu ra phụ Deep Supervision (d2..d6).",
                    "impact": "Đòi hỏi xử lý bất đẳng hướng trục Z (anisotropic kernels/strides) và nội suy trilinear 3D.",
                },
                {
                    "dimension": "Chiến lược Nhiễu (Perturbation)",
                    "paper": "Nhiễu đối xứng độc lập trên cả Student và Teacher (dịch chuyển + Gaussian noise + Dropout ở cả 2 mạng).",
                    "impl": "Nhiễu bất đối xứng (Asymmetric): Chung biến đổi hình học 3D tại Dataset; Teacher ở eval() nhận ảnh sạch, chỉ Student nhận scale (±5%) + Gaussian noise (σ=0.02).",
                    "impact": "Giữ cho dự đoán của Teacher sắc nét và khớp tọa độ từng voxel 3D với Student, tránh làm nhòe ranh giới xuất huyết mỏng.",
                },
                {
                    "dimension": "Hàm Consistency Loss",
                    "paper": "MSE đồng đều trên mọi lớp hoặc KL Divergence.",
                    "impl": "Hybrid Loss: 50% Foreground-only Soft Dice (bỏ kênh 0 Background) + 50% Class-Balanced MSE (bg_consistency_weight = 0.05).",
                    "impact": "Chống sụp đổ mô hình (model collapse về toàn Background) do >98% voxel trong CT sọ não là nền.",
                },
                {
                    "dimension": "Lịch Consistency Weight w_cons(t)",
                    "paper": "Chỉ có Sigmoid Ramp-up ở đầu quá trình train rồi giữ cố định.",
                    "impl": "Sigmoid Ramp-up (4000 iters) + Plateau + Late-Stage Cosine Ramp-down (từ iter 7800 giảm về 25% ở iter 12000).",
                    "impact": "Ngăn chặn confirmation bias / pseudo-label drift ở giai đoạn cuối đối với các lớp hiếm (IVH, SAH).",
                },
                {
                    "dimension": "Lịch Hệ số EMA α_t",
                    "paper": "Hằng số α = 0.99/0.999 hoặc min(1 - 1/(t+1), α).",
                    "impl": "Cosine EMA Schedule tăng mượt từ min_alpha = 0.99 lên ema_decay = 0.999 theo tiến độ t / T_max.",
                    "impact": "Giúp Teacher học nhanh ở giai đoạn đầu (cửa sổ nhớ ~100 bước) và cực kỳ ổn định ở cuối train (cửa sổ nhớ ~1000 bước).",
                },
            ],
        },
        "cps": {
            "id": "cps",
            "name": "Cross Pseudo Supervision (CPS)",
            "short_name": "CPS",
            "badge": "Dual-Network θ₁ ↔ θ₂ • Cross Argmax Pseudo-Labels",
            "file": "methods/cps.py",
            "class_name": "CPSMethod",
            "paper": {
                "title": "Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision",
                "authors": "Xiaokang Chen et al.",
                "venue": "CVPR 2021",
                "adaptation": "Hai mạng 3D độc lập (model và model2) khởi tạo Kaiming khác nhau, cùng huấn luyện bằng 2 bộ tối ưu SGD riêng biệt và giám sát chéo nhau bằng nhãn giả cứng (argmax).",
            },
            "networks": [
                {"id": "net1", "name": "Network 1 (self.model, θ₁)", "role": "Branch 1 Trainable Model", "mode": "train()", "grad": True, "update_rule": "self.optimizer (SGD)", "color": "blue"},
                {"id": "net2", "name": "Network 2 (self.model2, θ₂)", "role": "Branch 2 Re-initialized Model", "mode": "train()", "grad": True, "update_rule": "self.optimizer2 (SGD)", "color": "emerald"},
            ],
            "nodes": [
                {"id": "cps_data", "title": "TwoStreamBatchSampler", "subtitle": "Full Volume Batch [0..B-1]", "category": "data", "x": 40, "y": 195, "w": 215, "h": 90, "code_ref": "two_stream_sampler", "shape_key": "batch_input", "grad_state": "no_grad", "summary": "Cung cấp cùng một volume_batch (B_L + B_U) cho cả Network 1 và Network 2.", "details": ["Không dùng EMA; sự đa dạng đến từ khởi tạo trọng số Kaiming độc lập giữa model1 và model2."]},
                {"id": "cps_net1", "title": "Network 1 (self.model)", "subtitle": "Parameters θ₁ • Optimizer 1", "category": "model_trainable", "x": 320, "y": 75, "w": 225, "h": 95, "code_ref": "cps_train_step", "shape_key": "student_outputs", "grad_state": "requires_grad", "summary": "Tạo logits1 và aux1 trên toàn batch.", "details": ["Có nhánh Deep Supervision riêng trên tập mẫu có nhãn [:labeled_bs]."]},
                {"id": "cps_net2", "title": "Network 2 (self.model2)", "subtitle": "Parameters θ₂ (Re-init) • Optimizer 2", "category": "model_trainable", "x": 320, "y": 305, "w": 225, "h": 95, "code_ref": "cps_train_step", "shape_key": "student_outputs", "grad_state": "requires_grad", "summary": "Khởi tạo lại trọng số Kaiming Normal độc lập trong CPSMethod.__init__, tạo logits2 và aux2.", "details": ["Được bọc DDP riêng và gắn với self.optimizer2."]},
                {"id": "cps_pl", "title": "Hard Argmax Pseudo-Labels", "subtitle": "pl1 = argmax(P₁), pl2 = argmax(P₂)", "category": "transform", "x": 615, "y": 195, "w": 230, "h": 95, "code_ref": "cps_train_step", "shape_key": "pseudo_labels", "grad_state": "detached", "summary": "Trong khối torch.no_grad(), lấy argmax theo chiều kênh để tạo nhãn giả rời rạc pl1, pl2 ∈ {0..5}.", "details": ["Phép argmax không khả vi nên tự động chặn gradient đi ngược qua nhánh tạo nhãn giả."]},
                {"id": "cps_sup", "title": "Dual Supervised Loss", "subtitle": "L_sup1(θ₁) + L_sup2(θ₂) on [:B_L]", "category": "loss_sup", "x": 900, "y": 75, "w": 240, "h": 95, "code_ref": "cps_train_step", "shape_key": "sup_loss_tensors", "grad_state": "requires_grad", "summary": "Tính 0.5*(CE + MaskDice) đa tỉ lệ cho cả Network 1 và Network 2 trên lát cắt có nhãn [:labeled_bs].", "details": ["Trọng số Deep Supervision mặc định: [0.533, 0.267, 0.133, 0.067]."]},
                {"id": "cps_cross", "title": "Cross-Supervision CE Loss", "subtitle": "CE(logits1, pl2) + CE(logits2, pl1)", "category": "loss_cons", "x": 900, "y": 305, "w": 240, "h": 95, "code_ref": "cps_train_step", "shape_key": "cons_loss_tensors", "grad_state": "requires_grad", "summary": "Network 1 lấy pl2 làm nhãn đích; Network 2 lấy pl1 làm nhãn đích trên toàn bộ batch [0..B-1].", "details": ["L_total = (L_sup1 + w_cps * L_cps1) + (L_sup2 + w_cps * L_cps2). GradientHandler cập nhật đồng thời cả optimizer1 và optimizer2."]},
            ],
            "edges": [
                {"id": "c1", "from": "cps_data", "to": "cps_net1", "label": "volume_batch", "kind": "data"},
                {"id": "c2", "from": "cps_data", "to": "cps_net2", "label": "volume_batch", "kind": "data"},
                {"id": "c3", "from": "cps_net1", "to": "cps_pl", "label": "logits1 (no_grad)", "kind": "forward_nograd"},
                {"id": "c4", "from": "cps_net2", "to": "cps_pl", "label": "logits2 (no_grad)", "kind": "forward_nograd"},
                {"id": "c5", "from": "cps_net1", "to": "cps_sup", "label": "logits1[:B_L]", "kind": "forward_grad"},
                {"id": "c6", "from": "cps_net2", "to": "cps_sup", "label": "logits2[:B_L]", "kind": "forward_grad"},
                {"id": "c7", "from": "cps_pl", "to": "cps_cross", "label": "pl1, pl2 (Targets)", "kind": "label"},
                {"id": "c8", "from": "cps_net1", "to": "cps_cross", "label": "logits1 -> match pl2", "kind": "forward_grad"},
                {"id": "c9", "from": "cps_net2", "to": "cps_cross", "label": "logits2 -> match pl1", "kind": "forward_grad"},
            ],
            "execution_steps": [
                {"step": 1, "title": "Dual Network Forward", "code_line": "methods/cps.py:95-96", "active_nodes": ["cps_data", "cps_net1", "cps_net2"], "active_edges": ["c1", "c2"], "what_happens": "Cả 2 mạng chạy forward trên cùng volume_batch tạo logits1, aux1 và logits2, aux2.", "key_formula": "Z_1 = f_{\\theta_1}(X), \\quad Z_2 = f_{\\theta_2}(X)"},
                {"step": 2, "title": "Dual Multi-Scale Supervised Loss", "code_line": "methods/cps.py:99-107", "active_nodes": ["cps_net1", "cps_net2", "cps_sup"], "active_edges": ["c5", "c6"], "what_happens": "Tính CE + Dice đa tỉ lệ trên [:labeled_bs] cho cả hai mạng.", "key_formula": "\\mathcal{L}_{\\text{sup}} = \\mathcal{L}_{\\text{sup}}^{(1)}(Z_{1,[:B_L]}, Y) + \\mathcal{L}_{\\text{sup}}^{(2)}(Z_{2,[:B_L]}, Y)"},
                {"step": 3, "title": "One-Hot Argmax Pseudo-Label Generation", "code_line": "methods/cps.py:110-114", "active_nodes": ["cps_pl"], "active_edges": ["c3", "c4"], "what_happens": "Tạo nhãn giả rời rạc pl1 = argmax(softmax(logits1)) và pl2 = argmax(softmax(logits2)) trong torch.no_grad().", "key_formula": "\\hat{Y}_1 = \\arg\\max_c P_1, \\quad \\hat{Y}_2 = \\arg\\max_c P_2"},
                {"step": 4, "title": "Cross Pseudo Supervision & Joint Optimization", "code_line": "methods/cps.py:117-125, 141-142", "active_nodes": ["cps_cross", "cps_net1", "cps_net2"], "active_edges": ["c7", "c8", "c9"], "what_happens": "Tính CE chéo nhau: CE(logits1, pl2) + CE(logits2, pl1) và cập nhật đồng thời cả 2 bộ tối ưu optimizer1, optimizer2.", "key_formula": "\\mathcal{L}_{\\text{cps}} = \\text{CE}(Z_1, \\hat{Y}_2) + \\text{CE}(Z_2, \\hat{Y}_1)"},
            ],
            "paper_vs_impl": [],
        },
        "bcp": {
            "id": "bcp",
            "name": "Bidirectional Copy-Paste (BCP)",
            "short_name": "BCP",
            "badge": "3D CutMix Labeled ↔ Unlabeled • Region-Weighted Loss",
            "file": "methods/bcp.py",
            "class_name": "BCPMethod",
            "paper": {
                "title": "Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation",
                "authors": "Yunhao Bai et al.",
                "venue": "CVPR 2023",
                "adaptation": "Cắt-dán hai chiều bằng mặt nạ hộp 3D context_mask (mask_ratio = 2/3) giữa các cặp mẫu Labeled (img_a, img_b) và Unlabeled (unimg_a, unimg_b); gán trọng số voxel 1.0 cho vùng nhãn thật và u_weight=0.5 cho vùng nhãn giả.",
            },
            "networks": [
                {"id": "bcp_net", "name": "Single Shared Network (self.model)", "role": "Pseudo-Labeler (no_grad) & Mixed Predictor (grad)", "mode": "train()", "grad": True, "update_rule": "SGD Optimizer", "color": "blue"},
            ],
            "nodes": [
                {"id": "bcp_split", "title": "SSLBatch Partition (split_ssl_batch)", "subtitle": "img_a, img_b (Lab) | unimg_a, unimg_b (Unlab)", "category": "data", "x": 40, "y": 195, "w": 235, "h": 95, "code_ref": "bcp_train_step", "shape_key": "batch_input", "grad_state": "no_grad", "summary": "Yêu cầu số mẫu Labeled và Unlabeled trên mỗi GPU là số chẵn >= 2 để chia đôi thành sub_bs = B_L / 2.", "details": ["Ví dụ B_local = 4 (2L + 2U) -> sub_bs = 1: img_a (1), img_b (1), unimg_a (1), unimg_b (1)."]},
                {"id": "bcp_pl", "title": "Unlabeled Pseudo-Labeling", "subtitle": "plab = argmax(softmax(model(unimg)))", "category": "model_ema", "x": 330, "y": 75, "w": 235, "h": 95, "code_ref": "bcp_train_step", "shape_key": "teacher_outputs", "grad_state": "detached", "summary": "Chạy model(unimg) trong torch.no_grad() để lấy nhãn giả plab_a, plab_b và sinh mặt nạ hộp 3D context_mask (tỉ lệ thể tích 2/3).", "details": ["img_mask có kích thước (D, H, W): vùng bên trong hộp bằng 0, vùng ngoài bằng 1 (hoặc ngược lại)."]},
                {"id": "bcp_mix", "title": "Bidirectional 3D Copy-Paste", "subtitle": "mix_img_1 (L in + U out) & mix_img_2 (U in + L out)", "category": "transform", "x": 330, "y": 295, "w": 235, "h": 95, "code_ref": "bcp_train_step", "shape_key": "student_input", "grad_state": "no_grad", "summary": "Trộn 2 chiều giữa Labeled và Unlabeled cho cả ảnh 3D và nhãn mục tiêu.", "details": ["mix_img_1 = img_a * M + unimg_a * (1 - M)", "mix_img_2 = unimg_b * M + img_b * (1 - M)"]},
                {"id": "bcp_forward", "title": "Model Forward on Mixed Batch", "subtitle": "mix_logits = model(cat(mix_1, mix_2))", "category": "model_trainable", "x": 635, "y": 195, "w": 235, "h": 95, "code_ref": "bcp_train_step", "shape_key": "student_outputs", "grad_state": "requires_grad", "summary": "Đưa tổ hợp ảnh đã ghép qua mạng chính có tính gradient.", "details": ["Mạng buộc phải học ngữ cảnh liền mạch qua biên cắt giữa mẫu Labeled và Unlabeled."]},
                {"id": "bcp_loss", "title": "Region-Weighted Masked CE + Dice", "subtitle": "Weight = 1.0 (GT region) + u_weight (PL region)", "category": "loss_cons", "x": 920, "y": 195, "w": 235, "h": 95, "code_ref": "bcp_train_step", "shape_key": "cons_loss_tensors", "grad_state": "requires_grad", "summary": "Tính compute_masked_ce_loss và MaskDiceLoss với mặt nạ trọng số không gian.", "details": ["forward_weights = M + u_weight * (1 - M)", "backward_weights = (1 - M) + u_weight * M"]},
            ],
            "edges": [
                {"id": "b1", "from": "bcp_split", "to": "bcp_pl", "label": "unimg", "kind": "data"},
                {"id": "b2", "from": "bcp_split", "to": "bcp_mix", "label": "img_a,b & unimg_a,b", "kind": "data"},
                {"id": "b3", "from": "bcp_pl", "to": "bcp_mix", "label": "plab_a,b + img_mask", "kind": "forward_nograd"},
                {"id": "b4", "from": "bcp_mix", "to": "bcp_forward", "label": "mix_imgs", "kind": "data"},
                {"id": "b5", "from": "bcp_forward", "to": "bcp_loss", "label": "mix_logits", "kind": "forward_grad"},
                {"id": "b6", "from": "bcp_mix", "to": "bcp_loss", "label": "mix_labs + spatial weights", "kind": "label"},
            ],
            "execution_steps": [
                {"step": 1, "title": "Paired Batch Splitting", "code_line": "methods/bcp.py:34-39", "active_nodes": ["bcp_split"], "active_edges": [], "what_happens": "Chia batch thành các cặp Labeled (img_a, img_b) và Unlabeled (unimg_a, unimg_b).", "key_formula": "B_L = 2 \\cdot B_{\\text{sub}}, \\quad B_U = 2 \\cdot B_{\\text{sub}}"},
                {"step": 2, "title": "Unlabeled Pseudo-Labeling & 3D Context Mask", "code_line": "methods/bcp.py:42-48", "active_nodes": ["bcp_pl"], "active_edges": ["b1"], "what_happens": "Tạo nhãn giả plab trên unimg và sinh mặt nạ hộp 3D M với tỉ lệ mask_ratio = 2/3.", "key_formula": "\\hat{Y}_u = \\arg\\max_c \\text{Softmax}(f_\\theta(X_u)), \\quad M \\in \\{0, 1\\}^{Z \\times Y \\times X}"},
                {"step": 3, "title": "Bidirectional CutMix Synthesis", "code_line": "methods/bcp.py:51-59", "active_nodes": ["bcp_mix", "bcp_forward"], "active_edges": ["b2", "b3", "b4"], "what_happens": "Ghép hai chiều giữa Labeled và Unlabeled rồi đưa qua model.", "key_formula": "X_{\\text{mix1}} = M \\odot X_a^l + (1-M) \\odot X_a^u, \\quad X_{\\text{mix2}} = M \\odot X_b^u + (1-M) \\odot X_b^l"},
                {"step": 4, "title": "Spatially-Weighted Dice + CE Optimization", "code_line": "methods/bcp.py:64-72", "active_nodes": ["bcp_loss"], "active_edges": ["b5", "b6"], "what_happens": "Tính CE và Dice có trọng số vùng (1.0 cho vùng nhãn thật, u_weight=0.5 cho vùng nhãn giả).", "key_formula": "W_1 = M + w_u(1 - M), \\quad W_2 = (1 - M) + w_u M"},
            ],
            "paper_vs_impl": [],
        },
        "pick": {
            "id": "pick",
            "name": "PICK (2-Phase CutMix + Lesion-Informed MIM)",
            "short_name": "PICK",
            "badge": "2 Phases (Pre-train + Self-train) • 3 Decoders (Main, MIM, Aux)",
            "file": "methods/pick.py",
            "class_name": "PICKMethod",
            "paper": {
                "title": "PICK: Pseudo-label Informed CutMix & Masked Image Modeling for 3D Hemorrhage Segmentation",
                "authors": "MedSSL-BHSD Implementation",
                "venue": "BHSD Benchmark",
                "adaptation": "Huấn luyện 2 giai đoạn (Pre-train 4000 bước trên Labeled + Self-train 12000 bước trên Labeled & Unlabeled) với kiến trúc pick_vnet gồm 1 Encoder và 3 nhánh Decoder (Main Segmentation, MIM L1 Reconstruction, Auxiliary Refinement).",
            },
            "networks": [
                {"id": "pick_net", "name": "PICK-VNet (Main + MIM + Aux Decoders)", "role": "Shared Encoder + 3 Specialized Decoders", "mode": "mode='main' | 'mim' | 'aux'", "grad": True, "update_rule": "SGD Optimizer across 2 Phases", "color": "blue"},
            ],
            "nodes": [
                {"id": "pick_data", "title": "SSLBatch & MIM Mask Generator", "subtitle": "Pre-train (Labeled) / Self-train (Lab + Unlab)", "category": "data", "x": 40, "y": 195, "w": 225, "h": 95, "code_ref": "pick_train_step", "shape_key": "batch_input", "grad_state": "no_grad", "summary": "Quản lý 2 phase: pre_train (4000 iters) và self_train (12000 iters).", "details": ["Tự động khôi phục best_model.pth từ pre_train trước khi bắt đầu self_train."]},
                {"id": "pick_main", "title": "Main Decoder (CutMix Seg)", "subtitle": "Bidirectional CutMix + Lesion Mask", "category": "model_trainable", "x": 325, "y": 75, "w": 235, "h": 95, "code_ref": "pick_train_step", "shape_key": "student_outputs", "grad_state": "requires_grad", "summary": "Dự đoán phân đoạn trên batch CutMix và trích xuất vùng nghi ngờ tổn thương (fg_prob > 0.5) làm mặt nạ mask_region.", "details": ["Trong self_train, loss_main chỉ tính trên vùng có nhãn thật của ảnh ghép (mask=img_mask)."]},
                {"id": "pick_mim", "title": "MIM Decoder (mode='mim')", "subtitle": "Lesion-Informed Volumetric Inpainting", "category": "transform", "x": 325, "y": 300, "w": 235, "h": 95, "code_ref": "pick_train_step", "shape_key": "student_input", "grad_state": "requires_grad", "summary": "Che vùng tổn thương X * (1 - mask_region) và buộc nhánh MIM tái tạo lại cường độ CT gốc bằng L1 Loss.", "details": ["loss_mim = lambda_mim * (|X - X_rec| * mask_region).sum() / mask_region.sum()"]},
                {"id": "pick_aux", "title": "Aux Decoder (mode='aux')", "subtitle": "Segmentation on Inpainted Volume", "category": "model_trainable", "x": 635, "y": 195, "w": 235, "h": 95, "code_ref": "pick_train_step", "shape_key": "student_outputs", "grad_state": "requires_grad", "summary": "Ghép vùng vừa tái tạo (detach) vào ảnh gốc tạo re_batch rồi đưa qua nhánh Aux để phân đoạn.", "details": ["re_batch = X * (1 - M) + mim_outputs.detach() * M"]},
                {"id": "pick_loss", "title": "Triple Compound Loss", "subtitle": "L_total = L_main + λ_mim * L_mim + L_aux", "category": "loss_sup", "x": 920, "y": 195, "w": 235, "h": 95, "code_ref": "pick_train_step", "shape_key": "sup_loss_tensors", "grad_state": "requires_grad", "summary": "Tổng hợp 3 hàm mục tiêu từ 3 nhánh giải mã trong cùng một bước backward.", "details": ["Sử dụng DiceCELoss(include_background=False) cho cả L_main và L_aux."]},
            ],
            "edges": [
                {"id": "p1", "from": "pick_data", "to": "pick_main", "label": "cutmix_batch", "kind": "data"},
                {"id": "p2", "from": "pick_main", "to": "pick_mim", "label": "mask_region (fg_prob > 0.5)", "kind": "forward_nograd"},
                {"id": "p3", "from": "pick_mim", "to": "pick_aux", "label": "re_batch (inpainted)", "kind": "data"},
                {"id": "p4", "from": "pick_main", "to": "pick_loss", "label": "loss_main", "kind": "loss_grad"},
                {"id": "p5", "from": "pick_mim", "to": "pick_loss", "label": "loss_mim (L1)", "kind": "loss_grad"},
                {"id": "p6", "from": "pick_aux", "to": "pick_loss", "label": "loss_aux", "kind": "loss_grad"},
            ],
            "execution_steps": [
                {"step": 1, "title": "CutMix & Lesion Pseudo-Mask Extraction", "code_line": "methods/pick.py:125-138", "active_nodes": ["pick_data", "pick_main"], "active_edges": ["p1"], "what_happens": "Tạo mặt nạ tổn thương un_ps_lab (xác suất foreground > 0.5) và thực hiện CutMix 2 chiều.", "key_formula": "M_{\\text{lesion}} = \\mathbb{I}\\left(\\sum_{c=1}^5 P_{u,c} > 0.5\\right)"},
                {"step": 2, "title": "Lesion-Masked Image Modeling (mode='mim')", "code_line": "methods/pick.py:145-154", "active_nodes": ["pick_mim"], "active_edges": ["p2"], "what_happens": "Che vùng tổn thương và tái tạo lại cấu trúc CT qua nhánh MIM với hàm mất mát L1.", "key_formula": "\\mathcal{L}_{\\text{mim}} = \\lambda_{\\text{mim}} \\frac{\\| (X_u - \\hat{X}_{\\text{mim}}) \\odot M_{\\text{lesion}} \\|_1}{\\|M_{\\text{lesion}}\\|_1 + \\epsilon}"},
                {"step": 3, "title": "Auxiliary Segmentation on Reconstructed Volume (mode='aux')", "code_line": "methods/pick.py:149-157", "active_nodes": ["pick_aux", "pick_loss"], "active_edges": ["p3", "p4", "p5", "p6"], "what_happens": "Thay thế vùng bị che bằng kết quả tái tạo (đã detach) rồi đưa qua nhánh Aux và tối ưu tổng L_main + L_mim + L_aux.", "key_formula": "\\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{main}} + \\mathcal{L}_{\\text{mim}} + \\mathcal{L}_{\\text{aux}}"},
            ],
            "paper_vs_impl": [],
        },
        "lefed": {
            "id": "lefed",
            "name": "LeFeD (Learning Feedback Dual-Branch)",
            "short_name": "LeFeD",
            "badge": "3-Step Recurrent Feedback Loop • Dual Decoders",
            "file": "methods/lefed.py",
            "class_name": "LeFeDMethod",
            "paper": {
                "title": "Learning Feedback for Semi-Supervised 3D Medical Image Segmentation",
                "authors": "LeFeD Adaptation for BHSD",
                "venue": "Medical SSL",
                "adaptation": "Mỗi batch thực hiện vòng lặp 3 bước phản hồi (feedback_steps=3). Hiệu số đặc trưng giữa 2 nhánh giải mã en = scale * (feat2.detach() - feat1.detach()) được truyền ngược về điều biến Encoder ở bước kế tiếp. Mỗi bước gọi apply_gradients riêng!",
            },
            "networks": [
                {"id": "lefed_net", "name": "LeFeD Dual-Decoder Network", "role": "Shared Feedback-Modulated Encoder + Decoder 1 & 2", "mode": "train() (3 steps/batch)", "grad": True, "update_rule": "3 sequential backward + optimizer steps per batch!", "color": "blue"},
            ],
            "nodes": [
                {"id": "lf_data", "title": "Input Batch [0..B-1]", "subtitle": "Reused across 3 Feedback Steps", "category": "data", "x": 40, "y": 195, "w": 215, "h": 90, "code_ref": "lefed_train_step", "shape_key": "batch_input", "grad_state": "no_grad", "summary": "Cùng một volume_batch được đưa qua mạng 3 lần liên tiếp (step = 0, 1, 2).", "details": ["Ở step = 0: tín hiệu phản hồi en = None.", "Ở step = 1, 2: en = feedback_scale * (feat2.detach() - feat1.detach())."]},
                {"id": "lf_enc", "title": "Feedback-Modulated Encoder", "subtitle": "Input: (volume_batch, en)", "category": "model_trainable", "x": 315, "y": 195, "w": 235, "h": 95, "code_ref": "lefed_train_step", "shape_key": "student_input", "grad_state": "requires_grad", "summary": "Nhận ảnh 3D và danh sách tensor phản hồi en từ bước trước để tinh chỉnh đặc trưng.", "details": ["feedback_scale mặc định = 1e-3."]},
                {"id": "lf_dec", "title": "Dual Decoders (Out 1 & Out 2)", "subtitle": "outputs1, outputs2, masks, deep_outs", "category": "model_trainable", "x": 615, "y": 195, "w": 235, "h": 95, "code_ref": "lefed_train_step", "shape_key": "student_outputs", "grad_state": "requires_grad", "summary": "Sinh ra 2 dự đoán chính (outputs1, outputs2), các đặc trưng trung gian masks=[feat2, feat1], và đầu ra phụ deep_outs.", "details": ["Mỗi bước s ∈ {0,1,2} đều tính đầy đủ Supervised Loss, Deep Supervision Loss và Dual-Decoder Consistency Loss."]},
                {"id": "lf_loss", "title": "Per-Step Loss & Immediate Step()", "subtitle": "L_sup1,2 + L_ds + w_cons * MSE(P₁, P₂)", "category": "loss_cons", "x": 910, "y": 195, "w": 240, "h": 95, "code_ref": "lefed_train_step", "shape_key": "cons_loss_tensors", "grad_state": "backward_sink", "summary": "Trong optimize_batch(), mỗi bước yield loss_step và gọi ngay self.apply_gradients(loss, iter_num).", "details": ["Tổng cộng có 3 lần backward() và 3 lần optimizer.step() trong 1 iteration!"]},
            ],
            "edges": [
                {"id": "l1", "from": "lf_data", "to": "lf_enc", "label": "volume_batch", "kind": "data"},
                {"id": "l2", "from": "lf_enc", "to": "lf_dec", "label": "multi-stage features", "kind": "forward_grad"},
                {"id": "l3", "from": "lf_dec", "to": "lf_enc", "label": "Feedback: en = scale*(f₂ - f₁)", "kind": "ema_transfer"},
                {"id": "l4", "from": "lf_dec", "to": "lf_loss", "label": "outputs1, outputs2, deep_outs", "kind": "loss_grad"},
            ],
            "execution_steps": [
                {"step": 1, "title": "Step s=0: Initial Pass (en = None)", "code_line": "methods/lefed.py:62-64", "active_nodes": ["lf_data", "lf_enc", "lf_dec"], "active_edges": ["l1", "l2"], "what_happens": "Chạy forward lần đầu khi chưa có tín hiệu phản hồi (en = None), thu được outputs1, outputs2 và đặc trưng trung gian.", "key_formula": "O_1^{(0)}, O_2^{(0)}, [F_2^{(0)}, F_1^{(0)}] = f_\\theta(X, \\text{None})"},
                {"step": 2, "title": "Feedback Signal Computation & Step Optimization", "code_line": "methods/lefed.py:70-75, 128-135", "active_nodes": ["lf_dec", "lf_loss", "lf_enc"], "active_edges": ["l3", "l4"], "what_happens": "Tính sự khác biệt đặc trưng đã detach: en = 10⁻³ * (F₂ - F₁), đồng thời thực hiện backward() và optimizer.step() ngay cho bước hiện tại, rồi lặp lại cho s=1, 2.", "key_formula": "e_n^{(s)} = \\gamma \\cdot \\big(\\text{stop\\_grad}(F_2^{(s)}) - \\text{stop\\_grad}(F_1^{(s)})\\big)"},
            ],
            "paper_vs_impl": [],
        },
        "supervised": {
            "id": "supervised",
            "name": "Supervised Baseline (Labeled-Only Lower Bound)",
            "short_name": "Supervised",
            "badge": "Labeled Pool Only • 0.5*(CE + Foreground Dice)",
            "file": "methods/supervised.py",
            "class_name": "SupervisedMethod",
            "paper": {
                "title": "Supervised 3D Volumetric Segmentation Baseline",
                "authors": "MedSSL-BHSD Benchmark",
                "venue": "Baseline",
                "adaptation": "Chỉ sử dụng tập mẫu có nhãn (Subset(db_train, labeled_idxs)); toàn bộ batch_size đều lấy từ labeled pool (unlabeled_bs = 0).",
            },
            "networks": [
                {"id": "sup_net", "name": "Single Trainable Network (self.model)", "role": "Standard Supervised Model", "mode": "train()", "grad": True, "update_rule": "SGD Optimizer", "color": "blue"},
            ],
            "nodes": [
                {"id": "s_data", "title": "Labeled-Only DataLoader", "subtitle": "Subset(db_train, labeled_idxs)", "category": "data", "x": 80, "y": 195, "w": 235, "h": 95, "code_ref": "supervised_train_step", "shape_key": "batch_input", "grad_state": "no_grad", "summary": "Trong resolve_batch_layout(), khi method=='supervised', toàn bộ batch_size đều là mẫu Labeled (unlabeled = 0).", "details": ["Không dùng TwoStreamBatchSampler; dùng DistributedSampler hoặc shuffle tiêu chuẩn trên labeled_dataset."]},
                {"id": "s_net", "title": "Backbone Network (self.model)", "subtitle": "logits = extract_logits(model(X))", "category": "model_trainable", "x": 430, "y": 195, "w": 240, "h": 95, "code_ref": "supervised_train_step", "shape_key": "student_outputs", "grad_state": "requires_grad", "summary": "Chạy forward trực tiếp và trích xuất logits chính.", "details": ["Lưu ý: extract_logits() chỉ lấy đầu ra chính 'seg', không cộng vòng lặp Deep Supervision phụ trong SupervisedMethod.train_step."]},
                {"id": "s_loss", "title": "Supervised CE + Dice Loss", "subtitle": "0.5 * CE + 0.5 * MaskDiceLoss", "category": "loss_sup", "x": 790, "y": 195, "w": 240, "h": 95, "code_ref": "supervised_train_step", "shape_key": "sup_loss_tensors", "grad_state": "requires_grad", "summary": "Tính trung bình cộng giữa Cross-Entropy và MaskDiceLoss rồi cập nhật qua GradientHandler.", "details": ["Đóng vai trò cận dưới (Lower Bound) để đo lường lợi ích thực sự của dữ liệu không nhãn trong Mean Teacher, CPS, BCP, PICK."]},
            ],
            "edges": [
                {"id": "s1", "from": "s_data", "to": "s_net", "label": "volume_batch (All Labeled)", "kind": "data"},
                {"id": "s2", "from": "s_net", "to": "s_loss", "label": "logits", "kind": "forward_grad"},
                {"id": "s3", "from": "s_data", "to": "s_loss", "label": "label_batch", "kind": "label"},
            ],
            "execution_steps": [
                {"step": 1, "title": "Labeled Batch Forward & CE+Dice Optimization", "code_line": "methods/supervised.py:26-41", "active_nodes": ["s_data", "s_net", "s_loss"], "active_edges": ["s1", "s2", "s3"], "what_happens": "Lấy toàn bộ mẫu có nhãn, chạy forward, tính (CE + Dice)/2 và cập nhật trọng số.", "key_formula": "\\mathcal{L}_{\\text{total}} = \\frac{1}{2}\\mathcal{L}_{\\text{CE}}(Z, Y) + \\frac{1}{2}\\mathcal{L}_{\\text{Dice}}(Z, Y)"},
            ],
            "paper_vs_impl": [],
        },
    }

