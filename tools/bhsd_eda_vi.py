"""Nhãn tiếng Việt cho phần trình bày EDA; không đổi khóa hay giá trị dữ liệu gốc."""
import pandas as pd

LABELS = {
'split':'Tập dữ liệu', 'cases':'Số ca', 'case_id':'Mã ca', 'is_labeled':'Có nhãn',
'profiled_cases':'Số ca đã thống kê', 'value':'Giá trị', 'version':'Phiên bản',
'train':'Huấn luyện', 'val':'Xác thực', 'test':'Kiểm thử',
'train_labeled':'Huấn luyện có nhãn', 'train_unlabeled':'Huấn luyện không nhãn',
'metadata_errors':'Lỗi siêu dữ liệu', 'nonfinite_voxels':'Số voxel NaN/Inf',
'raw_status':'Trạng thái dữ liệu gốc', 'metadata_consistent':'Siêu dữ liệu nhất quán',
'geometry_mismatch':'Hình học không khớp', 'missing_raw_pair':'Thiếu cặp ảnh gốc',
'label_values':'Giá trị nhãn', 'crop_reduction_pct':'Thể tích giảm do cắt (%)',
'lesion_ml':'Thể tích tổn thương (mL)', 'min':'Nhỏ nhất', 'median':'Trung vị',
'max':'Lớn nhất', 'mean':'Trung bình', 'count':'Số lượng', 'sum':'Tổng',
'patch':'Kích thước khối ảnh', 'physical_extent_mm':'Phạm vi vật lý (mm)',
'cases_needing_padding_pct':'Ca cần đệm (%)', 'mean_added_volume_fraction':'Tỷ phần đệm trung bình trên thể tích đã đệm',
'patch_voxels':'Số voxel mỗi khối', 'class':'Lớp tổn thương', 'class_name':'Lớp tổn thương',
'class_id':'Mã lớp', 'positive_cases':'Số ca có lớp', 'prevalence_pct':'Tỷ lệ hiện diện (%)',
'pooled_voxel_share_pct':'Tỷ lệ voxel gộp (%)', 'median_ml_when_present':'Thể tích trung vị khi có lớp (mL)',
'min_ml_when_present':'Thể tích nhỏ nhất khi có lớp (mL)', 'cases_below_1ml':'Số ca dưới 1 mL',
'ml':'Thể tích (mL)', 'num_lesion_types':'Số loại tổn thương',
'clip_min':'Ngưỡng HU dưới', 'clip_max':'Ngưỡng HU trên', 'norm_mean':'Trung bình chuẩn hóa',
'norm_std':'Độ lệch chuẩn dùng chuẩn hóa', 'feature':'Đặc trưng',
'n_labeled_train':'Số ca huấn luyện có nhãn', 'n_unlabeled':'Số ca không nhãn',
'KS_D':'Thống kê D của KS', 'p':'Giá trị p', 'p_holm':'p hiệu chỉnh Holm',
'difference_detected_at_0.05':'Phát hiện khác biệt ở mức 0,05',
'img_mean':'Cường độ trung bình', 'img_std':'Độ lệch chuẩn cường độ',
'positive_raw_cases':'Số ca có lớp trong GT gốc', 'cropped_out_voxels':'Số voxel bị cắt mất',
'disappeared_cases':'Số ca mất lớp', 'median_volume_change_pct':'Thay đổi thể tích trung vị (%)',
'min_diagnostic_dice':'Dice kiểm tra nhỏ nhất', 'median_diagnostic_dice':'Dice kiểm tra trung vị',
'raw_voxels':'Số voxel GT gốc', 'cropped_voxels':'Số voxel sau cắt', 'h5_voxels':'Số voxel H5',
'disappeared':'Lớp biến mất', 'appeared':'Lớp xuất hiện mới', 'raw_ml':'Thể tích gốc theo spacing (mL)',
'h5_ml':'Thể tích H5 (mL)', 'raw_affine_ml':'Thể tích gốc theo affine (mL)',
'volume_change_pct':'Thay đổi thể tích (%)', 'diagnostic_forward_dice':'Dice kiểm tra lấy mẫu thuận',
'foreground_prob':'Xác suất chọn tâm theo tổn thương', 'any_hit_rate':'Tỷ lệ gặp tổn thương',
'mean_case_hit_rate':'Tỷ lệ gặp tổn thương trung bình theo ca',
'median_case_hit_rate':'Tỷ lệ gặp tổn thương trung vị theo ca',
'min_case_hit_rate':'Tỷ lệ gặp tổn thương nhỏ nhất theo ca',
'mean_patch_fg_fraction':'Tỷ phần tổn thương trung bình trong khối',
'case_bootstrap_low_95':'Cận dưới bootstrap 95%', 'case_bootstrap_high_95':'Cận trên bootstrap 95%',
'mean_conditional_hit_rate':'Tỷ lệ gặp lớp trung bình khi có lớp', 'burden':'Nhóm thể tích tổn thương',
'case_class_unions':'Số hợp vùng ca–lớp', 'pca_defined':'Số mặt nạ có PCA xác định',
'median_sphericity':'Độ cầu trung vị', 'median_savr_cm_inverse':'SAVR trung vị (cm⁻¹)',
'median_flatness':'Độ phẳng trung vị', 'median_components':'Số thành phần trung vị',
'voxels':'Số voxel', 'vol_ml':'Thể tích (mL)', 'surface_area_cm2':'Diện tích bề mặt (cm²)',
'sphericity':'Độ cầu', 'savr':'SAVR', 'savr_unit':'Đơn vị SAVR', 'elongation':'Độ kéo dài',
'flatness':'Độ phẳng', 'extent':'Tỷ phần lấp đầy hộp bao', 'components_26':'Số thành phần liên thông 26 lân cận',
'dataset':'Đường dẫn dữ liệu', 'raw_root':'Đường dẫn dữ liệu gốc', 'module_sha256':'SHA256 mã xử lý',
'labeled':'Có nhãn', 'unlabeled':'Không nhãn', 'split_sha256':'SHA256 danh sách phân hoạch', 'scope':'Phạm vi',
'All H5 volumes; raw GT and morphology for labeled cases only':'Toàn bộ H5; GT gốc và hình thái chỉ cho ca có nhãn',
'Proportion':'Tỷ lệ tích lũy', 'Density':'Mật độ',
}
for axis in 'zyx':
    for prefix,label,unit in [('shape','Kích thước','voxel'),('orig_shape','Kích thước gốc','voxel'),
                              ('orig_spacing','Khoảng cách lấy mẫu gốc','mm'),('target_spacing','Khoảng cách lấy mẫu đích','mm')]:
        LABELS[f'{prefix}_{axis}'] = f'{label} {axis.upper()} ({unit})'
for c,name in enumerate(['EDH','IPH','IVH','SAH','SDH'],1):
    LABELS[f'c{c}_ml'] = name


def vi(value):
    if isinstance(value, tuple):
        return tuple(vi(v) for v in value)
    if isinstance(value, bool):
        return 'Có' if value else 'Không'
    return LABELS.get(value, value) if isinstance(value, str) else value


def vietnamese_table(obj):
    """Dịch bản sao để giữ nguyên dữ liệu xuất CSV và các phép tính tiếp theo."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if not isinstance(obj, pd.DataFrame):
        return obj
    result = obj.copy()
    for c in result.columns:
        if result[c].dtype == object or result[c].dtype == bool:
            result[c] = result[c].map(vi)
    result.columns = result.columns.map(vi)
    result.index = result.index.map(vi)
    result.index.names = [vi(n) for n in result.index.names]
    result.columns.names = [vi(n) for n in result.columns.names]
    return result


def translate_figure(fig):
    """Dịch nhãn tự sinh của seaborn mà không đổi các cột dùng để vẽ."""
    for ax in fig.axes:
        ax.set_xlabel(vi(ax.get_xlabel()))
        ax.set_ylabel(vi(ax.get_ylabel()))
        title = ax.get_title()
        for key in ['shape_z','shape_y','shape_x','sphericity','flatness']:
            title = title.replace(key,LABELS[key])
        ax.set_title(title)
        legend = ax.get_legend()
        if legend:
            legend.set_title(vi(legend.get_title().get_text()))
            for text in legend.get_texts():
                text.set_text(vi(text.get_text()))
        # Chỉ đổi nhãn phân loại; giữ nguyên định dạng số tự động.
        labels = [t.get_text() for t in ax.get_xticklabels()]
        if any(vi(t) != t for t in labels):
            ax.set_xticks(ax.get_xticks(), [vi(t) for t in labels])
