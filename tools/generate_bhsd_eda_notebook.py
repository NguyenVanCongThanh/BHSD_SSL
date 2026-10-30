#!/usr/bin/env python3
"""Build and optionally execute the reproducible BHSD fold0 EDA notebook.

Usage: python tools/generate_bhsd_eda_notebook.py [--no-execute]
Dataset paths, sampling trials and profiling workers can be overridden through
BHSD_DATASET_DIR, BHSD_RAW_DIR, BHSD_EDA_TRIALS and BHSD_EDA_WORKERS.
"""
import argparse
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def create_eda_notebook(execute=True):
    nb = new_notebook(metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    def md(s):
        nb.cells.append(new_markdown_cell(s.strip()))
    def code(s):
        nb.cells.append(new_code_cell(s.strip()))

    md('''# BHSD_fold0: phân tích khám phá dữ liệu có thể tái lập

Notebook kiểm tra toàn bộ thể tích H5 trong phân hoạch fold0 hiện tại, thống kê riêng tập huấn luyện có nhãn và tập xác thực, đồng thời so sánh đặc trưng ảnh giữa nhóm có nhãn và không nhãn. Mặt nạ của dữ liệu không nhãn chỉ là dữ liệu giữ chỗ, **không phải nhãn chuẩn âm tính**.

Mọi bảng được tạo lại từ H5 nguồn bằng `tools/bhsd_eda.py`. Bộ nhớ đệm của từng ca được kiểm tra theo kích thước, thời điểm sửa tệp nguồn và mã băm của mã xử lý. Tệp kê khai lưu đường dẫn cùng mã băm mã nguồn và phân hoạch. Chỉ các tệp kết quả được ghi; ảnh nguồn, nhãn và cấu hình huấn luyện được giữ nguyên.

**Diễn giải:** kết quả mô tả bộ dữ liệu và quy trình tiền xử lý này. EDA chưa đủ để xác lập giá trị lâm sàng, tính tương đương của máy chụp, kiến trúc tối ưu hay hàm mất mát/phương pháp lấy mẫu bắt buộc. Danh sách xác thực và kiểm thử có thể trùng nhau; mã ca không giao nhau chưa chứng minh các tập độc lập ở cấp bệnh nhân.

**Thuật ngữ:** GT là nhãn chuẩn; voxel là phần tử thể tích; patch là khối ảnh con; spacing là khoảng cách lấy mẫu theo trục; foreground là vùng tổn thương. EDH: xuất huyết ngoài màng cứng; IPH: xuất huyết trong nhu mô; IVH: xuất huyết trong não thất; SAH: xuất huyết dưới nhện; SDH: xuất huyết dưới màng cứng. Tên biến, khóa dữ liệu và viết tắt kỹ thuật được giữ nguyên để bảo đảm khả năng tái lập.''')
    code('''import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, ndimage
from IPython.display import display as original_display

# Chạy từ gốc kho mã hoặc notebooks/; dùng đường dẫn CSV rõ ràng.
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'tools/bhsd_eda.py').exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.bhsd_eda import CLASS_NAMES, split_audit, build_tables, parse, sample_case, fingerprint
from tools.bhsd_eda_vi import vi, vietnamese_table, translate_figure

def display(obj):
    original_display(vietnamese_table(obj))

# Dịch nhãn tự sinh trước khi hiển thị; giữ nguyên dữ liệu tính toán.
original_show = getattr(plt, '_bhsd_original_show', plt.show)
plt._bhsd_original_show = original_show
def show_vi(*args, **kwargs):
    for number in plt.get_fignums():
        translate_figure(plt.figure(number))
    return original_show(*args, **kwargs)
plt.show = show_vi

DATASET_DIR = Path(os.environ.get('BHSD_DATASET_DIR', str(Path.home()/'dataset/BHSD_fold0'))).expanduser()
RAW_DIR = Path(os.environ.get('BHSD_RAW_DIR', str(Path.home()/'dataset/BHSD/label_192/label_192'))).expanduser()
OUT = ROOT/'notebooks'
SEED = 42
TRIALS = int(os.environ.get('BHSD_EDA_TRIALS', '40'))
sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 110, 'figure.figsize': (10, 4)})
print('Bộ dữ liệu:', DATASET_DIR, '\\nDữ liệu gốc:', RAW_DIR, '\\nKết quả:', OUT)
print('Hạt giống lấy mẫu:', SEED, '| số lần thử mỗi ca/cấu hình:', TRIALS)
from importlib.metadata import version
versions = {package: version(package) for package in ['numpy','scipy','pandas','h5py','nibabel','scikit-image','torch','matplotlib','seaborn','nbformat','nbclient']}
(OUT/'bhsd_eda_environment.json').write_text(json.dumps(versions,indent=2))
display(pd.Series(versions,name='version').to_frame())
''')
    md('''## 1. Nguồn dữ liệu và tính toàn vẹn của phân hoạch

Các kiểm tra sẽ báo lỗi nếu mã ca bị lặp, tập huấn luyện giao với tập xác thực, nhóm có nhãn giao với nhóm không nhãn, thiếu H5 hoặc H5 chưa được phân nhóm, hay hợp hai nhóm huấn luyện không đúng. Phạm vi ở đây là các danh sách fold0 hiện hành tại thư mục gốc dữ liệu; chưa kiểm tra toàn bộ năm fold.''')
    code('''splits, available = split_audit(DATASET_DIR)
# Thu thông báo của bộ trích xuất để trình bày tiến độ bằng tiếng Việt.
import io
from contextlib import redirect_stdout
progress = io.StringIO()
with redirect_stdout(progress):
    df, shape, retention = build_tables(DATASET_DIR, RAW_DIR, OUT,
                                       workers=int(os.environ.get('BHSD_EDA_WORKERS', '2')))
print(f'Đã thống kê {len(df)}/{len(available)} ca H5.')
assert len(df) == len(available) and df.case_id.nunique() == len(available)
assert set(df.case_id) == available
# Đồng bộ bản CSV cũ tại gốc; kết quả chính nằm trong notebooks/.
for name, table in [('bhsd_eda_summary.csv',df),('bhsd_shape_morphology.csv',shape)]:
    if (ROOT/name).exists():
        table.to_csv(ROOT/name,index=False)
labeled = df[df.is_labeled].copy()
unlabeled = df[~df.is_labeled].copy()
train = df[df.split == 'train_labeled'].copy()
val = df[df.split == 'val'].copy()
display(pd.DataFrame([{'split': k, 'cases': len(v)} for k,v in splits.items()]))
display(df.groupby(['split', 'is_labeled']).size().rename('profiled_cases').reset_index())
print('Mã ca xác thực trùng mã ca kiểm thử:', vi(set(splits['val']) == set(splits['test'])))
print('Đã kiểm tra mã ca không giao nhau. Chưa xác minh độc lập cấp bệnh nhân do thiếu siêu dữ liệu bệnh nhân.')
if set(splits['val']) == set(splits['test']):
    print('Danh sách kiểm thử trùng tập xác thực; không có tập kiểm thử độc lập.')
display(pd.read_json(OUT/'bhsd_eda_manifest.json', typ='series').to_frame('value'))
''')
    md('''## 2. Chất lượng dữ liệu và ngoại lệ hình học

Toàn bộ ảnh được đọc theo từng khối lát cắt để kiểm tra NaN/Inf và thống kê cường độ. Mỗi ca đều được kiểm tra miền giá trị nhãn, sự nhất quán giữa kích thước và siêu dữ liệu, giới hạn hộp bao (bbox), cùng cờ có nhãn so với phân hoạch. Mặt nạ toàn số 0 của dữ liệu không nhãn không được tính vào thống kê tổn thương.

Cắt theo hộp bao loại bỏ phần ngoại vi hình hộp; **không đồng nghĩa với loại bỏ hộp sọ**. Số voxel được tính trên toàn bộ thể tích sau cắt, không phải riêng mô não đã phân đoạn. Khoảng cách lấy mẫu không nhất thiết bằng độ dày lát cắt khi chụp.''')
    code('''issues = df[df.metadata_errors.fillna('').ne('') | df.nonfinite_voxels.gt(0)]
display(issues[['case_id','split','metadata_errors','nonfinite_voxels']])
print('Số ca H5 có vấn đề:', len(issues))
print('Trạng thái cặp ảnh gốc (chỉ ca có nhãn):')
display(labeled.raw_status.value_counts().to_frame('cases'))
display(labeled.loc[labeled.raw_status.ne('metadata_consistent'), ['case_id','split','raw_status']])
print('Ca có nhãn nhưng không có tổn thương (cần kiểm tra nguồn/chú giải):')
display(labeled.loc[labeled.lesion_voxels.eq(0), ['case_id','split','label_values']])
features = ['shape_z','shape_y','shape_x','orig_spacing_z','orig_spacing_y','orig_spacing_x','crop_reduction_pct']
display(df.groupby('split')[features].agg(['min','median','max']))
outliers = labeled[(labeled.shape_z < 16) | (labeled.orig_spacing_z < 2) | (labeled.orig_spacing_z > 7)]
display(outliers[['case_id','split','orig_shape_z','orig_spacing_z','shape_z','target_spacing_z','lesion_ml']])
print('Số ca có tổn thương chạm biên H5:', int(labeled.lesion_touches_h5_boundary.sum()),
      '(cần rà soát; chưa chứng minh tổn thương bị cắt)')
''')
    code('''fig, axes = plt.subplots(1, 3, figsize=(15,4))
for ax, feature in zip(axes, ['shape_z','shape_y','shape_x']):
    sns.ecdfplot(data=df, x=feature, hue='split', ax=ax)
    ax.set_title(feature + ' — toàn bộ ca')
plt.tight_layout(); plt.show()
# Đo mức đệm; không suy ra Z tối ưu chỉ từ trung vị.
padding_rows = []
for z,y,x in [(32,160,160),(32,192,192),(32,224,224),(48,160,160)]:
    dims = df[['shape_z','shape_y','shape_x']].to_numpy()
    padded = np.maximum(dims, [z,y,x])
    fraction = 1-np.prod(dims,axis=1)/np.prod(padded,axis=1)
    padding_rows.append({'patch': f'{z}x{y}x{x}', 'physical_extent_mm': str((z*5,y*.5,x*.5)),
        'cases_needing_padding_pct': 100*np.mean(np.any(dims < [z,y,x],axis=1)),
        'mean_added_volume_fraction': fraction.mean(), 'patch_voxels': z*y*x})
display(pd.DataFrame(padding_rows))
print('Các tỷ lệ mô tả đệm trên toàn thể tích, chưa đo VRAM hoặc phần đệm trong từng khối ảnh.')
''')
    md('''## 3. Cân bằng các loại tổn thương giữa huấn luyện và xác thực

Một ca có thể chứa nhiều lớp tổn thương. Tỷ lệ hiện diện được tính theo số ca; thống kê thể tích chỉ tính các ca có lớp tương ứng. Tỷ lệ tổn thương gộp dùng toàn bộ voxel của thể tích có nhãn sau cắt làm mẫu số. Khác biệt giữa các tập mang tính mô tả, không nên dùng để liên tục điều chỉnh tiền xử lý theo nhãn xác thực.''')
    code('''class_rows = []
for split_name, group in labeled.groupby('split'):
    for c, name in enumerate(CLASS_NAMES, 1):
        present = group[group[f'c{c}_voxels'] > 0]
        class_rows.append({'split': split_name, 'class': name, 'cases': len(group),
            'positive_cases': len(present), 'prevalence_pct': 100*len(present)/len(group),
            'pooled_voxel_share_pct': 100*group[f'c{c}_voxels'].sum()/group.total_voxels.sum(),
            'median_ml_when_present': present[f'c{c}_ml'].median(),
            'min_ml_when_present': present[f'c{c}_ml'].min(),
            'cases_below_1ml': int((present[f'c{c}_ml'] < 1).sum())})
class_table = pd.DataFrame(class_rows)
display(class_table)
class_table.to_csv(OUT/'bhsd_split_class_balance.csv', index=False)
ratio = labeled.lesion_voxels.sum()/labeled.total_voxels.sum()
print(f'Tỷ lệ tổn thương gộp trên dữ liệu có nhãn: {100*ratio:.4f}% voxel của thể tích sau cắt')
print(f'Độ chính xác voxel khi dự đoán toàn bộ là nền: {100*(1-ratio):.4f}%; chỉ số này không hữu ích để đánh giá phân đoạn.')
fig, axes = plt.subplots(1,2, figsize=(14,4))
sns.barplot(data=class_table, x='class', y='prevalence_pct', hue='split', ax=axes[0])
volumes = labeled.melt(id_vars=['case_id','split'], value_vars=[f'c{c}_ml' for c in range(1,6)], value_name='ml', var_name='class')
sns.boxplot(data=volumes[volumes.ml > 0], x='class', y='ml', hue='split', ax=axes[1])
axes[1].set_yscale('log'); axes[1].set_title('Thể tích lớp khi hiện diện')
plt.tight_layout(); plt.show()
presence = labeled[[f'c{c}_voxels' for c in range(1,6)]].gt(0).astype(int)
cooccurrence = presence.T @ presence
cooccurrence.index = cooccurrence.columns = CLASS_NAMES
sns.heatmap(cooccurrence, annot=True, fmt='d', cmap='YlOrRd'); plt.title('Đồng xuất hiện theo ca (toàn bộ dữ liệu có nhãn)'); plt.show()
display(labeled.groupby(['split','num_lesion_types']).size().rename('cases').reset_index())
''')
    md('''## 4. Biến đổi cường độ và so sánh các nhóm dữ liệu

Giới hạn cửa sổ HU làm bão hòa giá trị ngoài khoảng, không loại bỏ xương hay không khí. Chuẩn hóa toàn cục không bảo đảm mỗi ảnh có trung bình 0 và phương sai 1; `image > image.min()` không phải mặt nạ não đã được kiểm chứng. Các hằng số dưới đây lấy từ thuộc tính H5, kèm số ca sử dụng từng tổ hợp tham số.

Kiểm định KS so sánh **tập huấn luyện có nhãn** với toàn bộ tập không nhãn, loại tập xác thực khỏi phép so sánh. Báo cáo thống kê D, giá trị p gốc và p hiệu chỉnh Holm trên các đặc trưng được kiểm định. Giá trị p lớn không chứng minh tính tương đương; các đặc trưng này chưa đủ để loại trừ khác biệt về máy chụp, quy trình chụp, bệnh lý hoặc cách chọn mẫu. Đặc trưng cường độ được đo sau tiền xử lý.''')
    code('''display(df.groupby(['clip_min','clip_max','norm_mean','norm_std']).size().rename('cases').reset_index())
case_id = train.sort_values('case_id').iloc[0].case_id
raw_image = nib.load(RAW_DIR/'images'/f'{case_id}.nii.gz')
raw_hu = np.asarray(raw_image.dataobj, dtype=np.float32)
with h5py.File(DATASET_DIR/'data'/f'{case_id}.h5') as f:
    image = f['image'][:]
    attrs = dict(f.attrs)
clip, norm = parse(attrs['clip_range']), parse(attrs['norm_params'])
fig, axes = plt.subplots(1,2, figsize=(13,4))
axes[0].hist(raw_hu.ravel(), bins=100, range=(-150,200), density=True)
for value in clip: axes[0].axvline(value, color='red', linestyle='--')
axes[0].set_title('HU gốc (biểu đồ trong khoảng -150 đến 200)')
axes[1].hist(image.ravel(), bins=80, density=True)
axes[1].set_title('Toàn bộ voxel H5; không suy diễn mặt nạ não')
plt.tight_layout(); plt.show()
print('Phép biến đổi đã ghi:', f'(clip(HU, {clip}) - {norm["mean"]}) / {norm["std"]}')
print('Giới hạn lý thuyết sau cắt ngưỡng và chuẩn hóa:', [(v-norm['mean'])/norm['std'] for v in clip])
features = ['shape_z','shape_y','shape_x','orig_spacing_z','orig_spacing_y','orig_spacing_x','img_mean','img_std']
ks_rows = []
for feature in features:
    a,b = train[feature].dropna(), unlabeled[feature].dropna()
    result = stats.ks_2samp(a,b)
    ks_rows.append({'feature': feature, 'n_labeled_train': len(a), 'n_unlabeled': len(b),
                    'KS_D': result.statistic, 'p': result.pvalue})
ks = pd.DataFrame(ks_rows)
order = np.argsort(ks.p.to_numpy())
adjusted = np.minimum(1, np.maximum.accumulate(ks.p.to_numpy()[order]*(len(ks)-np.arange(len(ks)))))
ks.loc[order,'p_holm'] = adjusted
ks['difference_detected_at_0.05'] = ks.p_holm < .05
display(ks)
ks.to_csv(OUT/'bhsd_cohort_comparison.csv', index=False)
print('Đặc trưng phát hiện khác biệt:', [vi(f) for f in ks.loc[ks['difference_detected_at_0.05'],'feature']])
print('Chưa thể kết luận không có dịch chuyển phân phối hoặc máy chụp tương đương.')
''')
    md('''## 5. Kiểm tra định lượng bảo toàn nhãn: GT gốc → cắt → H5

NIfTI gốc được đọc theo thứ tự X,Y,Z và chuyển sang Z,Y,X của H5. Ảnh gốc và GT phải có cùng ma trận affine, cùng kích thước, đồng thời khớp với kích thước và khoảng cách lấy mẫu gốc ghi trong H5. Đây là kiểm tra siêu dữ liệu, chưa phải bằng chứng độc lập về độ chính xác của đăng ký ảnh.

Số voxel bị cắt mất được đếm trực tiếp trên GT gốc trước khi lấy mẫu lại. Thay đổi thể tích dùng tích các khoảng cách lấy mẫu đã ghi; `raw_affine_ml` dùng riêng định thức ma trận affine để xem xét hình học nghiêng/xiên. Hai quy ước này có thể cho kết quả khác nhau.

`diagnostic_forward_dice` dùng nội suy láng giềng gần nhất của `scipy.ndimage.zoom` với quy ước căn theo điểm đầu/cuối mặc định trên nhãn gốc đã cắt. H5 không ghi thư viện lấy mẫu lại, quy ước tọa độ và phép nội suy nhãn ban đầu, nên Dice này chỉ là **ước lượng phục vụ kiểm tra**, không xác nhận tiền xử lý đúng. Lớp biến mất, thay đổi thể tích lớn và tổn thương chạm biên là các dấu hiệu cần rà soát.''')
    code('''print('Số cặp ca–lớp gốc/H5 đã kiểm tra:', len(retention))
positive_raw = retention[retention.raw_voxels > 0]
display(positive_raw.groupby('class_name').agg(
    positive_raw_cases=('case_id','size'), cropped_out_voxels=('cropped_out_voxels','sum'),
    disappeared_cases=('disappeared','sum'), median_volume_change_pct=('volume_change_pct','median'),
    min_diagnostic_dice=('diagnostic_forward_dice','min'), median_diagnostic_dice=('diagnostic_forward_dice','median')))
flags = retention[(retention.cropped_out_voxels > 0) | retention.disappeared | retention.appeared |
                  (retention.volume_change_pct.abs() > 20)]
display(flags.sort_values('volume_change_pct', key=lambda s: s.abs(), ascending=False))
flags.to_csv(OUT/'bhsd_preprocessing_review_cases.csv', index=False)
print('Ngoại lệ hình học nguồn cần rà soát:')
display(outliers[['case_id','orig_shape_z','orig_spacing_z','shape_z','lesion_ml']])
print('Cần rà soát siêu dữ liệu thu nhận và nguồn gốc tiền xử lý; không tự động sửa ảnh/nhãn nguồn.')
''')
    md('''### Kiểm tra đa mặt phẳng trên cùng lưới gốc

H5 được lấy mẫu ngược xấp xỉ về kích thước sau cắt đã ghi, đặt lại theo độ lệch hộp bao, rồi so sánh tại cùng chỉ số voxel trên lưới gốc. Chỉ số voxel được ánh xạ sang tọa độ vật lý qua affine của ảnh gốc; ảnh và nhãn phải dùng cùng affine. Các hình thể hiện mặt phẳng thu nhận gốc, có thể nghiêng, chứ chưa phải tái tạo theo mặt phẳng giải phẫu chuẩn. Tỷ lệ hiển thị vật lý dùng độ dài các cột affine.

Cách này thay cho phép so sánh trước đây dùng tỷ lệ chỉ số lát cắt và các điểm giữa không tương ứng. Phép khôi phục giả định căn theo điểm đầu/cuối; chưa thể xác nhận đường biên khớp tuyệt đối.''')
    code('''# Chọn ca huấn luyện có nhiều lớp để minh họa.
selected = train.sort_values(['num_lesion_types','case_id'], ascending=[False,True]).iloc[0].case_id
raw_img_n = nib.load(RAW_DIR/'images'/f'{selected}.nii.gz')
raw_gt_n = nib.load(RAW_DIR/'ground truths'/f'{selected}.nii.gz')
assert raw_img_n.shape == raw_gt_n.shape and np.allclose(raw_img_n.affine, raw_gt_n.affine)
raw_img = np.asarray(raw_img_n.dataobj, dtype=np.float32).transpose(2,1,0)
raw_gt = np.asarray(raw_gt_n.dataobj).transpose(2,1,0)
with h5py.File(DATASET_DIR/'data'/f'{selected}.h5') as f:
    h5_img, h5_gt = f['image'][:], f['label'][:]
    attrs = dict(f.attrs)
bbox, cropped = np.array(parse(attrs['bbox_used'])), np.array(parse(attrs['cropped_shape']))
assert tuple(parse(attrs['orig_shape'])) == raw_img.shape
inverse_img = ndimage.zoom(h5_img, cropped/np.array(h5_img.shape), order=1, prefilter=False)
inverse_gt = ndimage.zoom(h5_gt, cropped/np.array(h5_gt.shape), order=0, prefilter=False)
restored_img = np.full(raw_img.shape, np.nan, dtype=np.float32)
restored_gt = np.zeros_like(raw_gt)
region = tuple(slice(a,b) for a,b in bbox)
restored_img[region], restored_gt[region] = inverse_img, inverse_gt
z = int(np.argmax((raw_gt > 0).sum(axis=(1,2))))
coords = np.argwhere(raw_gt[z] > 0)
y,x = np.median(coords, axis=0).astype(int) if len(coords) else np.array(raw_img.shape[1:])//2
world = nib.affines.apply_affine(raw_img_n.affine, [x,y,z])
spacing_zyx = np.linalg.norm(raw_img_n.affine[:3,:3],axis=0)[::-1]
planes = [(lambda a: a[z], spacing_zyx[1]/spacing_zyx[2], 'XY gốc'),
          (lambda a: a[:,y,:], spacing_zyx[0]/spacing_zyx[2], 'XZ gốc'),
          (lambda a: a[:,:,x], spacing_zyx[0]/spacing_zyx[1], 'YZ gốc')]
fig, axes = plt.subplots(2,3,figsize=(15,10))
for j,(plane,aspect,title) in enumerate(planes):
    for i,(img,lbl,label) in enumerate([(raw_img,raw_gt,'HU gốc'),(restored_img,restored_gt,'H5 khôi phục')]):
        axes[i,j].imshow(plane(img), cmap='gray', aspect=aspect,
                         **({'vmin':0,'vmax':80} if i == 0 else {}))
        axes[i,j].imshow(np.ma.masked_where(plane(lbl)==0,plane(lbl)), cmap='tab10',vmin=0,vmax=5,alpha=.5,aspect=aspect)
        axes[i,j].set_title(f'{label}: {title}'); axes[i,j].axis('off')
plt.suptitle(f'{selected} | ZYX gốc={z,y,x} | tọa độ RAS (mm)={np.round(world,2)}')
plt.tight_layout(); plt.show()
''')
    md('''## 6. Lấy mẫu khối ảnh trên toàn bộ ca huấn luyện có nhãn

Sử dụng đúng `data.transforms3d.RandomCrop` của mã huấn luyện, gồm đệm số 0 đối xứng, dịch ngẫu nhiên tâm quanh vùng tổn thương và xử lý GT rỗng. Ảnh được thay bằng nhãn vì việc lấy mẫu chỉ phụ thuộc hình học và GT. Hạt giống ngẫu nhiên được cố định cho từng ca/cấu hình; kết quả là ước lượng Monte Carlo, không bảo đảm mô hình hội tụ.

So sánh ba kích thước XY và xác suất chọn tâm theo tổn thương 0; 0,33; 0,5. Báo cáo tỷ lệ gặp tổn thương trung bình với trọng số bằng nhau giữa các ca, tỷ lệ theo lớp có điều kiện, tỷ phần tổn thương trong khối ảnh và phân bố theo tổng thể tích tổn thương. Trung bình kèm khoảng bootstrap 95% theo ca (1.000 lần lấy mẫu lại; phụ thuộc mô phỏng hữu hạn này). Các ca không có một lớp được loại khỏi mẫu số khi tính tỷ lệ gặp lớp đó.

Đây là xác suất trên **khối ảnh**, không phải trên lô huấn luyện. Dữ liệu không nhãn không thể lấy mẫu dựa trên GT; lấy mẫu dựa trên nhãn giả cần một thí nghiệm riêng.''')
    code('''# Kiểm tra đệm theo nguồn, mã lấy mẫu/xử lý và cấu hình hạt giống/số lần thử.
import hashlib
sampling_path = OUT/'bhsd_patch_sampling.csv'
sampling_manifest = OUT/'bhsd_patch_sampling_manifest.json'
sampling_key = {'trials': TRIALS, 'seed': SEED,
    'patches': [[32,160,160],[32,192,192],[32,224,224]], 'foreground_probabilities':[0.,.33,.5],
    'sources': fingerprint([DATASET_DIR/'data'/f'{c}.h5' for c in sorted(train.case_id)]),
    'sampler_sha256': hashlib.sha256((ROOT/'data/transforms3d.py').read_bytes()).hexdigest()}
if sampling_path.exists() and sampling_manifest.exists() and json.loads(sampling_manifest.read_text()) == sampling_key:
    sampling = pd.read_csv(sampling_path)
else:
    rows = []
    for i, case_id in enumerate(sorted(train.case_id),1):
        rows.extend(sample_case(case_id, DATASET_DIR, trials=TRIALS, seed=SEED))
        if i % 25 == 0: print(f'Đã lấy mẫu {i}/{len(train)} ca huấn luyện', flush=True)
    sampling = pd.DataFrame(rows)
    sampling.to_csv(sampling_path,index=False)
    sampling_manifest.write_text(json.dumps(sampling_key,indent=2))
assert sampling.case_id.nunique() == len(train)
summary = sampling.groupby(['patch','foreground_prob']).agg(
    cases=('case_id','size'), mean_case_hit_rate=('any_hit_rate','mean'),
    median_case_hit_rate=('any_hit_rate','median'), min_case_hit_rate=('any_hit_rate','min'),
    mean_patch_fg_fraction=('fg_fraction','mean'))
# Bootstrap theo ca độc lập; các khối ảnh trong cùng ca có thể tương quan.
bootstrap_rng = np.random.default_rng(SEED)
for key, group in sampling.groupby(['patch','foreground_prob']):
    rates = group.any_hit_rate.to_numpy()
    means = bootstrap_rng.choice(rates, size=(1000,len(rates)), replace=True).mean(axis=1)
    summary.loc[key,'case_bootstrap_low_95'] = np.quantile(means,.025)
    summary.loc[key,'case_bootstrap_high_95'] = np.quantile(means,.975)
display(summary)
summary.to_csv(OUT/'bhsd_patch_sampling_summary.csv')
class_sampling = []
for (patch,prob), group in sampling.groupby(['patch','foreground_prob']):
    for c,name in enumerate(CLASS_NAMES,1):
        present = group[group[f'c{c}_present']]
        class_sampling.append({'patch':patch,'foreground_prob':prob,'class':name,
            'positive_cases':len(present),'mean_conditional_hit_rate':present[f'c{c}_hit_rate'].mean()})
class_sampling = pd.DataFrame(class_sampling)
display(class_sampling)
class_sampling.to_csv(OUT/'bhsd_patch_sampling_by_class.csv',index=False)
sampling['lesion_ml'] = sampling.case_id.map(train.set_index('case_id').lesion_ml)
sampling['burden'] = pd.cut(sampling.lesion_ml, [-np.inf,0,1,10,np.inf], labels=['Không có tổn thương','(0,1] ml','(1,10] ml','>10 ml'])
display(sampling.groupby(['patch','foreground_prob','burden'],observed=True).any_hit_rate.agg(['count','mean','min']))
fig, ax = plt.subplots(figsize=(12,5))
sns.boxplot(data=sampling,x='patch',y='any_hit_rate',hue='foreground_prob',ax=ax)
ax.set_ylabel('Xác suất khối ảnh có tổn thương theo ca'); plt.tight_layout(); plt.show()
print('Chọn xác suất lấy mẫu qua thí nghiệm huấn luyện; chưa có cơ sở bắt buộc dùng ngưỡng 0,5.')
''')
    md('''## 7. Hình thái tổn thương: đơn vị và phạm vi phân tích

Mỗi dòng biểu diễn **hợp vùng của một cặp ca–lớp**, bao gồm cả các thành phần rời nhau, không phải một tổn thương riêng lẻ. `components_26` đếm số thành phần liên thông 26 lân cận; chưa phân tích hình dạng từng thành phần. Diện tích bề mặt được tính bằng marching cubes trên mặt nạ đã đệm theo đơn vị mm vật lý; thể tích bằng số voxel nhân tích khoảng cách lấy mẫu. SAVR là A/V theo **cm⁻¹**: `surface_area_cm2 / vol_ml`. Độ cầu không có đơn vị.

Giữ lại mọi mặt nạ dương tính, kể cả chỉ có một voxel. Chỉ số PCA không xác định trên mặt nạ suy biến được ghi NaN. Các chỉ số tự định nghĩa là `elongation = 1-sqrt(lambda2/lambda1)` và `flatness = 1-sqrt(lambda3/lambda2)`, với lambda là trị riêng của ma trận hiệp phương sai tọa độ vật lý; tên gọi không hàm ý đây là định nghĩa radiomics chuẩn.

Lấy mẫu lại bất đẳng hướng, rời rạc hóa voxel, mặt nạ nhỏ và hợp các vùng rời nhau ảnh hưởng đến ước lượng. Đặc biệt, thể tích đếm theo voxel kết hợp bề mặt tam giác hóa có thể cho độ cầu > 1 ở mặt nạ rất nhỏ. Các sai lệch số này được đánh dấu, không diễn giải như hình dạng vật lý thực. Thống kê hình thái mang tính mô tả, chưa phải chẩn đoán hay bằng chứng bắt buộc dùng một hàm mất mát/kiến trúc.''')
    code('''expected_rows = int(sum((labeled[f'c{c}_voxels'] > 0).sum() for c in range(1,6)))
assert len(shape) == expected_rows and not shape.duplicated(['case_id','class_id']).any()
assert np.allclose(shape.savr,shape.surface_area_cm2/shape.vol_ml)
assert np.isfinite(shape.sphericity).all() and np.isfinite(shape.savr).all()
print('Sai lệch rời rạc hóa: độ cầu > 1 (giữ nguyên giá trị):')
display(shape[shape.sphericity > 1][['case_id','class_name','voxels','sphericity']])
shape['split'] = shape.case_id.map(labeled.set_index('case_id').split)
display(shape.groupby(['split','class_name']).agg(
    case_class_unions=('case_id','size'), pca_defined=('pca_defined','sum'),
    median_sphericity=('sphericity','median'), median_savr_cm_inverse=('savr','median'),
    median_flatness=('flatness','median'), median_components=('components_26','median')))
print('Mặt nạ rất nhỏ (được giữ lại đầy đủ):')
display(shape[shape.voxels < 10])
fig, axes = plt.subplots(1,3,figsize=(15,4))
for ax,metric in zip(axes,['sphericity','savr','flatness']):
    sns.boxplot(data=shape,x='class_name',y=metric,order=CLASS_NAMES,ax=ax)
    ax.set_title(metric + (' (cm⁻¹)' if metric == 'savr' else ''))
plt.tight_layout(); plt.show()
''')
    md('''### Minh họa từng lớp từ tập huấn luyện

Với mỗi lớp, chọn ca huấn luyện dương tính có thể tích lớp gần trung vị nhất. Hiển thị lát cắt ngang có diện tích lớp lớn nhất. Quy tắc chọn dựa trên thể tích và có thể tái lập; các ví dụ không được khẳng định là hình thái bệnh lý điển hình.''')
    code('''fig, axes = plt.subplots(1,5,figsize=(20,5))
for c,(name,ax) in enumerate(zip(CLASS_NAMES,axes),1):
    positives = train[train[f'c{c}_ml'] > 0].copy()
    positives['distance'] = (positives[f'c{c}_ml']-positives[f'c{c}_ml'].median()).abs()
    chosen = positives.sort_values(['distance','case_id']).iloc[0]
    with h5py.File(DATASET_DIR/'data'/f'{chosen.case_id}.h5') as f:
        image, label = f['image'][:], f['label'][:]
    z = int(np.argmax((label == c).sum(axis=(1,2))))
    points = np.argwhere(label[z] == c)
    lo = np.maximum(0,points.min(axis=0)-30)
    hi = np.minimum(image.shape[1:],points.max(axis=0)+31)
    plane = image[z,lo[0]:hi[0],lo[1]:hi[1]]
    mask = label[z,lo[0]:hi[0],lo[1]:hi[1]] == c
    ax.imshow(plane,cmap='gray',aspect=.5/.5)
    ax.contour(mask,levels=[.5],colors=['red'],linewidths=1)
    ax.set_title(f'{name} | {chosen[f"c{c}_ml"]:.2f} ml\\n{chosen.case_id}\\nz={z}',fontsize=8)
    ax.axis('off')
plt.tight_layout(); plt.show()
''')
    md('''## 8. Từ bằng chứng đến thí nghiệm và các vấn đề cần rà soát

| Bằng chứng cần xem ở trên | Bước tiếp theo phù hợp | Điều chưa được chứng minh |
|---|---|---|
| Kích thước toàn bộ dữ liệu, mức đệm và phạm vi vật lý của khối ảnh | So sánh Z=32 với Z lớn hơn và XY=160/192/224 trong cùng giới hạn bộ nhớ | Kích thước tối ưu, tốc độ và VRAM cần được đo |
| Ngoại lệ có khoảng cách lát cắt nhỏ và thay đổi thể tích lớn | Xác minh siêu dữ liệu thu nhận, kiểm tra GT gốc trước khi chọn khoảng cách lấy mẫu khác | Chưa xác nhận phép lấy mẫu lại hiện tại bảo toàn hoàn toàn nhãn |
| Vùng tổn thương thưa và lớp hiếm | Dùng Dice+CE làm mốc; so sánh trọng số lớp hoặc Tversky bằng chỉ số từng lớp | Chỉ tỷ lệ hiện diện chưa chứng minh CE sẽ suy sụp hay bắt buộc có trọng số lớp |
| Xác suất gặp tổn thương theo ca huấn luyện và theo lớp | Thử xác suất chọn tâm theo tổn thương 0/0,33/0,5; xử lý lấy mẫu không nhãn riêng | Một xác suất cố định không bảo đảm hội tụ ổn định |
| Hình thái hợp vùng ca–lớp | Cân nhắc đánh giá đường biên và thử hàm mất mát cho cấu trúc mỏng | Hình thái không bắt buộc dùng transformer hay hàm mất mát đường biên |
| Khác biệt KS ở các đặc trưng quan sát được | Kiểm tra siêu dữ liệu máy/quy trình chụp và chất lượng nhãn giả theo nhóm | Chưa thể khẳng định máy chụp tương đương hoặc không có dịch chuyển phân phối |
| Tập xác thực trùng tập kiểm thử và các lớp hiếm trong xác thực | Báo cáo chỉ số toàn thể tích theo lớp, số mẫu và độ bất định | Danh sách trùng nhau không cung cấp kết quả kiểm thử độc lập |

Chọn bài toán nhị phân hay đa lớp theo câu hỏi nghiên cứu; mất cân bằng lớp chưa đủ để thay đổi nhiệm vụ. Đánh giá Dice và độ nhạy theo lớp, kích thước tổn thương, cùng dương tính giả trên ca không có tổn thương; quy định rõ cách xử lý GT rỗng. Nếu dùng chỉ số bề mặt, đo khoảng cách bằng mm. Xác thực trên toàn thể tích. Lịch tăng trọng số nhất quán, phạm vi tăng cường dữ liệu và trường tiếp nhận lớn hơn vẫn cần thí nghiệm có kiểm soát.

Trước khi thay đổi dữ liệu, xem `bhsd_preprocessing_review_cases.csv`, ngoại lệ hình học nguồn và GT có nhãn nhưng rỗng. Nguồn gốc chính xác của phép nội suy/quy ước tọa độ và mã bệnh nhân đáng tin cậy vẫn cần xác minh. Không tự động sửa dữ liệu huấn luyện.

### Tái lập kết quả

Chạy `python tools/generate_bhsd_eda_notebook.py` từ bất kỳ thư mục làm việc nào. Đặt `BHSD_DATASET_DIR` / `BHSD_RAW_DIR` để đổi đường dẫn nguồn và `BHSD_EDA_TRIALS` để tăng độ chính xác Monte Carlo (mặc định 40 lần cho mỗi ca/cấu hình).

Bảng và tệp kê khai nằm cạnh notebook. Các bản CSV thống kê/hình thái cũ tại gốc kho mã được đồng bộ để tương thích. Chỉ xóa `notebooks/eda_cache/` khi cần buộc quét lại nguồn; mỗi lần chạy bình thường đều tự kiểm tra dấu vân tay bộ nhớ đệm.''')
    # Kiểm tra cú pháp trước khi khởi động kernel hoặc quét dữ liệu.
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'EDA cell {i}', 'exec')
    path = ROOT/'notebooks/EDA_BHSD_Dataset.ipynb'
    nbformat.write(nb,path)
    if execute:
        client = NotebookClient(nb,timeout=3600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
        def log_cell(cell, cell_index, **kwargs):
            print(f'Executed code cell {cell_index}', flush=True)
        client.on_cell_executed = log_cell
        try:
            client.execute()
        finally:
            nbformat.write(nb,path)
    print(f'Notebook saved: {path}',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-execute',action='store_true')
    args = parser.parse_args()
    create_eda_notebook(execute=not args.no_execute)
