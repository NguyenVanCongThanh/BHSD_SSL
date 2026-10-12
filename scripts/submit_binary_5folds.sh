#!/bin/bash
#SBATCH --job-name=pick_binary_5f
#SBATCH --array=0-4%2
#SBATCH --output=/home/u001015/experiments/logs_pick_binary_fold_%a_%j.log
#SBATCH --error=/home/u001015/experiments/logs_pick_binary_fold_%a_%j.err
#SBATCH --partition=gpu-queue
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00

# 1. Kích hoạt môi trường ảo
source /home/u001015/.venv/nnunet/bin/activate

# 2. Lấy Fold ID từ Slurm Array (0, 1, 2, 3, 4)
FOLD_ID=$SLURM_ARRAY_TASK_ID

# 3. Chuyển vào thư mục PICK
cd /home/u001015/PICK

# 4. Thiết lập thư mục lưu logs/experiments
mkdir -p /home/u001015/experiments

echo "=========================================================================="
echo "=== BẮT ĐẦU CHẠY PICK BINARY FOLD $FOLD_ID (Slurm Job ID: $SLURM_JOB_ID, Task: $FOLD_ID) ==="
echo "=========================================================================="

# 5. Khởi chạy huấn luyện qua điểm chạy hợp nhất MedSSL-BHSD
python train.py \
    --config configs/methods/pick.yaml \
    --fold $FOLD_ID \
    --task binary \
    --experiment.name BHSD_binary_PICK_5folds \
    --gpu 0

echo "=== HOÀN TẤT HUẤN LUYỆN PICK BINARY CHO FOLD $FOLD_ID ==="
