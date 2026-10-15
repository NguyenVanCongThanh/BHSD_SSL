import os
import sys
import json
import wandb
import pandas as pd

# API Key và cấu hình dự án
WANDB_API_KEY = "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"
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

entity = "thanhnguyen2005243-ho-chi-minh-city-university-of-technology"
project = "BHSD_MeanTeacher"

def main():
    print(f"[*] Đang kết nối W&B API lấy kết quả từ: {entity}/{project} ...")
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}", order="-created_at")
        
        if not runs:
            print("[!] Không tìm thấy run nào trong project.")
            return

        print(f"\n✅ Tìm thấy tổng cộng {len(runs)} runs. Chi tiết các run gần nhất:")
        print("=" * 80)

        for i, run in enumerate(runs[:5]):
            print(f"\n🔹 RUN #{i+1}: {run.name} (ID: {run.id})")
            print(f"   - Trạng thái: {run.state.upper()}")
            print(f"   - Thời gian tạo: {run.created_at}")
            print(f"   - Link W&B: {run.url}")
            
            # Lọc các metrics quan trọng (Validation, Dice, Loss, Iteration)
            val_metrics = {}
            for k, v in run.summary.items():
                if any(x in k.lower() for x in ["val", "dice", "loss", "iter", "epoch"]):
                    val_metrics[k] = v
            
            if val_metrics:
                print("   - Metrics tóm tắt:")
                for k, v in sorted(val_metrics.items()):
                    if isinstance(v, float):
                        print(f"       * {k:25s}: {v:.4f}")
                    else:
                        print(f"       * {k:25s}: {v}")
            else:
                print("   - (Chưa có metrics nào được log hoặc run đang khởi tạo)")

            # In chi tiết toàn bộ lịch sử (History) của biểu đồ Validation qua các mốc iter
            print("\n   📈 CHI TIẾT LỊCH SỬ VALIDATION QUA TỪNG MỐC (History Table):")
            try:
                # Lọc trực tiếp các bản ghi có chứa val/mDice để không bị dính NaN từ các step train
                rows = []
                for row in run.scan_history():
                    if row.get("val/mDice") is not None:
                        rows.append(row)
                
                if rows:
                    val_df = pd.DataFrame(rows)
                    target_cols = ["_step", "iter", "val/mDice", "val/best_mDice", 
                                   "val/dice_Epidural", "val/dice_Intraparenchymal", 
                                   "val/dice_Intraventricular", "val/dice_Subarachnoid", "val/dice_Subdural"]
                    existing_cols = [c for c in target_cols if c in val_df.columns]
                    val_df_clean = val_df[existing_cols].dropna(subset=["val/mDice"]).reset_index(drop=True)
                    print(val_df_clean.to_string(index=False))
                else:
                    print("       (Không có bản ghi validation nào được lưu trong history)")
            except Exception as e:
                print(f"       Không thể tải history table: {e}")



            # Kiểm tra Artifacts (Checkpoint / Models)
            try:
                artifacts = list(run.logged_artifacts())
                if artifacts:
                    print("\n   📦 ARTIFACTS ĐÃ LƯU:")
                    for art in artifacts:
                        print(f"       * {art.name} ({art.type}) - {art.version}")
            except Exception:
                pass
            print("=" * 80)
            
    except Exception as e:
        print(f"[!] Lỗi khi gọi W&B API: {e}")

if __name__ == "__main__":
    main()


