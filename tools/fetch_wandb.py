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

TARGET_RUN_RECENT = "20261017_095556"
TARGET_RUN_PREV = "20261017_152451"

def clean_data(obj):
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if hasattr(obj, "items"):
        return {str(k): clean_data(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [clean_data(x) for x in obj]
    return str(obj)

def extract_metrics(run):
    val_metrics = {}
    try:
        summary_dict = dict(run.summary._json_dict) if hasattr(run.summary, "_json_dict") else dict(run.summary)
    except Exception:
        summary_dict = {k: v for k, v in run.summary.items()}

    for k, v in summary_dict.items():
        if any(x in str(k).lower() for x in ["val", "dice", "loss", "iter", "epoch"]):
            val_metrics[str(k)] = clean_data(v)
    
    # Lấy history table
    rows = []
    try:
        for row in run.scan_history():
            if row.get("val/mDice") is not None:
                rows.append(row)
    except Exception as e:
        print(f"       [!] Lỗi lấy scan_history cho {run.name}: {e}")

    history_records = []
    if rows:
        val_df = pd.DataFrame(rows)
        target_cols = ["_step", "iter", "val/mDice", "val/best_mDice", 
                       "val/dice_Epidural", "val/dice_Intraparenchymal", 
                       "val/dice_Intraventricular", "val/dice_Subarachnoid", "val/dice_Subdural"]
        existing_cols = [c for c in target_cols if c in val_df.columns]
        val_df_clean = val_df[existing_cols].dropna(subset=["val/mDice"]).reset_index(drop=True)
        history_records = [clean_data(r) for r in val_df_clean.to_dict(orient="records")]

    artifacts_list = []
    try:
        for art in run.logged_artifacts():
            artifacts_list.append({"name": str(art.name), "type": str(art.type), "version": str(art.version)})
    except Exception:
        pass

    cfg_clean = {}
    try:
        for k, v in run.config.items():
            if not str(k).startswith("_"):
                cfg_clean[str(k)] = clean_data(v)
    except Exception:
        pass

    return {
        "id": str(run.id),
        "name": str(run.name),
        "state": str(run.state),
        "created_at": str(run.created_at),
        "url": str(run.url),
        "config": cfg_clean,
        "summary": val_metrics,
        "history": history_records,
        "artifacts": artifacts_list
    }

def main():
    print(f"[*] Đang kết nối W&B API lấy kết quả từ: {entity}/{project} ...")
    try:
        api = wandb.Api()
        runs = list(api.runs(f"{entity}/{project}", order="-created_at"))
        
        if not runs:
            print("[!] Không tìm thấy run nào trong project.")
            return

        print(f"✅ Tìm thấy tổng cộng {len(runs)} runs trên W&B:")
        for idx, r in enumerate(runs):
            print(f"   [{idx+1}] {r.name} (State: {r.state}, Created: {r.created_at})")
        
        results = {
            "all_runs": [],
            "target_recent": None,
            "target_prev": None
        }

        for run in runs:
            run_info = {
                "name": str(run.name),
                "id": str(run.id),
                "state": str(run.state),
                "created_at": str(run.created_at)
            }
            results["all_runs"].append(run_info)
            if TARGET_RUN_RECENT in run.name:
                print(f"\n[+] Đã chọn Run Gần Nhất (Recent): {run.name}")
                results["target_recent"] = extract_metrics(run)
            elif TARGET_RUN_PREV in run.name:
                print(f"\n[+] Đã chọn Run Đối Chứng (Previous): {run.name}")
                results["target_prev"] = extract_metrics(run)

        # Nếu không tìm thấy bằng substring chính xác, lấy run đầu tiên làm recent
        if results["target_recent"] is None and len(runs) > 0:
            print(f"[!] Không tìm thấy substring {TARGET_RUN_RECENT}, chọn run mới nhất: {runs[0].name}")
            results["target_recent"] = extract_metrics(runs[0])

        # Nếu chưa tìm thấy previous run, tìm run hoàn chỉnh 12000 iter khác nếu có
        if results["target_prev"] is None:
            for r in runs:
                if r != runs[0] and ("12000" in r.name or "complete" in r.name.lower() or "152451" in r.name):
                    print(f"[+] Tự động khớp Run Đối Chứng tiềm năng: {r.name}")
                    results["target_prev"] = extract_metrics(r)
                    break

        # Lưu JSON an toàn
        output_json_path = os.path.join(os.path.dirname(__file__), "..", "wandb_results.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[SUCCESS] Đã ghi toàn bộ dữ liệu ra: {output_json_path}")

        # In tóm tắt ra màn hình
        print("\n" + "="*80)
        print("TỔNG HỢP SO SÁNH HAI RUN")
        print("="*80)

        for label, r in [("Run Gần Nhất (Recent)", results["target_recent"]), 
                         ("Run Trước Đó (12k it Complete)", results["target_prev"])]:
            if not r:
                print(f"\n[-] {label}: Không tìm thấy dữ liệu!")
                continue
            print(f"\n🔹 {label}: {r['name']}")
            print(f"   - State: {r['state']} | Created: {r['created_at']}")
            print(f"   - URL: {r['url']}")
            s = r["summary"]
            print(f"   - Final/Best mDice: {s.get('val/best_mDice', s.get('val/mDice', 'N/A'))}")
            print(f"   - Per-class Dice:")
            for cls_name in ["Epidural", "Intraparenchymal", "Intraventricular", "Subarachnoid", "Subdural"]:
                k = f"val/dice_{cls_name}"
                val = s.get(k, "N/A")
                if isinstance(val, (int, float)):
                    print(f"       * {cls_name:18s}: {float(val):.4f}")
                else:
                    print(f"       * {cls_name:18s}: {val}")
            print(f"   - Số mốc Validation đã log: {len(r['history'])}")

        print("\n" + "="*80)

    except Exception as e:
        print(f"[!] Lỗi khi gọi W&B API: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()




