try:
    from .crawl_storage import new_snapshot, save_run_export, write_json
except ImportError:
    from crawl_storage import new_snapshot, save_run_export, write_json

import os
import sys
import json
import wandb


entity = "thanhnguyen2005243-ho-chi-minh-city-university-of-technology"
project = "BHSD_MeanTeacher"

def clean_data(obj):
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if hasattr(obj, "items"):
        return {str(k): clean_data(v) for k, v in obj.items() if not str(k).startswith("_") or k == "_step"}
    if isinstance(obj, (list, tuple)):
        return [clean_data(x) for x in obj]
    return str(obj)

def extract_metrics(run):
    print(f"[*] Đang trích xuất dữ liệu cho {run.name} ({run.id})...", flush=True)
    val_metrics = {}
    try:
        summary_dict = dict(run.summary._json_dict) if hasattr(run.summary, "_json_dict") else dict(run.summary)
    except Exception:
        summary_dict = {k: v for k, v in run.summary.items()}

    for k, v in summary_dict.items():
        if any(x in str(k).lower() for x in ["val", "dice", "loss", "iter", "epoch"]):
            val_metrics[str(k)] = clean_data(v)
    
    target_cols = [
        "_step", "iter", "val/mDice", "val/best_mDice", 
        "val/dice_Epidural", "val/dice_Intraparenchymal", 
        "val/dice_Intraventricular", "val/dice_Subarachnoid", "val/dice_Subdural",
        "train/total", "train/supervised", "train/consistency", "train/weight", "train/lr"
    ]
    rows = []
    try:
        for row in run.scan_history():
            if row.get("val/mDice") is not None:
                if row.get("iter") is None:
                    row["iter"] = row.get("_step")
                cleaned_row = {c: clean_data(row.get(c)) for c in target_cols if c in row}
                rows.append(cleaned_row)
    except Exception as e:
        print(f"[!] Lỗi scan_history: {e}", flush=True)

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
        "history": rows,
        "artifacts": artifacts_list
    }

def main():
    print(f"[*] Kết nối W&B: {entity}/{project} ...", flush=True)
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", order="-created_at"))
    
    print(f"Tìm thấy {len(runs)} runs:")
    for i, r in enumerate(runs[:10]):
        print(f"  [{i}] {r.name} | id: {r.id} | state: {r.state} | created: {r.created_at}")

    if not runs:
        print("Không có run nào.")
        return

    latest_run = runs[0]
    print(f"\n=> Run mới nhất hiện tại trên W&B: {latest_run.name} ({latest_run.id})")
    data = extract_metrics(latest_run)

    folder = save_run_export(latest_run, data)
    print(f"Saved run snapshot: {folder}")

if __name__ == "__main__":
    main()
