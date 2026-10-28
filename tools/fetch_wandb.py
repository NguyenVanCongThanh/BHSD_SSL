try:
    from .crawl_storage import clean_data, new_snapshot, write_json
except ImportError:
    from crawl_storage import clean_data, new_snapshot, write_json

import json
import wandb

# API Key và cấu hình dự án

entity = "thanhnguyen2005243-ho-chi-minh-city-university-of-technology"
project = "BHSD_MeanTeacher"

def extract_metrics(run):
    print(f"       [*] Đang trích xuất dữ liệu cho {run.name} ({run.id})...", flush=True)
    val_metrics = {}
    try:
        summary_dict = dict(run.summary._json_dict) if hasattr(run.summary, "_json_dict") else dict(run.summary)
    except Exception:
        summary_dict = {k: v for k, v in run.summary.items()}

    for k, v in summary_dict.items():
        if any(x in str(k).lower() for x in ["val", "dice", "loss", "iter", "epoch"]):
            val_metrics[str(k)] = clean_data(v)
    
    # Lấy history table
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
        print(f"       [!] Lỗi lấy scan_history cho {run.name}: {e}", flush=True)

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
    print(f"[*] Đang kết nối W&B API lấy kết quả từ: {entity}/{project} ...", flush=True)
    try:
        api = wandb.Api()
        runs = list(api.runs(f"{entity}/{project}", order="-created_at"))
        
        if not runs:
            print("[!] Không tìm thấy run nào trong project.")
            return

        print(f"✅ Tìm thấy tổng cộng {len(runs)} runs trên W&B:")
        for idx, r in enumerate(runs):
            print(f"   [{idx+1}] {r.name} (State: {r.state}, Created: {r.created_at})", flush=True)
        
        results = {
            "all_runs": [],
            "target_recent": None,      # Run mới nhất (runs[0])
            "target_prev": None,        # Run ngay trước đó (20261017_095556)
            "target_baseline": None     # Run chuẩn 12k iter (20261016_152451)
        }

        for run in runs:
            run_info = {
                "name": str(run.name),
                "id": str(run.id),
                "state": str(run.state),
                "created_at": str(run.created_at)
            }
            results["all_runs"].append(run_info)

        # 1. Chọn run mới nhất làm target_recent
        print(f"\n[+] Đang lấy dữ liệu Run Mới Nhất (Recent): {runs[0].name}", flush=True)
        results["target_recent"] = extract_metrics(runs[0])

        # 2. Chọn run ngay trước đó (20261017_095556)
        prev_run = None
        for r in runs[1:]:
            if "20261017_095556" in r.name:
                prev_run = r
                break
        if prev_run is None and len(runs) > 1:
            prev_run = runs[1]
        
        if prev_run:
            print(f"\n[+] Đang lấy dữ liệu Run Liền Kề Trước (Previous): {prev_run.name}", flush=True)
            results["target_prev"] = extract_metrics(prev_run)

        # 3. Chọn run chuẩn 12k iters (20261016_152451)
        baseline_run = None
        for r in runs:
            if "20261016_152451" in r.name or ("finished" in str(r.state).lower() and "12000" in r.name):
                baseline_run = r
                break
        if baseline_run:
            print(f"\n[+] Đang lấy dữ liệu Run Baseline Hoàn Chỉnh (12k it): {baseline_run.name}", flush=True)
            results["target_baseline"] = extract_metrics(baseline_run)

        for key in ('target_recent', 'target_prev', 'target_baseline'):
            record = results.get(key)
            if record:
                folder = new_snapshot('wandb', record['id'], entity=entity, project=project)
                write_json(folder / 'export.json', record)
                write_json(folder / 'config.json', record.get('config', {}))
                write_json(folder / 'summary.json', record.get('summary', {}))

        # Lưu JSON an toàn
        output_json_path = new_snapshot("wandb", "_collections", entity=entity, project=project) / "wandb_results.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[SUCCESS] Đã ghi toàn bộ dữ liệu ra: {output_json_path}", flush=True)

        # In tóm tắt ra màn hình
        print("\n" + "="*80)
        print("TỔNG HỢP SO SÁNH CÁC RUN MEAN TEACHER")
        print("="*80)

        runs_to_show = [
            ("Run Mới Nhất (20261018_061745)", results["target_recent"]), 
            ("Run Liền Trước (20261017_095556)", results["target_prev"]),
            ("Run Baseline Chuẩn (20261016_152451)", results["target_baseline"])
        ]

        for label, r in runs_to_show:
            if not r:
                print(f"\n[-] {label}: Không tìm thấy dữ liệu!")
                continue
            print(f"\n🔹 {label}: {r['name']}")
            print(f"   - ID: {r['id']} | State: {r['state']} | Created: {r['created_at']}")
            print(f"   - URL: {r['url']}")
            s = r["summary"]
            print(f"   - Best mDice: {s.get('val/best_mDice', 'N/A')}")
            print(f"   - Final mDice: {s.get('val/mDice', 'N/A')}")
            print(f"   - Per-class Dice (Final/Summary):")
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
