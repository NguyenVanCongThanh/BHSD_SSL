import argparse
import json
from pathlib import Path
import pandas as pd
import wandb

def export_run_history(run_path_or_id, output_path=None, project="BHSD_MeanTeacher", entity="thanhnguyen2005243-ho-chi-minh-city-university-of-technology"):
    api = wandb.Api()
    if "/" in run_path_or_id:
        full_path = run_path_or_id
    else:
        full_path = f"{entity}/{project}/{run_path_or_id}"
        
    print(f"[*] Đang kết nối W&B lấy dữ liệu run: {full_path} ...", flush=True)
    run = api.run(full_path)
    print(f"[*] Run: {run.name} | Trạng thái: {run.state}")
    
    # 1. Trích xuất toàn bộ history metrics
    print("[*] Đang kéo toàn bộ lịch sử (scan_history)...", flush=True)
    records = []
    for step_data in run.scan_history():
        # Lọc bỏ các object hình ảnh / media phức tạp để giữ dạng tabular gọn gàng
        row = {}
        for k, v in step_data.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[k] = v
            elif isinstance(v, dict) and "_type" in v and v["_type"] == "images/separated":
                continue # bỏ qua media binary
        records.append(row)
        
    df = pd.DataFrame(records)
    print(f"✅ Đã tải về {len(df)} dòng metrics với {len(df.columns)} cột biểu đồ.")
    
    if output_path is None:
        output_dir = Path("experiment_data/wandb_exports") / run.id
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_file = output_dir / "metrics_history.csv"
        json_file = output_dir / "metrics_history.json"
    else:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        csv_file = out_p.with_suffix(".csv")
        json_file = out_p.with_suffix(".json")
        
    df.to_csv(csv_file, index=False)
    with open(json_file, "w") as f:
        json.dump(records, f, indent=2)
        
    print(f"📁 Đã lưu file CSV: {csv_file}")
    print(f"📁 Đã lưu file JSON: {json_file}")
    print("\n[Danh sách các metrics / biểu đồ có trong file]:")
    print(sorted(df.columns.tolist()))
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kéo toàn bộ giá trị biểu đồ từ W&B run về máy.")
    parser.add_argument("--run_id", type=str, required=True, help="ID của run (ví dụ: 9tvq8qld hoặc entity/project/id)")
    parser.add_argument("--project", type=str, default="BHSD_MeanTeacher", help="Tên project W&B")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn file đầu ra")
    args = parser.parse_args()
    
    export_run_history(args.run_id, output_path=args.output, project=args.project)
