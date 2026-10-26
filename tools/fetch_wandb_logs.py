try:
    from .crawl_storage import new_snapshot, save_run_export, write_json
except ImportError:
    from crawl_storage import new_snapshot, save_run_export, write_json

import os
import json
import wandb


entity = "thanhnguyen2005243-ho-chi-minh-city-university-of-technology"
project = "BHSD_MeanTeacher"
run_id = "s9i9nsui"

def fetch_run_files():
    print(f"[*] Connecting to W&B for run {run_id}...", flush=True)
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    print(f"Run name: {run.name}, state: {run.state}")
    
    folder = new_snapshot("wandb", run.id, entity=entity, project=project)
    files = list(run.files())
    print(f"Found {len(files)} files in run:")
    for f in files:
        print(f"  - {f.name} ({f.size} bytes)")
        if "output.log" in f.name or "log" in f.name.lower():
            dest = folder / f.name
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            print(f"    Downloading {f.name} to {dest}...")
            f.download(replace=True, root=str(folder))
            print(f"    Downloaded successfully!")

if __name__ == "__main__":
    fetch_run_files()
