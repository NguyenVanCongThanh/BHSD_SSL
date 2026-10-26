try:
    from .crawl_storage import new_snapshot, save_run_export, write_json
except ImportError:
    from crawl_storage import new_snapshot, save_run_export, write_json

import os
import wandb


api = wandb.Api()
for run_id in ["a35mtk0i", "fum8iuny", "1ex7ty08"]:
    run = api.run(f"thanhnguyen2005243-ho-chi-minh-city-university-of-technology/BHSD_MeanTeacher/{run_id}")
    print(f"\nRun: {run.name} ({run.id}) | State: {run.state}")
    folder = new_snapshot("wandb", run.id, entity=run.entity, project=run.project)
    for f in run.files():
        print(f"  - {f.name} ({f.size} B)")
        if "output.log" in f.name:
            out_dir = str(folder)
            os.makedirs(out_dir, exist_ok=True)
            f.download(replace=True, root=out_dir)
            print(f"    -> Downloaded to {out_dir}")
