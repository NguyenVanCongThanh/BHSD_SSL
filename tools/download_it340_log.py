import os
import wandb

WANDB_API_KEY = "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"
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

api = wandb.Api()
for run_id in ["a35mtk0i", "fum8iuny", "1ex7ty08"]:
    run = api.run(f"thanhnguyen2005243-ho-chi-minh-city-university-of-technology/BHSD_MeanTeacher/{run_id}")
    print(f"\nRun: {run.name} ({run.id}) | State: {run.state}")
    for f in run.files():
        print(f"  - {f.name} ({f.size} B)")
        if "output.log" in f.name:
            out_dir = f"tools/{run_id}"
            os.makedirs(out_dir, exist_ok=True)
            f.download(replace=True, root=out_dir)
            print(f"    -> Downloaded to {out_dir}")
