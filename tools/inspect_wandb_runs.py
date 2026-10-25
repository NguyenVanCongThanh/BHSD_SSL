import os
import json
import wandb

WANDB_API_KEY = "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"
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

entity = "thanhnguyen2005243-ho-chi-minh-city-university-of-technology"
project = "BHSD_MeanTeacher"

def inspect_runs():
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", order="-created_at"))
    print(f"Total runs: {len(runs)}", flush=True)
    
    for i, r in enumerate(runs[:5]):
        summary = dict(r.summary)
        runtime = summary.get("_runtime", "N/A")
        step = summary.get("_step", "N/A")
        train_iter = summary.get("train/iter", summary.get("iter", "N/A"))
        val_mdice = summary.get("val/mDice", "N/A")
        best_mdice = summary.get("val/best_mDice", "N/A")
        
        batch_size = r.config.get("data", {}).get("batch_size") if isinstance(r.config.get("data"), dict) else r.config.get("batch_size")
        patch_size = r.config.get("data", {}).get("patch_size") if isinstance(r.config.get("data"), dict) else r.config.get("patch_size")
        
        print(f"\n--- [{i}] Run: {r.name} ---", flush=True)
        print(f"  ID: {r.id} | State: {r.state} | Created: {r.created_at}", flush=True)
        print(f"  Runtime (s): {runtime} | Last Step: {step} | Last Train Iter: {train_iter}", flush=True)
        print(f"  Batch size: {batch_size} | Patch size: {patch_size}", flush=True)
        print(f"  val/mDice: {val_mdice} | val/best_mDice: {best_mdice}", flush=True)

if __name__ == "__main__":
    inspect_runs()
