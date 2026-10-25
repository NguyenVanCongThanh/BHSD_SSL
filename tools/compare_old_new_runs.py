import os
import wandb

WANDB_API_KEY = "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"
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

api = wandb.Api()
r_old = api.run("thanhnguyen2005243-ho-chi-minh-city-university-of-technology/BHSD_MeanTeacher/fum8iuny")
r_new = api.run("thanhnguyen2005243-ho-chi-minh-city-university-of-technology/BHSD_MeanTeacher/s9i9nsui")

cfg_old = {k: v for k, v in r_old.config.items() if not str(k).startswith("_")}
cfg_new = {k: v for k, v in r_new.config.items() if not str(k).startswith("_")}

print("=== RUN fum8iuny (iter 341) vs s9i9nsui (iter 3996) ===")
print("Old batch size:", cfg_old.get("data", {}).get("batch_size"))
print("New batch size:", cfg_new.get("data", {}).get("batch_size"))
print("Old patch size:", cfg_old.get("data", {}).get("patch_size"))
print("New patch size:", cfg_new.get("data", {}).get("patch_size"))

for sec in ["data", "model", "method", "logging", "experiment"]:
    d1 = cfg_old.get(sec, {})
    d2 = cfg_new.get(sec, {})
    if isinstance(d1, dict) and isinstance(d2, dict):
        diffs = {k: (d1.get(k), d2.get(k)) for k in set(d1) | set(d2) if d1.get(k) != d2.get(k)}
        if diffs:
            print(f"\nDiffs in [{sec}]:")
            for k, (v1, v2) in diffs.items():
                print(f"  {k}: {v1} -> {v2}")
