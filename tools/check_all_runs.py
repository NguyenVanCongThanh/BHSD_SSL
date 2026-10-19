import os
import json
import wandb

WANDB_API_KEY = "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"
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

api = wandb.Api()
runs = list(api.runs("thanhnguyen2005243-ho-chi-minh-city-university-of-technology/BHSD_MeanTeacher", order="-created_at"))

print(f"Total runs: {len(runs)}")
for i, r in enumerate(runs):
    s_dict = dict(r.summary)
    print(f"[{i}] {r.name} | id: {r.id} | state: {r.state} | created: {r.created_at} | summary keys count: {len(s_dict)}")
    val_mDice = s_dict.get("val/mDice")
    best_mDice = s_dict.get("val/best_mDice")
    iter_val = s_dict.get("iter")
    best_iter = s_dict.get("best_iter")
    print(f"    iter: {iter_val}, best_iter: {best_iter}, val/mDice: {val_mDice}, val/best_mDice: {best_mDice}")
