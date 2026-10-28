try:
    from .crawl_storage import new_snapshot
except ImportError:
    from crawl_storage import new_snapshot

import wandb


entity = "thanhnguyen2005243-ho-chi-minh-city-university-of-technology"
project = "BHSD_MeanTeacher"

def main():
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", order="-created_at"))
    
    lines = []
    lines.append(f"Total runs: {len(runs)}")
    for i, r in enumerate(runs):
        s = dict(r.summary)
        train_iter = s.get("train/iter", s.get("iter", s.get("_step")))
        runtime = s.get("_runtime", "N/A")
        best_dice = s.get("val/best_mDice", "N/A")
        line = f"[{i}] {r.name} | id: {r.id} | state: {r.state} | iter: {train_iter} | runtime: {runtime}s | best_dice: {best_dice} | created: {r.created_at}"
        lines.append(line)
        print(line, flush=True)

    folder = new_snapshot("wandb", "_collections", entity=entity, project=project)
    print(f"Snapshot: {folder}")
    with open(folder / "runs_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
