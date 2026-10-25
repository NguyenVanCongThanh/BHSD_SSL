import os
import json
import wandb

WANDB_API_KEY = "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"
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

entity = "thanhnguyen2005243-ho-chi-minh-city-university-of-technology"
project = "BHSD_MeanTeacher"
run_id = "s9i9nsui"

def inspect_system_metrics():
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    
    # Try getting system stream
    try:
        sys_hist = list(run.history(stream="events"))
        print(f"System events count: {len(sys_hist)}")
        if sys_hist:
            print("First event keys:", list(sys_hist[0].keys()))
            print("First event:", sys_hist[0])
            print("Last event:", sys_hist[-1])
            # Check max memory
            mems = [e.get("system.memory", 0) for e in sys_hist if "system.memory" in e]
            proc_mems = [e.get("system.proc.memory.rssMB", 0) for e in sys_hist if "system.proc.memory.rssMB" in e]
            disks = [e.get("system.disk", 0) for e in sys_hist if "system.disk" in e]
            if mems:
                print(f"RAM %: min={min(mems):.1f}%, max={max(mems):.1f}%, last={mems[-1]:.1f}%")
            if proc_mems:
                print(f"Proc RSS MB: min={min(proc_mems):.1f}MB, max={max(proc_mems):.1f}MB, last={proc_mems[-1]:.1f}MB")
            if disks:
                print(f"Disk %: min={min(disks):.1f}%, max={max(disks):.1f}%, last={disks[-1]:.1f}%")
    except Exception as e:
        print("Error getting events:", e)

if __name__ == "__main__":
    inspect_system_metrics()
