try:
    from .crawl_storage import comparison_file
except ImportError:
    from crawl_storage import comparison_file

import json

with comparison_file().open(encoding="utf-8") as f:
    data = json.load(f)

print(f"{'Iter':>6} | {'mDice':>7} | {'Epidural':>8} {'IntraPar':>8} {'IntraVen':>8} {'SubArach':>8} {'Subdural':>8}")
print("-" * 65)

for h in data["target_recent"]["history"]:
    it = h.get("iter", 0)
    m = h.get("val/mDice", 0)
    ed = h.get("val/dice_Epidural", 0)
    ip = h.get("val/dice_Intraparenchymal", 0)
    iv = h.get("val/dice_Intraventricular", 0)
    sa = h.get("val/dice_Subarachnoid", 0)
    sd = h.get("val/dice_Subdural", 0)
    print(f"{it:6d} | {m:7.4f} | {ed:8.4f} {ip:8.4f} {iv:8.4f} {sa:8.4f} {sd:8.4f}")
