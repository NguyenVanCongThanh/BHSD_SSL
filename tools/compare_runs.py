import json

with open("wandb_results.json") as f:
    data = json.load(f)

r_recent = data["target_recent"]
r_prev = data["target_prev"]
r_base = data["target_baseline"]

print("="*60)
print("RUN IDENTIFIERS")
print(f"Recent:   {r_recent['name']} ({r_recent['id']}) - State: {r_recent['state']}")
print(f"Previous: {r_prev['name']} ({r_prev['id']}) - State: {r_prev['state']}")
print(f"Baseline: {r_base['name']} ({r_base['id']}) - State: {r_base['state']}")

print("\n" + "="*60)
print("CONFIG DIFFERENCES (method & model)")
for key in set(list(r_recent['config']['method'].keys()) + list(r_prev['config']['method'].keys())):
    v_rec = r_recent['config']['method'].get(key)
    v_prev = r_prev['config']['method'].get(key)
    v_base = r_base['config']['method'].get(key)
    if v_rec != v_prev or v_rec != v_base or v_prev != v_base:
        print(f"  {key:25s} | Rec: {str(v_rec):15s} | Prev: {str(v_prev):15s} | Base: {str(v_base):15s}")

print("\n" + "="*60)
print("METRICS SUMMARY")
for name, r in [("Recent (20261018_061745)", r_recent), ("Prev (20261017_095556)", r_prev), ("Base (20261016_152451)", r_base)]:
    s = r['summary']
    print(f"\n--- {name} ---")
    print(f"  Best mDice:  {s.get('val/best_mDice')} (at best_iter: {s.get('best_iter')})")
    print(f"  Final mDice: {s.get('val/mDice')}")
    print(f"  Epidural:          {s.get('val/dice_Epidural')}")
    print(f"  Intraparenchymal:  {s.get('val/dice_Intraparenchymal')}")
    print(f"  Intraventricular:  {s.get('val/dice_Intraventricular')}")
    print(f"  Subarachnoid:      {s.get('val/dice_Subarachnoid')}")
    print(f"  Subdural:          {s.get('val/dice_Subdural')}")
    print(f"  Logged val steps:  {len(r.get('history', []))}")
    if r.get('history'):
        iters = [h.get('iter') for h in r['history']]
        print(f"  Iter range: {iters[0]} -> {iters[-1]}")
