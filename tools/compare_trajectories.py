import json

with open("wandb_results.json") as f:
    data = json.load(f)

rec_hist = {h["iter"]: h for h in data["target_recent"]["history"]}
prev_hist = {h["iter"]: h for h in data["target_prev"]["history"]}
base_hist = {h["iter"]: h for h in data["target_baseline"]["history"]}

all_iters = sorted(list(set(list(rec_hist.keys()) + list(prev_hist.keys()))))

print(f"{'Iter':>6} | {'Recent mDice':>12} {'(Best)':>7} | {'Prev mDice':>10} {'(Best)':>7} | {'Base mDice':>10} | {'Rec Loss Tot':>12} {'Sup':>8} {'Cons':>8} {'w_cons':>8}")
print("-" * 105)

for it in all_iters:
    r = rec_hist.get(it, {})
    p = prev_hist.get(it, {})
    b = base_hist.get(it, {})
    
    r_m = f"{r.get('val/mDice', 0):.4f}" if r.get('val/mDice') is not None else "-"
    r_b = f"{r.get('val/best_mDice', 0):.4f}" if r.get('val/best_mDice') is not None else "-"
    p_m = f"{p.get('val/mDice', 0):.4f}" if p.get('val/mDice') is not None else "-"
    p_b = f"{p.get('val/best_mDice', 0):.4f}" if p.get('val/best_mDice') is not None else "-"
    b_m = f"{b.get('val/mDice', 0):.4f}" if b.get('val/mDice') is not None else "-"

    tot = f"{r.get('train/total', 0):.4f}" if r.get('train/total') is not None else "-"
    sup = f"{r.get('train/supervised', 0):.4f}" if r.get('train/supervised') is not None else "-"
    cons = f"{r.get('train/consistency', 0):.4f}" if r.get('train/consistency') is not None else "-"
    w = f"{r.get('train/weight', 0):.5f}" if r.get('train/weight') is not None else "-"

    print(f"{it:6d} | {r_m:>12} {r_b:>7} | {p_m:>10} {p_b:>7} | {b_m:>10} | {tot:>12} {sup:>8} {cons:>8} {w:>8}")
