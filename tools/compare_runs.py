"""
Unified Offline Experiment Comparison Tool for MedSSL-BHSD.
Consolidates run summary, training trajectory, and per-class segmentation metrics
from collection snapshots (wandb_results.json).
"""
import sys
import json
import argparse
from pathlib import Path

try:
    from .crawl_storage import comparison_file
except ImportError:
    from crawl_storage import comparison_file


def print_summary(data):
    r_recent = data.get("target_recent", {})
    r_prev = data.get("target_prev", {})
    r_base = data.get("target_baseline", {})

    print("=" * 60)
    print("RUN IDENTIFIERS")
    print(f"Recent:   {r_recent.get('name')} ({r_recent.get('id')}) - State: {r_recent.get('state')}")
    print(f"Previous: {r_prev.get('name')} ({r_prev.get('id')}) - State: {r_prev.get('state')}")
    print(f"Baseline: {r_base.get('name')} ({r_base.get('id')}) - State: {r_base.get('state')}")

    print("\n" + "=" * 60)
    print("CONFIG DIFFERENCES (method & model)")
    recent_method = (r_recent.get('config') or {}).get('method', {})
    prev_method = (r_prev.get('config') or {}).get('method', {})
    base_method = (r_base.get('config') or {}).get('method', {})

    for key in sorted(set(list(recent_method.keys()) + list(prev_method.keys()))):
        v_rec = recent_method.get(key)
        v_prev = prev_method.get(key)
        v_base = base_method.get(key)
        if v_rec != v_prev or v_rec != v_base or v_prev != v_base:
            print(f"  {key:25s} | Rec: {str(v_rec):15s} | Prev: {str(v_prev):15s} | Base: {str(v_base):15s}")

    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    for name, r in [("Recent", r_recent), ("Previous", r_prev), ("Baseline", r_base)]:
        s = r.get('summary', {})
        print(f"\n--- {name} ({r.get('id')}) ---")
        print(f"  Best mDice:  {s.get('val/best_mDice')} (at best_iter: {s.get('best_iter')})")
        print(f"  Final mDice: {s.get('val/mDice')}")
        print(f"  Epidural:          {s.get('val/dice_Epidural')}")
        print(f"  Intraparenchymal:  {s.get('val/dice_Intraparenchymal')}")
        print(f"  Intraventricular:  {s.get('val/dice_Intraventricular')}")
        print(f"  Subarachnoid:      {s.get('val/dice_Subarachnoid')}")
        print(f"  Subdural:          {s.get('val/dice_Subdural')}")
        hist = r.get('history', [])
        print(f"  Logged val steps:  {len(hist)}")
        if hist:
            iters = [h.get('iter') for h in hist if 'iter' in h]
            if iters:
                print(f"  Iter range: {iters[0]} -> {iters[-1]}")


def print_trajectory(data):
    rec_hist = {h["iter"]: h for h in data.get("target_recent", {}).get("history", []) if "iter" in h}
    prev_hist = {h["iter"]: h for h in data.get("target_prev", {}).get("history", []) if "iter" in h}
    base_hist = {h["iter"]: h for h in data.get("target_baseline", {}).get("history", []) if "iter" in h}

    all_iters = sorted(list(set(list(rec_hist.keys()) + list(prev_hist.keys()))))
    if not all_iters:
        print("\n[Trajectory] Không có dữ liệu lịch sử lặp (history iter) để hiển thị.")
        return

    print("\n" + "=" * 105)
    print("TRAJECTORY COMPARISON (Validation mDice & Losses across Iterations)")
    print("=" * 105)
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


def print_classes(data):
    recent_history = data.get("target_recent", {}).get("history", [])
    if not recent_history:
        print("\n[Classes] Không có dữ liệu lịch sử per-class để hiển thị.")
        return

    print("\n" + "=" * 65)
    print("PER-CLASS DICE TREND (Target Recent)")
    print("=" * 65)
    print(f"{'Iter':>6} | {'mDice':>7} | {'Epidural':>8} {'IntraPar':>8} {'IntraVen':>8} {'SubArach':>8} {'Subdural':>8}")
    print("-" * 65)

    for h in recent_history:
        it = h.get("iter", 0)
        m = h.get("val/mDice", 0)
        ed = h.get("val/dice_Epidural", 0)
        ip = h.get("val/dice_Intraparenchymal", 0)
        iv = h.get("val/dice_Intraventricular", 0)
        sa = h.get("val/dice_Subarachnoid", 0)
        sd = h.get("val/dice_Subdural", 0)
        print(f"{it:6d} | {m:7.4f} | {ed:8.4f} {ip:8.4f} {iv:8.4f} {sa:8.4f} {sd:8.4f}")


def main(mode=None, input_file=None):
    if mode is None:
        parser = argparse.ArgumentParser(description="Unified MedSSL-BHSD Experiment Comparison CLI")
        parser.add_argument('--input', type=Path, help='Path to wandb_results.json')
        parser.add_argument('--mode', choices=['all', 'summary', 'trajectory', 'classes'], default='all',
                            help='Report display mode: all, summary, trajectory, classes (default: all)')
        args, _ = parser.parse_known_args()
        target_file = comparison_file(args.input) if not input_file else Path(input_file)
        selected_mode = args.mode
    else:
        target_file = comparison_file(input_file)
        selected_mode = mode

    with open(target_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if selected_mode in ('all', 'summary'):
        print_summary(data)
    if selected_mode in ('all', 'trajectory'):
        print_trajectory(data)
    if selected_mode in ('all', 'classes'):
        print_classes(data)


if __name__ == "__main__":
    main()
