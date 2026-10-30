"""Reproduce offline run analysis using Python's standard library only."""
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiment_data/wandb/o6s4pex3/20261030T042721_825592Z_f68e06a0'
OUT = ROOT / 'docs/analysis'
CLASSES = ['Epidural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid', 'Subdural']


def main():
    history = json.loads((SOURCE / 'history.json').read_text())
    validation = [r for r in history if r.get('val/Epidural_dice_positive_mean') is not None]
    train = [r for r in history if r.get('train/supervised') is not None]
    rows = []
    for r in validation:
        positive = [r[f'val/{c}_dice_positive_mean'] for c in CLASSES]
        rows.append(dict(iteration=r['global_step'],
                         legacy_mdice=mean(r[f'val/{c}_dice_case_mean'] for c in CLASSES),
                         positive_macro=mean(positive),
                         **dict(zip(CLASSES, positive))))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'run_o6s4pex3_validation.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(training_rows=len(train), validation_rows=len(rows),
                  last_train_iteration=train[-1]['train/iter'],
                  last_validation_iteration=rows[-1]['iteration'],
                  best_legacy=max(rows, key=lambda r: r['legacy_mdice']),
                  best_positive_macro=max(rows, key=lambda r: r['positive_macro']),
                  all_background_legacy=sum(validation[-1][f'val/{c}_negative_cases'] for c in CLASSES) / (38 * 5))
    (OUT / 'run_o6s4pex3_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    # A standalone HTML view with tooltips; fixed axes preserve comparison of all metrics.
    colors = ['#111827', '#dc2626', '#7c3aed', '#2563eb', '#0891b2', '#ea580c', '#16a34a']
    keys = ['legacy_mdice', 'positive_macro'] + CLASSES
    svg = ['<svg viewBox="0 0 1080 550" role="img" aria-label="Validation metric history">']
    for tick in range(7):
        y = 450 - tick / 10 / .6 * 380
        svg.append(f'<line x1="70" y1="{y}" x2="1020" y2="{y}" stroke="#ddd"/><text x="25" y="{y+5}">{tick/10:.1f}</text>')
    for tick in range(0, 8401, 1000):
        x = 70 + tick / 8400 * 950
        svg.append(f'<text x="{x-15}" y="480">{tick}</text>')
    for key, color in zip(keys, colors):
        points = ' '.join(f'{70+r["iteration"]/8400*950},{450-r[key]/.6*380}' for r in rows)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for r in rows:
            x, y = 70 + r['iteration']/8400*950, 450-r[key]/.6*380
            svg.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"><title>{key}: {r[key]:.5f} @ {r["iteration"]}</title></circle>')
    svg.append('<text x="450" y="520">Validation iteration</text></svg>')
    legend = ' &nbsp; '.join(f'<span style="color:{c}">● {k}</span>' for k, c in zip(keys, colors))
    html = '<!doctype html><meta charset="utf-8"><title>Run o6s4pex3</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto}svg{width:100%}</style>'
    html += '<h1>Run o6s4pex3 — validation tới 8.400</h1><p>Train tới 8.600; không có validation 8.600. Positive macro gồm đủ 5 lớp. Di chuột vào điểm để xem giá trị.</p>'
    html += legend + ''.join(svg) + '<p>Nguồn: history.json snapshot 20261030T042721_825592Z_f68e06a0. Mỗi điểm là một validation; không làm trơn.</p>'
    (OUT / 'run_o6s4pex3_curves.html').write_text(html)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
