"""Aggregate every finished run of one dataset into tables, figures and a report."""
from pathlib import Path

import pandas as pd

from models import NAMES
from utils.io import dump
from utils.params import parameter_counts
from utils.plotting import plot_mae_bars, plot_parameter_vs_mae

DISCLOSURE = """
## Reading this report

- Means and sample standard deviations use completed runs only. The count column
  shows the number of available seeds; a partial report is not a three-seed result.
- All errors are on the training-only z-score scale, not in physical units.
- Every method uses the same splits, history/horizon, valid-label masks and training
  budget. Baselines are adapted implementations under this protocol, not reproductions
  of the original papers' configurations (`MultiGSA_adapted` in particular).

"""


def report(dataset, cfg, dirs):
    results = Path(dirs['results'])
    rows = [__import__('json').loads(p.read_text(encoding='utf-8'))
            for p in results.glob('*/*/*/*/metrics.json')
            if (p.parent.parent / 'complete.json').exists()]
    if not rows:
        print('No finished runs yet', flush=True)
        return None
    frame = pd.DataFrame(rows)
    frame.to_csv(results / 'all_metrics.csv', index=False)
    summary = (frame.groupby(['model', 'strategy', 'condition'])[['mae', 'rmse']]
               .agg(['mean', 'std', 'count']))
    summary.to_csv(results / 'summary.csv')
    dump(results / 'summary.json',
         {'rows': len(frame),
          'complete_runs': len(list(results.glob('*/*/*/complete.json'))),
          'expected_runs': len(NAMES) * len(cfg['strategies']) * len(cfg['seeds'])})

    title = f"{dataset}: {frame.model.nunique()} models"
    plot_mae_bars(frame, results / 'mae_by_model', title)

    efficiency = results / 'efficiency' / 'efficiency_summary.json'
    if efficiency.exists():
        import json
        measured = {row['model']: row['parameters']
                    for row in json.loads(efficiency.read_text(encoding='utf-8'))}
        plot_parameter_vs_mae(frame, measured, results / 'parameters_vs_mae')

    lines = [f'# {dataset} benchmark', '',
             f"Completed runs: {len(list(results.glob('*/*/*/complete.json')))}/"
             f"{len(NAMES) * len(cfg['strategies']) * len(cfg['seeds'])}", '',
             summary.to_string(), DISCLOSURE]
    (results / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary
