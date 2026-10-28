"""CPU-only structural checks: no training, no dataset, no GPU required.

Verifies that every architecture still builds and returns the documented output
shape, that the parameter counts equal the values reported in the paper, that the
two configuration files agree with the preprocessing split and the Hill21 audit,
and that every entry point still answers ``--help``.

The two protocols are checked in separate processes on purpose: both ship a module
named ``adapters_extended``, so importing them into one interpreter could compare a
model against the wrong protocol's adapter.

Usage:  python tools/smoke_checks.py [--out provenance/smoke_checks.json]
Adapters whose inputs are not redistributed (graph priors, StemGNN upstream source)
are reported as skipped with the reason instead of failing.
"""
import argparse
import ast
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Parameter counts printed in the paper (Tables 4 and 7), real-valued scalars.
EXPECTED = {
    'SDWPF': {'SGSA': 23216, 'MultiGSA_adapted': 34033, 'LSTM': 75056,
              'Transformer': 81584, 'MTGNN': 5264032, 'FEDformer': 225040,
              'GraphWaveNet': 764520, 'AGCRN': 807020, 'STID_no_calendar': 163184,
              'DLinear': 9312, 'PatchTST': 105746, 'PDFormer_train_graph': 444017,
              'StemGNN_power_only': 67496332},
    'Hill21': {'SGSA': 19024, 'MultiGSA_adapted': 29841, 'LSTM': 74480,
               'Transformer': 81008, 'MTGNN': 1988112, 'FEDformer': 219271,
               'GraphWaveNet': 761684, 'AGCRN': 771330, 'STID_no_calendar': 104272,
               'DLinear': 9312, 'PatchTST': 105728, 'PDFormer_train_graph': 443441,
               'StemGNN_power_only': 67410339},
}
SHAPE = {'SDWPF': (96, 134, 16, 48), 'Hill21': (96, 21, 7, 48)}
CONFIG = {'SDWPF': 'experiments/SDWPF/config.json', 'Hill21': 'experiments/Hill21/config.json'}
ENTRY_POINTS = ['experiments/SDWPF/code/run_suite.py',
                'experiments/SDWPF/code/sgsa_train.py',
                'experiments/SDWPF/code/sgsa_evaluate.py',
                'experiments/Hill21/code/worker.py',
                'tools/fetch_stemgnn.py',
                'tools/update_manifest.py',
                'tools/smoke_checks.py']


def parse_sources():
    count = 0
    for path in ROOT.rglob('*.py'):
        if '.git' in path.parts:
            continue
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        count += 1
    return count


def cli_help():
    results = []
    for script in ENTRY_POINTS:
        completed = subprocess.run([sys.executable, str(ROOT / script), '--help'],
                                   capture_output=True, text=True, cwd=str(ROOT))
        results.append(dict(script=script, passed=completed.returncode == 0,
                            returncode=completed.returncode))
    return results


def forward_checks(dataset):
    """Build and run every architecture of one protocol on CPU (batch 1)."""
    import torch
    code_dir = ROOT / 'experiments' / ('SDWPF' if dataset == 'SDWPF' else 'Hill21') / 'code'
    sys.path.insert(0, str(code_dir))
    cfg = json.loads((ROOT / CONFIG[dataset]).read_text(encoding='utf-8'))
    seq, nodes, feats, _ = SHAPE[dataset]
    x = torch.zeros(1, seq, nodes, feats)
    rows = []
    torch.manual_seed(0)
    for name, expected in EXPECTED[dataset].items():
        try:
            if dataset == 'SDWPF':
                import models
                model = models.build(name, cfg, device='cpu')
            else:
                import worker
                worker.validate_protocol(cfg)
                model = worker.build(name, nodes, feats, cfg['dropout'], cfg['rho'], device='cpu')
            model.eval()
            with torch.no_grad():
                out = model(x)
            params = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
            shape_ok = tuple(out.shape) == (1, SHAPE[dataset][3], nodes)
            finite = bool(torch.isfinite(out).all())
            rows.append(dict(dataset=dataset, model=name, output_shape=list(out.shape),
                             output_shape_matches=shape_ok, finite_output=finite,
                             parameters=params, expected_parameters=expected,
                             status=('output_mismatch' if not shape_ok or not finite else
                                     'checked' if params == expected else 'parameter_mismatch')))
            del model
        except (FileNotFoundError, ImportError) as exc:
            # Missing optional priors/upstream source are the only expected skips.
            # A missing dependency for a core model must fail the check.
            optional = name in ('PDFormer_train_graph', 'StemGNN_power_only')
            rows.append(dict(dataset=dataset, model=name, parameters=None,
                             expected_parameters=expected, status='skipped' if optional else 'failed',
                             reason=f'{type(exc).__name__}: {exc}'.splitlines()[0][:200]))
        print(f"{dataset:6s} {name:22s} {rows[-1]['status']:18s} {rows[-1].get('parameters')}",
              flush=True)
    return rows


def config_checks():
    import importlib
    sys.path.insert(0, str(ROOT / 'experiments/SDWPF/code'))
    preprocessing = importlib.import_module('preprocessing')
    sdwpf = json.loads((ROOT / CONFIG['SDWPF']).read_text(encoding='utf-8'))
    hill = json.loads((ROOT / CONFIG['Hill21']).read_text(encoding='utf-8'))
    audit = json.loads((ROOT / 'experiments/Hill21/data_audit.json').read_text(encoding='utf-8'))
    return dict(
        sdwpf_split_matches_preprocessing=(
            {k: list(v) for k, v in preprocessing.PROTOCOL_SPLIT.items()} == sdwpf['split']),
        hill21_split_matches_audit=(list(hill['split']) == list(audit['boundaries'])),
        hill21_window_counts_match=(audit['windows'] == dict(train=729, validation=26, test=26)),
        hill21_window_counts=audit['windows'],
        hill21_expected_windows=dict(train=729, validation=26, test=26),
        hill21_expected_valid_test_points=26187,
        sdwpf_expected_valid_test_points=966746,
        sdwpf_expected_windows=dict(train=3465, val=40, test=165))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='provenance/smoke_checks.json')
    parser.add_argument('--dataset', choices=['SDWPF', 'Hill21'],
                        help='internal: check a single protocol in this process')
    parser.add_argument('--json', help='internal: write per-dataset rows here')
    args = parser.parse_args()

    if args.dataset:
        rows = forward_checks(args.dataset)
        Path(args.json).write_text(json.dumps(rows), encoding='utf-8')
        return

    import torch
    forward_rows = []
    for dataset in ['SDWPF', 'Hill21']:
        target = ROOT / 'provenance' / f'.smoke_{dataset}.json'
        target.unlink(missing_ok=True)
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                    '--dataset', dataset, '--json', str(target)],
                                   cwd=str(ROOT))
        if target.exists():
            forward_rows.extend(json.loads(target.read_text(encoding='utf-8')))
            target.unlink()
        if completed.returncode != 0:
            forward_rows.append(dict(dataset=dataset, status='process_failed',
                                     returncode=completed.returncode))

    record = dict(
        generated=datetime.now().isoformat(timespec='seconds'),
        platform=platform.platform(),
        python=platform.python_version(),
        torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        training_started=False,
        python_sources_parsed=parse_sources(),
        config_checks=config_checks(),
        cli_help=cli_help(),
        model_forward_checks=forward_rows,
        limitations=[
            'CPU single-batch forward checks only; no dataset, checkpoint or GPU measurement.',
            'PDFormer requires the training-only prior file and StemGNN requires its separately '
            'fetched upstream source; both are reported as skipped when absent.',
            'The two protocols run in separate processes because they share adapter module names.',
        ])
    failures = [r for r in forward_rows if r['status'] not in ('checked', 'skipped')]
    failures += [r for r in record['cli_help'] if not r['passed']]
    failures += [k for k, v in record['config_checks'].items() if isinstance(v, bool) and not v]
    (ROOT / args.out).write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(f"\nWrote {args.out}")
    if failures:
        print('FAILURES:', json.dumps(failures, indent=2))
        sys.exit(1)
    checked = [r for r in forward_rows if r['status'] == 'checked']
    skipped = [r for r in forward_rows if r['status'] == 'skipped']
    print(f"{len(checked)} models match the paper parameter counts, {len(skipped)} skipped; "
          f"{record['config_checks'] and 'config checks passed'}; "
          f"{record['python_sources_parsed']} Python files parsed.")


if __name__ == '__main__':
    main()
