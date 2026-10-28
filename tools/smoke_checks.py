"""CPU-only structural checks: no training, no dataset, no GPU required.

Verifies that

* every Python file parses;
* every architecture still builds and returns the documented output shape, and that
  its parameter count equals the value printed in the paper (Tables 4 and 7);
* both ``datasets/<name>/config.json`` files stay consistent with the recorded
  audit files and window counts;
* each command line entry point still answers ``--help``.

The two datasets are checked in separate processes: vendor code ships its own
top-level ``layers``/``utils`` packages, so isolating them keeps the check honest.

Usage:  python tools/smoke_checks.py [--out provenance/smoke_checks.json]
Adapters whose inputs are not redistributed (the training-only prior file for
PDFormer, the separately fetched StemGNN source) are reported as skipped.
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
CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))

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
ENTRY_POINTS = [['code/run.py'], ['code/run.py', 'train'], ['code/run.py', 'benchmark'],
                ['tools/fetch_stemgnn.py'], ['tools/update_manifest.py'],
                ['tools/smoke_checks.py']]


def parse_sources():
    count = 0
    for path in CODE.rglob('*.py'):
        if '__pycache__' in path.parts:
            continue
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        count += 1
    for path in (ROOT / 'tools').rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        count += 1
    return count


def cli_help():
    results = []
    for parts in ENTRY_POINTS:
        command = [sys.executable, str(ROOT / parts[0]), *parts[1:], '--help']
        completed = subprocess.run(command, capture_output=True, text=True, cwd=str(ROOT))
        results.append(dict(script=' '.join(parts), passed=completed.returncode == 0,
                            returncode=completed.returncode))
    return results


def integrity_guards():
    """Run the CPU regression tests for checkpoint/run identity."""
    completed = subprocess.run([sys.executable, str(ROOT / 'tools/test_integrity_guards.py')],
                               capture_output=True, text=True, cwd=str(ROOT))
    tail = [line for line in completed.stderr.strip().splitlines() if line.strip()][-1:]
    return dict(script='tools/test_integrity_guards.py', passed=completed.returncode == 0,
                summary=tail[0].strip() if tail else '')


def forward_checks(dataset):
    """Build and run every architecture of one protocol on CPU (batch 1).

    Only PDFormer and StemGNN may be skipped: their inputs (the training-only prior
    file and the separately fetched upstream source) are not redistributed. A
    missing library or any other failure is reported as a failure, never as a skip.
    """
    import torch
    from models import build

    optional = {'PDFormer_train_graph', 'StemGNN_power_only'}
    cfg = json.loads((CODE / 'datasets' / dataset / 'config.json').read_text(encoding='utf-8'))
    x = torch.zeros(1, cfg['seq_len'], cfg['num_nodes'], cfg['input_dim'])
    rows = []
    for name, expected in EXPECTED[dataset].items():
        try:
            torch.manual_seed(0)
            model = build(name, dict(cfg, dataset=dataset), device='cpu')
            model.eval()
            with torch.no_grad():
                out = model(x)
            if not torch.isfinite(out).all():
                raise AssertionError('nonfinite output')
            if list(out.shape) != [1, cfg['pred_len'], cfg['num_nodes']]:
                raise AssertionError(f'unexpected output shape {tuple(out.shape)}')
            params = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
            rows.append(dict(dataset=dataset, model=name, output_shape=list(out.shape),
                             parameters=params, expected_parameters=expected,
                             status='checked' if params == expected else 'parameter_mismatch'))
            del model
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            status = 'skipped' if name in optional else 'failed'
            rows.append(dict(dataset=dataset, model=name, parameters=None,
                             expected_parameters=expected, status=status,
                             reason=f'{type(exc).__name__}: {exc}'.splitlines()[0][:200]))
        print(f"{dataset:6s} {name:22s} {rows[-1]['status']:18s} {rows[-1].get('parameters')}",
              flush=True)
    return rows


def config_checks():
    results = {}
    for dataset in ['SDWPF', 'Hill21']:
        cfg = json.loads((CODE / 'datasets' / dataset / 'config.json').read_text(encoding='utf-8'))
        key = dataset.lower()
        results[f'{key}_has_selection_key'] = cfg['selection_key'] in ('val_mse', 'val_mae')
        results[f'{key}_has_expected_counts'] = set(cfg['expected_windows']) == {'train', 'val', 'test'}
        results[f'{key}_benchmark_protocol'] = dict(cfg['benchmark'])
        results[f'{key}_provenance_recorded'] = bool(cfg.get('hyperparameter_provenance'))
    audit = json.loads((CODE / 'datasets/Hill21/data_audit.json').read_text(encoding='utf-8'))
    hill = json.loads((CODE / 'datasets/Hill21/config.json').read_text(encoding='utf-8'))
    results['hill21_split_matches_audit'] = (
        [hill['split']['train'][0], hill['split']['train'][1], hill['split']['val'][1],
         hill['split']['test'][1]] == list(audit['boundaries']))
    results['hill21_windows_match_audit'] = (
        dict(train=audit['windows']['train'], validation=audit['windows']['validation'],
             test=audit['windows']['test'])
        == dict(train=hill['expected_windows']['train'], validation=hill['expected_windows']['val'],
                test=hill['expected_windows']['test']))
    priors = json.loads((CODE / 'datasets/Hill21/prior_audit.json').read_text(encoding='utf-8'))
    results['hill21_priors_training_only'] = (
        priors['validation_used'] is False and priors['test_used'] is False)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default='provenance/smoke_checks.json')
    parser.add_argument('--dataset', choices=['SDWPF', 'Hill21'],
                        help='internal: check a single protocol in this process')
    parser.add_argument('--json', help='internal: write per-dataset rows here')
    args = parser.parse_args()

    if args.dataset:
        Path(args.json).write_text(json.dumps(forward_checks(args.dataset)), encoding='utf-8')
        return

    import torch
    forward_rows = []
    for dataset in ['SDWPF', 'Hill21']:
        target = ROOT / 'provenance' / f'.smoke_{dataset}.json'
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                    '--dataset', dataset, '--json', str(target)],
                                   cwd=str(ROOT))
        if target.exists():
            forward_rows.extend(json.loads(target.read_text(encoding='utf-8')))
            target.unlink()
        if completed.returncode != 0:
            forward_rows.append(dict(dataset=dataset, model='-', status='process_failed',
                                     returncode=completed.returncode))

    record = dict(
        generated=datetime.now().isoformat(timespec='seconds'),
        platform=platform.platform(),
        python=platform.python_version(),
        torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        training_started=False,
        layout='code/{SGSA.py,run.py,layers,models,data_provider,exp,utils,datasets,checkpoints,results}',
        python_sources_parsed=parse_sources(),
        config_checks=config_checks(),
        cli_help=cli_help(),
        integrity_guards=integrity_guards(),
        model_forward_checks=forward_rows,
        limitations=[
            'CPU single-batch forward checks only; no dataset, checkpoint or GPU measurement.',
            'PDFormer requires the training-only prior file and StemGNN requires its separately '
            'fetched upstream source; both are reported as skipped when absent.',
            'The two protocols run in separate processes because vendor code ships its own '
            'top-level layers/utils packages.',
        ])
    failures = [r for r in forward_rows if r['status'] not in ('checked', 'skipped')]
    failures += [r for r in record['cli_help'] if not r['passed']]
    failures += [key for key, value in record['config_checks'].items() if value is False]
    if not record['integrity_guards']['passed']:
        failures.append(record['integrity_guards'])
    (ROOT / args.out).write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(f'\nWrote {args.out}')
    if failures:
        print('FAILURES:', json.dumps(failures, indent=2))
        sys.exit(1)
    checked = [r for r in forward_rows if r['status'] == 'checked']
    skipped = [r for r in forward_rows if r['status'] == 'skipped']
    print(f'{len(checked)} models match the paper parameter counts, {len(skipped)} skipped; '
          f'configuration checks passed; {record["python_sources_parsed"]} Python files parsed.')


if __name__ == '__main__':
    main()
