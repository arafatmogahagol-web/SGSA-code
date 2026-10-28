"""Regenerate provenance/packaged_hashes.json after an intentional source edit.

Hashes are computed on CRLF-to-LF normalized bytes, matching tools/verify_sources.py,
so verification works across Windows and Linux checkouts.

Excluded: the manifest itself, Git internals, caches, dataset payloads, checkpoints,
generated results and the separately fetched StemGNN source.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'provenance' / 'packaged_hashes.json'
SKIP_DIRS = {'.git', '__pycache__', '.venv', '.idea', '.vscode', 'checkpoints', 'results'}
SKIP_SUFFIXES = {'.npz', '.pt', '.pth', '.zip', '.log', '.tmp', '.pyc'}
SKIP_NAMES = {'queue.lock', 'status.json', 'train_priors.npz', 'current_training.json'}
# Regenerated with a timestamp by tools/smoke_checks.py; pinning its hash would make
# verify_sources.py fail after every legitimate re-run of that check.
SKIP_PATHS = {Path('provenance/packaged_hashes.json'), Path('provenance/smoke_checks.json'),
              Path('vendor/StemGNN')}


def wanted(relative):
    if relative in SKIP_PATHS or relative.name in SKIP_NAMES:
        return False
    if relative.suffix in SKIP_SUFFIXES:
        return False
    if any(part in SKIP_DIRS for part in relative.parts[:-1]):
        return False
    if relative.parts[:2] == ('vendor', 'StemGNN'):
        return False
    # Dataset payloads stay out of the repository; configs and audit records are tracked.
    if len(relative.parts) > 3 and relative.parts[:2] == ('code', 'datasets'):
        if relative.parts[3] == 'data':
            return False
    return True


def main():
    entries = {}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if not wanted(relative):
            continue
        normalized = path.read_bytes().replace(b'\r\n', b'\n')
        entries[relative.as_posix()] = hashlib.sha256(normalized).hexdigest()
    MANIFEST.write_text(json.dumps(entries, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'Wrote {len(entries)} file hashes to {MANIFEST.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
