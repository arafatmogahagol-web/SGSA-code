"""Verify release source hashes and parse Python files without starting training."""
from pathlib import Path
import ast
import hashlib
import json


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'provenance/packaged_hashes.json').read_text(encoding='utf-8'))
    parsed = 0
    for name, expected in manifest.items():
        path = root / name
        content = path.read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('Source hash mismatch: ' + name)
        if path.suffix == '.py':
            ast.parse(content.decode('utf-8-sig'), filename=name)
            parsed += 1
    print(f'Verified {len(manifest)} source files; parsed {parsed} Python files. No training started.')


if __name__ == '__main__':
    main()
