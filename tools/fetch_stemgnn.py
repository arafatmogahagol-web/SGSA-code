"""Fetch the hash-checked upstream file locally; no upstream license is granted here."""
from pathlib import Path
import argparse,hashlib,json,urllib.request


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,help='Use an existing upstream base_model.py instead of downloading')
    args=parser.parse_args()
    lock=json.loads((root/'provenance/stemgnn_source.json').read_text())
    if args.source is not None:
        content=args.source.read_bytes()
    else:
        with urllib.request.urlopen(lock['raw_url'],timeout=60) as response:
            content=response.read()
    digest=hashlib.sha256(content).hexdigest()
    if digest!=lock['sha256']:
        raise ValueError('Upstream source hash changed. Do not silently replace the benchmark implementation.')
    path=root/'vendor/StemGNN/models/base_model.py'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(content)
    print('Verified upstream file installed locally:',path)
    print('This source remains subject to upstream rights; it is ignored by Git.')


if __name__=='__main__':main()
