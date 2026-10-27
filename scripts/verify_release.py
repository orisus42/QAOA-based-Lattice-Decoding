"""Verify published bytes against SHA256SUMS using only the Python standard library."""
from pathlib import Path
from hashlib import sha256
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    failures = []
    count = 0
    for line in (ROOT / 'reproducibility/SHA256SUMS').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        p = (ROOT / relative).resolve()
        if not p.is_relative_to(ROOT):
            raise ValueError('Manifest path outside repository')
        if not p.is_file() or sha256(p.read_bytes()).hexdigest() != expected:
            failures.append(relative)
        count += 1
    print(json.dumps({'files_checked': count, 'status': 'passed' if not failures else 'failed', 'mismatches': failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
