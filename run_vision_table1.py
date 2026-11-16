"""Score every CIFAR-10 and Food-101 table1 config.

Run from the repository root:

    uv run run_vision_table1.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ('cifar10', 'food101')


def configs():
    found = []
    for dataset in DATASETS:
        found.extend(sorted((ROOT / 'table1' / dataset).glob('*.json')))
    return found


def command(config: Path) -> list[str]:
    dataset = config.parent.name
    return [
        'uv', 'run', 'table1/run_table1.py',
        '--output-dir', f'vit_results/{dataset}/{config.stem}',
        '--config-file', f'{dataset}/{config.name}',
        '--data-subdir', dataset,
        '--plot', 'False',
    ]


def main():
    jobs = configs()
    if not jobs:
        raise SystemExit('No configs found under table1/cifar10 or table1/food101')

    failed = []
    for index, config in enumerate(jobs, start=1):
        cmd = command(config)
        print(f'[{index}/{len(jobs)}] {" ".join(cmd)}', flush=True)
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            failed.append((config, result.returncode))
            print(f'FAILED ({result.returncode}): {config.parent.name}/{config.name}', flush=True)

    if failed:
        print(f'{len(failed)} of {len(jobs)} configs failed:', flush=True)
        for config, code in failed:
            print(f'  {config.parent.name}/{config.name} (exit {code})', flush=True)
        sys.exit(1)
    print(f'Finished {len(jobs)} configs.', flush=True)


if __name__ == '__main__':
    main()
