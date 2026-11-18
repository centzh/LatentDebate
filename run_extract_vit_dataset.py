"""Run extract_vit_probabilities.py for every vision pair in a dataset.

From the repository root:

    uv run run_extract_vit_dataset.py --dataset food101
    uv run run_extract_vit_dataset.py --dataset cifar10
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ('cifar10', 'food101', 'imagenet100', 'oxford_flowers')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dataset',
        choices=DATASETS,
        required=True,
        help='Dataset name (runs every config/<dataset>/*.yaml)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without running them',
    )
    return parser.parse_args()


def configs_for_dataset(dataset: str) -> list[Path]:
    config_dir = ROOT / 'config' / dataset
    if not config_dir.is_dir():
        raise SystemExit(f'No config directory: {config_dir}')

    paths = sorted(config_dir.glob('*.yaml'))
    if not paths:
        raise SystemExit(f'No *.yaml configs in {config_dir}')
    return paths


def command(config: Path) -> list[str]:
    rel = config.relative_to(ROOT)
    return [
        'uv',
        'run',
        'extract_vit_probabilities.py',
        '--config',
        str(rel),
    ]


def main() -> None:
    args = parse_args()
    jobs = configs_for_dataset(args.dataset)

    failed: list[tuple[Path, int]] = []
    for index, config in enumerate(jobs, start=1):
        cmd = command(config)
        print(f'[{index}/{len(jobs)}] {" ".join(cmd)}', flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            failed.append((config, result.returncode))
            print(f'FAILED ({result.returncode}): {config}', flush=True)

    if args.dry_run:
        print(f'Dry run: {len(jobs)} commands.', flush=True)
        return

    if failed:
        print(f'{len(failed)} of {len(jobs)} configs failed:', flush=True)
        for config, code in failed:
            print(f'  {config} (exit {code})', flush=True)
        sys.exit(1)
    print(f'Finished {len(jobs)} extractions for {args.dataset}.', flush=True)


if __name__ == '__main__':
    main()