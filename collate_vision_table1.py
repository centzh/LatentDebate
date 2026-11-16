"""Stack table1.csv Avg columns within CIFAR-10 and within Food-101.

One row per method. One column per pair run. The model column is dropped.
Writes vit_results/<dataset>/table1.csv.

Run from the repository root:

    uv run collate_vision_table1.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'vit_results'
DATASETS = ('cifar10', 'food101')


def read_avgs(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    return [(row['method'], row['Avg']) for row in rows]


def collate(dataset: str) -> Path:
    paths = sorted(p for p in (RESULTS / dataset).glob('*/table1.csv'))
    if not paths:
        raise FileNotFoundError(f'No table1.csv files under vit_results/{dataset}')

    columns = [path.parent.name for path in paths]
    by_method: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for path, column in zip(paths, columns):
        for method, avg in read_avgs(path):
            if method not in by_method:
                order.append(method)
                by_method[method] = {}
            by_method[method][column] = avg

    out = RESULTS / dataset / 'table1.csv'
    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', *columns])
        writer.writeheader()
        for method in order:
            writer.writerow({'method': method, **by_method[method]})

    print(f'{dataset}: {len(columns)} columns -> {out.relative_to(ROOT)}', flush=True)
    for column in columns:
        print(f'  {column}', flush=True)
    return out


def main():
    for dataset in DATASETS:
        collate(dataset)


if __name__ == '__main__':
    main()
