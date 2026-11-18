"""Stack table1.csv Avg columns within CIFAR-10 and within Food-101.

One row per method. One column per pair run. The model column is dropped.
Writes vit_results/<dataset>/table1.csv and vit_results/<dataset>/table1_average.csv
(each cell as %% of Raw for that column, e.g. 52.2/92.2 → 56.62).

Run from the repository root:

    uv run collate_vision_table1.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'vit_results'
DATASETS = ('cifar10', 'food101')


def mean_across_pairs(values: list[float]) -> float:
    return sum(values) / len(values)


def read_avgs(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    return [(row['method'], row['Avg']) for row in rows]


def write_table1_relative(
    dataset: str,
    columns: list[str],
    order: list[str],
    by_method: dict[str, dict[str, str]],
) -> Path:
    if 'Raw' not in by_method:
        raise ValueError(f'No Raw method in vit_results/{dataset}/table1.csv rows')

    raw_row = by_method['Raw']
    out = RESULTS / dataset / 'table1_average.csv'
    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'Avg', *columns])
        writer.writeheader()
        for method in order:
            row: dict[str, str] = {'method': method}
            ratio_pcts: list[float] = []
            for column in columns:
                raw_value = float(raw_row[column])
                if raw_value == 0:
                    raise ValueError(f'Raw is 0 for column {column!r} in {dataset}')
                ratio_pct = 100.0 * float(by_method[method][column]) / raw_value
                ratio_pcts.append(ratio_pct)
                row[column] = f'{ratio_pct:.2f}'
            row['Avg'] = f'{mean_across_pairs(ratio_pcts):.2f}'
            writer.writerow(row)

    return out


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
        writer = csv.DictWriter(f, fieldnames=['method', 'Avg', *columns])
        writer.writeheader()
        for method in order:
            pair_values = [float(by_method[method][column]) for column in columns]
            row = {
                'method': method,
                'Avg': f'{mean_across_pairs(pair_values):.1f}',
                **by_method[method],
            }
            writer.writerow(row)

    avg_out = write_table1_relative(dataset, columns, order, by_method)

    print(f'{dataset}: {len(columns)} columns -> {out.relative_to(ROOT)}', flush=True)
    print(f'{dataset}: relative (÷ Raw) -> {avg_out.relative_to(ROOT)}', flush=True)
    for column in columns:
        print(f'  {column}', flush=True)
    return out


def main():
    for dataset in DATASETS:
        collate(dataset)


if __name__ == '__main__':
    main()
