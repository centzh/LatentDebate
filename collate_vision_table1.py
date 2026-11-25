"""Stack per-pair Table 1 scores into one CSV per dataset.

Columns are the unweighted average, the weighted average, then each class
pair with its unweighted column beside its weighted column. Each average is
computed inside its own half. ``table1.csv`` is absolute agreement with the
ViT head. ``table1_average.csv`` is 100 * method / Raw on that column.

    uv run collate_vision_table1.py
    uv run collate_vision_table1.py --dataset food101
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'vit_results'
DATASETS = ('cifar10', 'food101')
EXCLUDED_PAIR_DIRS = frozenset({'dumpllings_gyoza'})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, default=None)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def read_avgs(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    return [(row['method'], row['Avg']) for row in rows]


def split_halves(names: list[str]) -> tuple[list[str], list[str]]:
    unweighted = sorted(name for name in names if not name.endswith('_weighted'))
    weighted_stems = sorted(
        name.removesuffix('_weighted')
        for name in names
        if name.endswith('_weighted')
    )
    weighted = [f'{stem}_weighted' for stem in weighted_stems]
    return unweighted, weighted


def pair_columns(unweighted: list[str], weighted: list[str]) -> list[str]:
    """Each class, unweighted column then weighted column."""
    unweighted_set = set(unweighted)
    weighted_set = set(weighted)
    stems = sorted({*unweighted, *(name.removesuffix('_weighted') for name in weighted)})
    columns: list[str] = []
    for stem in stems:
        if stem in unweighted_set:
            columns.append(stem)
        weighted_name = f'{stem}_weighted'
        if weighted_name in weighted_set:
            columns.append(weighted_name)
    return columns


def format_half_avg(values: list[float], places: int) -> str:
    if not values:
        return ''
    return f'{mean(values):.{places}f}'


def write_tables(
    dataset: str,
    unweighted: list[str],
    weighted: list[str],
    order: list[str],
    by_method: dict[str, dict[str, str]],
) -> None:
    columns = pair_columns(unweighted, weighted)
    fieldnames = ['method', 'avg_unweighted', 'avg_weighted', *columns]

    absolute = RESULTS / dataset / 'table1.csv'
    with absolute.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in order:
            cells = by_method[method]
            row = {'method': method}
            for column in columns:
                row[column] = cells.get(column, '')
            row['avg_unweighted'] = format_half_avg(
                [float(cells[c]) for c in unweighted if cells.get(c)],
                1,
            )
            row['avg_weighted'] = format_half_avg(
                [float(cells[c]) for c in weighted if cells.get(c)],
                1,
            )
            writer.writerow(row)

    if 'Raw' not in by_method:
        raise ValueError(f'No Raw method under vit_results/{dataset}')
    raw_row = by_method['Raw']
    relative = RESULTS / dataset / 'table1_average.csv'
    with relative.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in order:
            cells = by_method[method]
            ratios: dict[str, float] = {}
            row = {'method': method}
            for column in columns:
                raw_text = raw_row.get(column, '')
                method_text = cells.get(column, '')
                if raw_text == '' or method_text == '':
                    row[column] = ''
                    continue
                raw_value = float(raw_text)
                if raw_value == 0:
                    raise ValueError(f'Raw is 0 for {dataset}/{column}')
                ratios[column] = 100.0 * float(method_text) / raw_value
                row[column] = f'{ratios[column]:.2f}'
            row['avg_unweighted'] = format_half_avg(
                [ratios[c] for c in unweighted if c in ratios],
                2,
            )
            row['avg_weighted'] = format_half_avg(
                [ratios[c] for c in weighted if c in ratios],
                2,
            )
            writer.writerow(row)

    print(
        f'Updated {absolute.relative_to(ROOT)} and {relative.relative_to(ROOT)} '
        f'({len(unweighted)} unweighted pairs, {len(weighted)} weighted pairs)',
        flush=True,
    )


def collate(dataset: str) -> None:
    paths = [
        path
        for path in sorted((RESULTS / dataset).glob('*/table1.csv'))
        if path.parent.name not in EXCLUDED_PAIR_DIRS
    ]
    if not paths:
        raise FileNotFoundError(f'No table1.csv files under vit_results/{dataset}')

    names = [path.parent.name for path in paths]
    unweighted, weighted = split_halves(names)
    by_method: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for path, column in zip(paths, names):
        for method, avg in read_avgs(path):
            if method not in by_method:
                order.append(method)
                by_method[method] = {}
            by_method[method][column] = avg
    write_tables(dataset, unweighted, weighted, order, by_method)


def main() -> None:
    args = parse_args()
    datasets = (args.dataset,) if args.dataset else DATASETS
    for dataset in datasets:
        collate(dataset)


if __name__ == '__main__':
    main()