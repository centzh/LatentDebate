"""Extract ViT probabilities and score Table 1.

extract writes the JSONL. score runs Table 1 from that JSONL.
run does both: extract, then score.

plot writes agreement, disagreement, and ViT-vs-ground-truth figures.
Add --compare to draw unweighted and weighted outputs on the same figure.

One pair (no plots):

    uv run vision.py extract food101 dumplings_gyoza
    uv run vision.py score food101 dumplings_gyoza
    uv run vision.py run food101 dumplings_gyoza          # extract, then score

Same pair, attention-weighted extract:

    uv run vision.py extract food101 dumplings_gyoza --weighted
    uv run vision.py score food101 dumplings_gyoza --weighted
    uv run vision.py run food101 dumplings_gyoza --weighted   # extract, then score

Every pair in a dataset. Add --weighted for the *_weighted configs only:

    uv run vision.py extract food101
    uv run vision.py score food101
    uv run vision.py run food101          # extract, then score every pair
    uv run vision.py run food101 --weighted   # extract, then score every weighted pair

score and run refresh vit_results/<dataset>/table1.csv and table1_average.csv.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from collate_vision_table1 import collate

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('dataset', help='e.g. food101 or cifar10')
    common.add_argument(
        'pair',
        nargs='?',
        help='class pair, e.g. dumplings_gyoza (omit to run every pair)',
    )
    common.add_argument(
        '--weighted',
        action='store_true',
        help='use the *_weighted config (attention-weighted strengths)',
    )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('extract', parents=[common], help='write JSONL strengths only')
    commands.add_parser('score', parents=[common], help='score Table 1 from existing JSONL')
    commands.add_parser('run', parents=[common], help='extract probabilities, then score Table 1')
    plot = commands.add_parser('plot', parents=[common], help='plot agreements, disagreements, and mismatches')
    plot.add_argument(
        '--compare',
        action='store_true',
        help='one figure with raw, unweighted output, and weighted output',
    )
    return parser.parse_args()


def stem_for(pair: str, weighted: bool) -> str:
    base = pair.removesuffix('_weighted')
    if weighted or pair.endswith('_weighted'):
        return f'{base}_weighted'
    return base


def pairs_for(dataset: str, weighted: bool) -> list[str]:
    config_dir = ROOT / 'config' / 'vit' / dataset
    if not config_dir.is_dir():
        raise SystemExit(f'No configs in {config_dir}')
    if weighted:
        found = sorted(path.stem.removesuffix('_weighted') for path in config_dir.glob('*_weighted.yaml'))
    else:
        found = sorted(
            path.stem
            for path in config_dir.glob('*.yaml')
            if not path.stem.endswith('_weighted')
        )
    if not found:
        label = 'weighted ' if weighted else ''
        raise SystemExit(f'No {label}configs in {config_dir}')
    return found


def selected_stems(dataset: str, pair: str | None, weighted: bool) -> list[str]:
    if pair:
        return [stem_for(pair, weighted)]
    return [stem_for(name, weighted) for name in pairs_for(dataset, weighted)]


def run_command(command: list[str]) -> None:
    print(' '.join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def extract_one(dataset: str, stem: str) -> None:
    config = ROOT / 'config' / 'vit' / dataset / f'{stem}.yaml'
    if not config.is_file():
        raise SystemExit(f'Missing extract config: {config}')
    run_command(['uv', 'run', 'extract_vit_probabilities.py', '--config', str(config.relative_to(ROOT))])


def score_one(dataset: str, stem: str) -> None:
    config = ROOT / 'table1' / 'vit' / dataset / f'{stem}.json'
    if not config.is_file():
        raise SystemExit(f'Missing Table 1 config: {config}')
    run_command([
        'uv', 'run', 'table1/run_table1.py',
        '--config-file', f'vit/{dataset}/{stem}.json',
        '--data-dir', 'data',
        '--data-subdir', f'vit/{dataset}',
        '--output-dir', f'vit_results/{dataset}/{stem}',
        '--plot', 'False',
    ])


def table1_api():
    import sys

    table1_dir = str(ROOT / 'table1')
    if table1_dir not in sys.path:
        sys.path.insert(0, table1_dir)
    import plots
    import run_table1
    return run_table1, plots


def plot_stem(dataset: str, stem: str, *, compare: bool) -> None:
    run_table1, plots = table1_api()
    config = run_table1.read_json(ROOT / 'table1' / 'vit' / dataset / f'{stem}.json')
    raw, llm, answer_texts, ground_truth, ground_truth_texts, weights = run_table1.load_input(
        ROOT / 'data' / 'vit' / dataset / f'{stem}.jsonl',
        config,
    )
    if compare:
        weighted_config = run_table1.read_json(
            ROOT / 'table1' / 'vit' / dataset / f'{stem}_weighted.json'
        )
        _, _, _, _, _, weights = run_table1.load_input(
            ROOT / 'data' / 'vit' / dataset / f'{stem}_weighted.jsonl',
            weighted_config,
        )
        if weights is None:
            raise SystemExit(
                f'{stem}_weighted.jsonl has no weights. Re-extract with --weighted first.'
            )
        if weights.shape != raw.shape:
            raise SystemExit(
                f'Weight shape {weights.shape} does not match strengths {raw.shape} for {stem}'
            )
        predictions = run_table1.predict_all(raw, config, None, None)
        output_dir = ROOT / 'vit_results' / dataset / f'{stem}_compared'
    else:
        predictions = run_table1.predict_all(raw, config, None, weights)
        output_dir = ROOT / 'vit_results' / dataset / stem

    output_dir.mkdir(parents=True, exist_ok=True)
    images = run_table1.load_vision_image_dataset(config)
    print(f'Plotting {output_dir.relative_to(ROOT)}', flush=True)
    plots.plot_prediction_agreements(
        raw, llm, predictions, answer_texts, config, output_dir, images,
        weights=weights, compare=compare,
    )
    plots.plot_prediction_disagreements(
        raw, llm, predictions, answer_texts, config, output_dir, images,
        weights=weights, compare=compare,
    )
    plots.plot_vit_ground_truth_mismatches(
        raw, llm, ground_truth, ground_truth_texts, predictions, answer_texts,
        config, output_dir, images, weights=weights, compare=compare,
    )


def main() -> None:
    args = parse_args()
    if args.command == 'plot':
        if getattr(args, 'compare', False) and args.weighted:
            raise SystemExit('Use plot --compare or plot --weighted, not both.')
        stems = selected_stems(args.dataset, args.pair, False if args.compare else args.weighted)
        for stem in stems:
            plot_stem(args.dataset, stem, compare=args.compare)
        return

    stems = selected_stems(args.dataset, args.pair, args.weighted)

    if args.command in ('extract', 'run'):
        for stem in stems:
            extract_one(args.dataset, stem)

    if args.command in ('score', 'run'):
        for stem in stems:
            score_one(args.dataset, stem)
        collate(args.dataset)


if __name__ == '__main__':
    main()