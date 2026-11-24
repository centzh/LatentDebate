"""Score ViT token-layer strengths with latent debate and write Table 1 rows.

Only numpy/scipy are required for scoring. Dataset images are loaded only when plotting.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import random
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

from plots import (
    plot_cls_patch_supporter_ablation,
    plot_prediction_agreements,
    plot_prediction_disagreements,
    plot_vit_ground_truth_mismatches,
)
from propagate import propagate

HERE = Path(__file__).resolve().parent
_VISION_DATASET_CACHE: dict[tuple, object] = {}
_LABEL_FEATURE_CACHE: dict[tuple, object] = {}


def dataset_label_feature(config: dict):
    """HF Features ClassLabel for mapping JSONL integer ``label`` to class names."""
    vd = config.get('vision_dataset')
    if not vd or not vd.get('hub_id'):
        return None
    hub_id = vd['hub_id']
    split = vd.get('split', 'test')
    cache_key = (hub_id, split)
    if cache_key in _LABEL_FEATURE_CACHE:
        return _LABEL_FEATURE_CACHE[cache_key]
    from datasets import load_dataset

    full_dataset = load_dataset(hub_id, split=split)
    _LABEL_FEATURE_CACHE[cache_key] = full_dataset.features['label']
    return _LABEL_FEATURE_CACHE[cache_key]


def ground_truth_name_from_record(record: dict, config: dict) -> str | None:
    label_feature = dataset_label_feature(config)
    if label_feature is None:
        return None
    raw_label = record.get('label')
    if raw_label is None:
        return None
    if isinstance(raw_label, str):
        return raw_label
    return label_feature.int2str(int(raw_label))


def load_vision_image_dataset(config: dict):
    """HF rows aligned with JSONL line index (same filter order as extract_vit_probabilities)."""
    from datasets import load_dataset

    vd = config.get('vision_dataset', {})
    hub_id = vd.get('hub_id', 'uoft-cs/cifar10')
    split = vd.get('split', 'test')
    include_classes = vd.get('include_classes') or [
        config['positive_class'],
        config['negative_class'],
    ]
    cache_key = (hub_id, split, tuple(include_classes))
    if cache_key in _VISION_DATASET_CACHE:
        return _VISION_DATASET_CACHE[cache_key]

    full_dataset = load_dataset(hub_id, split=split)
    allowed = {
        full_dataset.features['label'].str2int(name) for name in include_classes
    }
    labels = np.asarray(full_dataset['label'])
    indices = np.nonzero(np.isin(labels, list(allowed)))[0]
    dataset = full_dataset.select(indices)
    _VISION_DATASET_CACHE[cache_key] = dataset
    return dataset

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentage(count, n):
    return Decimal(count) * 100 / Decimal(n)


def display(value, places):
    quantized = Decimal(value).quantize(
        Decimal(1).scaleb(-places),
        rounding=ROUND_HALF_UP,
    )
    return format(quantized, 'f')


def map_answer_to_label(answer_raw: str, config: dict) -> int:
    positive = str(config['positive_class']).strip().lower()
    negative = str(config['negative_class']).strip().lower()
    answer = answer_raw.strip().lower()
    if answer == positive:
        return 1
    if answer == negative:
        return -1
    return 0


def load_input(path, config):
    records = [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(records) != config['samples_per_dataset']:
        raise ValueError(f'{path.name}: expected {config['samples_per_dataset']} records, found {len(records)}')

    raw = []
    answers = []
    answer_texts = []
    ground_truth = []
    ground_truth_texts = []

    for record in records:
        query = record['result']['query']
        p = np.asarray(query['query_probs'], dtype=np.float64)
        # Stored layout is (layers, tokens); scoring uses (tokens, layers).
        selected = p.T

        answer_raw = record['result']['answer']['answer']
        raw.append(selected)
        answers.append(map_answer_to_label(answer_raw, config))
        answer_texts.append(answer_raw.strip())

        gt_name = ground_truth_name_from_record(record, config)
        if gt_name is None:
            ground_truth_texts.append('')
            ground_truth.append(np.int8(0))
        else:
            ground_truth_texts.append(gt_name)
            ground_truth.append(map_answer_to_label(gt_name, config))

    raw = np.asarray(raw, dtype=np.float64)
    answers = np.asarray(answers, dtype=np.int8)
    ground_truth = np.asarray(ground_truth, dtype=np.int8)
    return raw, answers, answer_texts, ground_truth, ground_truth_texts


def _sign_from_cls(matrix: np.ndarray) -> np.ndarray:
    """Map CLS strength at the last layer to +1 / -1 / 0."""
    cls = matrix[:, 0, -1]
    return np.select([cls >= 0, cls < 0], [1, -1], default=0).astype(np.int8)


def predict_all(raw, config, output_dir: Path | None = None):
    rng = random.Random(config['random_seed'])
    n_tokens, n_layers = raw.shape[1], raw.shape[2]
    majority = (raw >= 0).sum(axis=(1, 2)) >= n_tokens * n_layers / 2

    random_pred = np.asarray(
        [1 if rng.choice(sample.ravel().tolist()) >= 0 else -1 for sample in raw],
        dtype=np.int8,
    )
    average_pred = np.where(raw.mean(axis=(1, 2)) >= 0, 1, -1).astype(np.int8)
    majority_pred = np.where(majority, 1, -1).astype(np.int8)

    propagated = propagate(raw)
    propagated_quadratic = propagate(raw, quadratic=True)
    propagated_cls_patch = propagate(raw, cls_patch_only=True)
    propagated_cls_only = propagate(raw, cls_only=True)
    propagated_weighted = propagate(raw, weighted=True)

    latent_debate = _sign_from_cls(propagated)
    cls_patch_only_pred = _sign_from_cls(propagated_cls_patch)
    cls_only_pred = _sign_from_cls(propagated_cls_only)
    weighted_pred = _sign_from_cls(propagated_weighted)
    # Raw is the unpropagated CLS token. Propagation changes a lot of these labels.
    unweighted = _sign_from_cls(raw)
    quadratic = _sign_from_cls(propagated_quadratic)
    
    by_method = {
        'Random': random_pred,
        'Average': average_pred,
        'Majority Voting': majority_pred,
        'Raw': unweighted,
        'Latent Debate -- w/o token weight': latent_debate,
        'Quadratic': quadratic,
        'CLS Patch only': cls_patch_only_pred,
        'CLS only': cls_only_pred,
        'Weighted': weighted_pred,
    }

    unknown = [m for m in config['methods'] if m not in by_method]
    if unknown:
        raise ValueError(f'Unsupported methods in config: {unknown}')
    return {m: by_method[m] for m in config['methods']}


def make_table(rows, config):
    datasets = [d['name'] for d in config['datasets']]
    table = []
    for model in config['models']:
        for method in config['methods']:
            cells = [
                next(
                    r for r in rows
                    if r['model'] == model['name']
                    and r['method'] == method
                    and r['dataset'] == ds
                )
                for ds in datasets
            ]
            mean = sum(percentage(c['consistent'], c['n']) for c in cells) / Decimal(len(cells))
            row = {'model': model['name'], 'method': method}
            row.update({
                r['dataset']: display(percentage(r['consistent'], r['n']), config['decimal_places'])
                for r in cells
            })
            row['Avg'] = display(mean, config['decimal_places'])
            total_n = sum(int(c['n']) for c in cells)
            disagree_zero = sum(int(c.get('disagree_llm_zero', 0)) for c in cells)
            disagree_nonzero = sum(int(c.get('disagree_llm_nonzero', 0)) for c in cells)
            row['disagree_pct_llm_0'] = display(
                percentage(disagree_zero, total_n),
                config['decimal_places'],
            )
            row['disagree_pct_otherwise'] = display(
                percentage(disagree_nonzero, total_n),
                config['decimal_places'],
            )
            table.append(row)
    return table


def str_to_bool(value):
    if value.lower() in ("true", "1", "yes"):
        return True
    elif value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Expected True or False")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=HERE.parent / 'data')
    parser.add_argument('--data-subdir', type=str, default='probability')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results')
    parser.add_argument('--config-file', type=str, default='config.json')
    parser.add_argument(
        '--plot',
        type=str_to_bool,
        default=False,
        help='Write vision diagnostic heatmaps (agreements, disagreements, per method)',
    )
    parser.add_argument(
        '--plot-method',
        type=str,
        default=None,
        metavar='METHOD',
        help=(
            'Only plot agreement/disagreement figures for this method '
            '(exact name from config "methods"). Implies plotting is enabled.'
        ),
    )
    parser.add_argument(
        '--plot-vit-gt-mismatch',
        type=str_to_bool,
        default=False,
        help=(
            'Plot instances where ViT head answer (JSONL result.answer) disagrees with '
            'dataset ground truth (JSONL label). Requires vision_dataset in config.'
        ),
    )
    return parser.parse_args()


def ablate_supporter_strengths(
    raw: np.ndarray,
    threshold: float,
    *,
    ablate_above: bool,
) -> np.ndarray:
    """Zero supporter strengths above or below a threshold (fresh copy each call)."""
    ablated = raw.copy()
    if ablate_above:
        ablated[ablated > threshold] = 0
    else:
        ablated[ablated < threshold] = 0
    return ablated


def cls_patch_supporter_ablation_metrics(
    raw: np.ndarray,
    threshold: float,
    *,
    ablate_above: bool,
    cls_patch_only_pred: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Flip rate on the ablated subset (positives or negatives) and CLS final strengths."""
    ablated_raw = ablate_supporter_strengths(raw, threshold, ablate_above=ablate_above)
    propagated = propagate(ablated_raw)
    ablated_pred = _sign_from_cls(propagated)
    if ablate_above:
        subset_mask = cls_patch_only_pred == 1
        keep_label = np.int8(1)
    else:
        subset_mask = cls_patch_only_pred == -1
        keep_label = np.int8(-1)
    ablated_on_subset = ablated_pred[subset_mask]
    n_subset = int(subset_mask.sum())
    if n_subset == 0:
        flip_rate = float('nan')
    else:
        flip_rate = 1.0 - float((ablated_on_subset == keep_label).sum()) / n_subset
    return flip_rate, propagated[:, 0, -1].copy()


def sweep_cls_patch_supporter_ablation(
    raw: np.ndarray,
    cls_patch_only_pred: np.ndarray,
    *,
    n_thresholds: int = 10,
) -> dict:
    positive_mask = cls_patch_only_pred == 1
    negative_mask = cls_patch_only_pred == -1

    def _append(
        bucket: dict,
        threshold: float,
        flip_rate: float,
        final_strength: np.ndarray,
        *,
        strength_key: str,
        subset_mask: np.ndarray,
    ) -> None:
        bucket['thresholds'].append(float(threshold))
        bucket['flip_rates'].append(flip_rate)
        if subset_mask.any():
            bucket[strength_key].append(float(final_strength[subset_mask].mean()))
        else:
            bucket[strength_key].append(float('nan'))

    sweep = {
        'ablate_above': {
            'thresholds': [],
            'flip_rates': [],
            'mean_final_strength_positives': [],
        },
        'ablate_below': {
            'thresholds': [],
            'flip_rates': [],
            'mean_final_strength_negatives': [],
        },
    }
    for threshold in np.linspace(0, 1, n_thresholds):
        flip_rate, final_strength = cls_patch_supporter_ablation_metrics(
            raw,
            float(threshold),
            ablate_above=True,
            cls_patch_only_pred=cls_patch_only_pred,
        )
        _append(
            sweep['ablate_above'],
            threshold,
            flip_rate,
            final_strength,
            strength_key='mean_final_strength_positives',
            subset_mask=positive_mask,
        )
    for threshold in np.linspace(0, -1, n_thresholds):
        flip_rate, final_strength = cls_patch_supporter_ablation_metrics(
            raw,
            float(threshold),
            ablate_above=False,
            cls_patch_only_pred=cls_patch_only_pred,
        )
        _append(
            sweep['ablate_below'],
            threshold,
            flip_rate,
            final_strength,
            strength_key='mean_final_strength_negatives',
            subset_mask=negative_mask,
        )
    return sweep


def analyze_cls_patch_supporter_ablation(
    raw: np.ndarray,
    output_dir: Path | None = None,
    *,
    quiet: bool = False,
) -> dict:
    cls_patch_only_strengths = propagate(raw, cls_patch_only=True)
    cls_patch_only_pred = _sign_from_cls(cls_patch_only_strengths)
    sweep = sweep_cls_patch_supporter_ablation(raw, cls_patch_only_pred)

    if not quiet:
        for mode, label in (
            ('ablate_above', 'ablate supporters > threshold'),
            ('ablate_below', 'ablate attackers < threshold'),
        ):
            for threshold, flip_rate in zip(
                sweep[mode]['thresholds'],
                sweep[mode]['flip_rates'],
                strict=True,
            ):
                print(
                    f'CLS patch only + {label}: threshold={threshold:.4f}, '
                    f'flip rate={flip_rate:.4f}',
                    flush=True,
                )

    if output_dir is not None:
        plot_cls_patch_supporter_ablation(sweep, output_dir)
    return sweep


def write_predictions(output_dir, model_name, dataset_name, llm, predictions):
    key = model_name.replace('-', '_') + '__' + dataset_name
    sample_rows = [
        {
            'index': i,
            'llm_prediction': int(llm[i]),
            **{method: int(values[i]) for method, values in predictions.items()},
        }
        for i in range(len(llm))
    ]
    write_csv(output_dir / 'predictions' / (key + '.csv'), sample_rows)


def consistency_rows(model_name, dataset_name, file, tokens, llm, predictions, config):
    rows = []
    for method, values in predictions.items():
        num_consistent = int((values == llm).sum())
        disagree = values != llm
        disagree_llm_zero = int((disagree & (llm == 0)).sum())
        disagree_llm_nonzero = int((disagree & (llm != 0)).sum())
        n_samples = len(llm)
        rows.append({
            'model': model_name,
            'dataset': dataset_name,
            'method': method,
            'n': n_samples,
            'consistent': num_consistent,
            'consistency_percent': float(percentage(num_consistent, len(llm))),
            'disagree_llm_zero': disagree_llm_zero,
            'disagree_llm_nonzero': disagree_llm_nonzero,
            'tokens': tokens,
            'layers': config['selected_layers'],
            'random_seed': config['random_seed'] if method == 'Random' else '',
            'input_file': file,
            'disagree_pct_llm_0': float(percentage(disagree_llm_zero, n_samples)),
            'disagree_pct_otherwise': float(percentage(disagree_llm_nonzero, n_samples)),
        })
    return rows


def _predictions_for_plots(
    predictions: dict[str, np.ndarray],
    plot_method: str | None,
) -> dict[str, np.ndarray]:
    if plot_method is None:
        return predictions
    if plot_method not in predictions:
        known = ', '.join(predictions)
        raise ValueError(f'Unknown --plot-method {plot_method!r}. Known methods: {known}')
    return {plot_method: predictions[plot_method]}


def score_input(
    path,
    file,
    model_name,
    dataset_name,
    tokens,
    config,
    output_dir,
    plot,
    *,
    plot_method: str | None = None,
    plot_vit_gt_mismatch: bool = False,
):
    raw, llm, answer_texts, ground_truth, ground_truth_texts = load_input(path, config)
    plot_comparisons = plot or plot_method is not None or plot_vit_gt_mismatch
    plot_output = output_dir if plot_comparisons else None
    vision_images = None
    if plot_output is not None:
        vision_images = load_vision_image_dataset(config)
    predictions = predict_all(raw, config, plot_output)
    analyze_cls_patch_supporter_ablation(raw, output_dir)
    if plot_output is not None:
        plot_predictions = _predictions_for_plots(predictions, plot_method)
        if plot or plot_method is not None:
            plot_prediction_agreements(
                raw,
                llm,
                plot_predictions,
                answer_texts,
                config,
                plot_output,
                vision_images,
            )
            plot_prediction_disagreements(
                raw,
                llm,
                plot_predictions,
                answer_texts,
                config,
                plot_output,
                vision_images,
            )
        if plot_vit_gt_mismatch:
            if dataset_label_feature(config) is None:
                raise ValueError(
                    '--plot-vit-gt-mismatch requires vision_dataset in config '
                    'to resolve JSONL label integers to class names',
                )
            plot_vit_ground_truth_mismatches(
                raw,
                llm,
                ground_truth,
                ground_truth_texts,
                plot_predictions,
                answer_texts,
                config,
                plot_output,
                vision_images,
            )
    write_predictions(output_dir, model_name, dataset_name, llm, predictions)
    print(
        f'{model_name} / {dataset_name}: n={len(llm)}, '
        f'tokens={tokens}, {len(config["methods"])} methods scored',
        flush=True,
    )
    return consistency_rows(model_name, dataset_name, file, tokens, llm, predictions, config)


def score_config(
    config,
    data_dir,
    output_dir,
    plot,
    *,
    plot_method: str | None = None,
    plot_vit_gt_mismatch: bool = False,
):
    rows = []
    inputs = []
    for model in config['models']:
        for dataset, file in zip(config['datasets'], model['files']):
            path = data_dir / file
            if not path.is_file():
                raise FileNotFoundError(f'Missing Table 1 input: {path}')
            rows.extend(score_input(
                path,
                file,
                model['name'],
                dataset['name'],
                dataset['tokens'],
                config,
                output_dir,
                plot,
                plot_method=plot_method,
                plot_vit_gt_mismatch=plot_vit_gt_mismatch,
            ))
            inputs.append({
                'model': model['name'],
                'dataset': dataset['name'],
                'file': file,
            })
    return rows, inputs


def write_result_tables(output_dir, rows, config):
    table = make_table(rows, config)
    write_csv(output_dir / 'table1.csv', table)
    write_csv(output_dir / 'table1_full_precision.csv', rows)
    mean_rows = []
    for row in table:
        cells = [r for r in rows if r['model'] == row['model'] and r['method'] == row['method']]
        mean = sum(percentage(r['consistent'], r['n']) for r in cells) / Decimal(len(cells))
        mean_rows.append({
            'model': row['model'],
            'method': row['method'],
            'mean_percent': float(mean),
            'display_mean': row['Avg'],
        })
    write_csv(output_dir / 'averages.csv', mean_rows)
    return table


def write_environment(output_dir):
    write_json(output_dir / 'environment.json', {
        'python': sys.version,
        'platform': platform.platform(),
        'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy']},
    })


def write_run_status(output_dir, config, rows, table, inputs, elapsed):
    write_json(output_dir / 'run_status.json', {
        'protocol_version': config['protocol_version'],
        'models': len(config['models']),
        'datasets_per_model': len(config['datasets']),
        'samples_per_input': config['samples_per_dataset'],
        'input_files': len(inputs),
        'dataset_cells': len(rows),
        'row_averages': len(table),
        'random_seed': config['random_seed'],
        'mean_rounding': 'arithmetic mean of unrounded scores, then ROUND_HALF_UP',
        'elapsed_seconds': elapsed,
        'inputs': inputs,
    })


def main():
    args = parse_args()
    start = time.time()
    config = read_json(HERE / args.config_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_method = args.plot_method
    if plot_method is not None and plot_method not in config['methods']:
        known = ', '.join(config['methods'])
        raise ValueError(f'Unknown --plot-method {plot_method!r}. Config methods: {known}')

    rows, inputs = score_config(
        config,
        args.data_dir / args.data_subdir,
        args.output_dir,
        args.plot,
        plot_method=plot_method,
        plot_vit_gt_mismatch=args.plot_vit_gt_mismatch,
    )
    table = write_result_tables(args.output_dir, rows, config)
    write_json(args.output_dir / 'config_used.json', config)
    write_environment(args.output_dir)
    write_run_status(args.output_dir, config, rows, table, inputs, time.time() - start)

    print(
        f'Wrote {len(rows)} cells and {len(table)} averages.',
        flush=True,
    )
    print('Results: ' + str(args.output_dir.resolve()), flush=True)


if __name__ == '__main__':
    main()
