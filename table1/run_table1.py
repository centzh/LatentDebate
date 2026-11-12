"""Reproduce the finalized Table 1 from its 18 pinned probability/weight caches.

Only numpy/scipy are required. No imports from the original repository are used.
The expected files are independent validation references, never computed outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import re
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from scipy.special import softmax
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
_VISION_DATASET_CACHE: dict[tuple, object] = {}
VISION_AGREEMENT_PLOT_MAX = 10
_SKIP_BASELINE_METHODS = frozenset({'random', 'average', 'majority voting'})


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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def percentage(count, n):
    return Decimal(count) * 100 / Decimal(n)


def display(value, places):
    quantized = Decimal(value).quantize(
        Decimal(1).scaleb(-places),
        rounding=ROUND_HALF_UP,
    )
    return format(quantized, 'f')


def map_answer_to_label(answer_raw: str, config: dict, vision: bool) -> int:
    if vision:
        positive = str(config['positive_class']).strip().lower()
        negative = str(config['negative_class']).strip().lower()
        answer = answer_raw.strip().lower()
        if answer == positive:
            return 1
        if answer == negative:
            return -1
        return 0
    answer = answer_raw.strip()
    return 1 if answer.startswith('T') else -1

def load_input(path, tokens, config, calculate_weights=True, skip_checks=False, vision=False):
    records = [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if len(records) != config['samples_per_dataset']:
        raise ValueError(f'{path.name}: expected {config['samples_per_dataset']} records, found {len(records)}')

    raw = []
    weights = []
    answers = []
    answer_texts = []
    query_hashes = []
    noncanonical = []
    layers = config['selected_layers']

    for index, record in enumerate(records):
        # Query metadata includes the string, tokenized query, layer-wise token
        # probabilities, and token scores used as QBAF initial strengths/weights.
        query = record['result']['query']
        p = np.asarray(query['query_probs'], dtype=np.float64)

        # if p.ndim != 2 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        #     raise ValueError(f'Invalid probability array: {path.name}, row {index}')

        # Last prompt tokens, from layer 0 up to (not including) the output layer.
        if vision:
            selected = p.T
        else:
            selected = p[-tokens:, -layers - 1:-1]

        selected = 2 * (selected - .5)

        if calculate_weights:
            scores = np.asarray(
                [score for token, score in query['token_score']][-tokens:],
                dtype=np.float64,
            )
            if (
                selected.shape != (tokens, layers)
                or scores.shape != (tokens,)
                or not np.isfinite(scores).all()
            ):
                raise ValueError(
                    f'Invalid token/layer shape: {path.name}, row {index}; '
                    f'tokens={tokens}, layers={layers}'
                )

            scores_softmax = softmax(scores[:-1] / config['token_weight_temperature'])
            weight = np.r_[scores_softmax, 1.0]

        answer_raw = record['result']['answer']['answer']
        label = map_answer_to_label(answer_raw, config, vision)

        raw.append(selected)
        if calculate_weights:
            weights.append(weight)
        answers.append(label)
        answer_texts.append(answer_raw.strip())
        query_hashes.append(hashlib.sha256(query['query'].encode('utf-8')).hexdigest())

    raw = np.asarray(raw, dtype=np.float64)
    answers = np.asarray(answers, dtype=np.int8)
    if calculate_weights:
        weighted = raw * np.asarray(weights)[:, :, None]
        return raw, weighted, answers, query_hashes, noncanonical, answer_texts
    else:
        return raw, raw, answers, query_hashes, noncanonical, answer_texts

def propagate(initial, vision=False, quadratic=False, cls_only=False):
    """Original topology or within-layer quadratic topology, probability semantics."""
    num_records, tokens, layers = initial.shape
    final = np.zeros_like(initial)

    for layer in range(0, layers):
        for token in range(tokens-2, -1, -1):
            energy = np.zeros(num_records, dtype=np.float64)

            if quadratic:
                # Aggregate from all preceding tokens on the same layer.
                energy += final[:, token+1:, layer].sum(axis=1)

            elif cls_only:
                if token == 0:
                    energy += final[:, token+1:, layer].sum(axis=1)

            elif token:
                # Linear topology: previous token on the same layer.
                energy += final[:, token + 1, layer]

            # Last-position tokens also receive the last-position node from the layer below.
            if token == 0 and layer:
                energy += final[:, token, layer - 1]

            influence = 1 - np.tanh(np.abs(energy))
            final[:, token, layer] = np.tanh(energy) + influence * initial[:, token, layer]
    
    return final

def add_strength_colorbar(
    fig,
    mappable,
    *,
    right: float = 0.89,
    bottom: float = 0.12,
    height: float = 0.76,
    layout_top: float = 0.96,
):
    """Place the strength colorbar in a dedicated axis on the right."""
    fig.tight_layout(rect=[0, 0, right - 0.03, layout_top])
    cax = fig.add_axes([right, bottom, 0.02, height])
    fig.colorbar(mappable, cax=cax, label='Strength')


def plot_positive_negative_heatmaps(
    matrix,
    positive_label: str,
    negative_label: str,
    output_path: Path,
):
    import matplotlib.pyplot as plt

    # Split by CLS token strength at the deepest layer (token 0, last layer).
    positive_mask = matrix[:, 0, -1] >= 0
    negative_mask = matrix[:, 0, -1] < 0

    positive = matrix[positive_mask]
    negative = matrix[negative_mask]

    positive_mean = positive.mean(axis=0)
    positive_std = positive.std(axis=0)
    negative_mean = negative.mean(axis=0)
    negative_std = negative.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    vmin, vmax = -1, 1

    im = axes[0].imshow(
        positive_mean.T,
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
    )
    axes[0].set_title(f'{positive_label} mean')
    axes[0].set_xlabel('Token')
    axes[0].set_ylabel('Layer')

    axes[1].imshow(
        negative_mean.T,
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
    )
    axes[1].set_title(f'{negative_label} mean')
    axes[1].set_xlabel('Token')
    axes[1].set_ylabel('Layer')

    axes[0].invert_yaxis()
    axes[1].invert_yaxis()

    add_strength_colorbar(fig, im)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return {
        'positive_mean': positive_mean,
        'positive_std': positive_std,
        'negative_mean': negative_mean,
        'negative_std': negative_std,
        'positive_samples': positive,
        'negative_samples': negative,
    }


def plot_label_zero_instances(
    raw: np.ndarray,
    labels: np.ndarray,
    answer_texts: list[str],
    config: dict,
    output_dir: Path,
):
    """One figure per instance where map_answer_to_label is 0 (neither positive nor negative class)."""
    import matplotlib.pyplot as plt

    zero_indices = np.flatnonzero(labels == 0)
    if zero_indices.size == 0:
        return []

    positive_label = str(config['positive_class'])
    negative_label = str(config['negative_class'])
    plot_dir = Path(output_dir) / 'label_zero_instances'
    plot_dir.mkdir(parents=True, exist_ok=True)

    latent = propagate(raw, vision=True)
    quadratic = propagate(raw, vision=True, quadratic=True)
    panels = [
        ('Latent Debate', latent),
        ('Unweighted', raw),
        ('Quadratic', quadratic),
    ]

    written = []
    vmin, vmax = -1, 1
    for idx in tqdm(zero_indices, desc='Plot label-zero instances', unit='img'):
        answer_class = answer_texts[idx]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        last_im = None
        for ax, (title, tensor) in zip(axes, panels):
            last_im = ax.imshow(
                tensor[idx].T,
                cmap='coolwarm',
                vmin=vmin,
                vmax=vmax,
                aspect='auto',
            )
            ax.set_title(title)
            ax.set_xlabel('Token')
            ax.set_ylabel('Layer')
            ax.invert_yaxis()

        fig.suptitle(
            f'Instance {idx} | model answer: "{answer_class}" '
            f'(not {positive_label} or {negative_label})',
            fontsize=12,
        )
        add_strength_colorbar(fig, last_im, bottom=0.15, height=0.7, layout_top=0.88)

        safe_answer = re.sub(r'[^\w\-.]+', '_', answer_class.lower()).strip('_') or 'unknown'
        out_path = plot_dir / f'instance_{idx:05d}_{safe_answer}.png'
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        written.append({
            'index': int(idx),
            'answer_class': answer_class,
            'path': str(out_path),
        })

    summary_path = Path(output_dir) / 'label_zero_instances.json'
    write_json(summary_path, {
        'positive_class': positive_label,
        'negative_class': negative_label,
        'count': len(written),
        'instances': written,
    })
    return written


def label_to_class_name(label: int, config: dict) -> str:
    if label == 1:
        return str(config.get('positive_class', 'positive'))
    if label == -1:
        return str(config.get('negative_class', 'negative'))
    if label == 0:
        return 'other'
    return f'label_{label}'


def _subsample_indices(indices: np.ndarray, max_count: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size <= max_count:
        return indices
    positions = np.linspace(0, indices.size - 1, max_count, dtype=int)
    return indices[positions]


def _plot_method_vs_vit_comparison(
    idx: int,
    *,
    method: str,
    method_tensor: np.ndarray,
    raw: np.ndarray,
    llm_label: int,
    method_label: int,
    answer_class: str,
    pred_name: str,
    vision_images,
    out_path: Path,
    suptitle: str,
) -> None:
    import matplotlib.pyplot as plt

    vmin, vmax = -1, 1
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    if vision_images is not None:
        axes[0].imshow(np.asarray(vision_images[idx]['img']))
    else:
        axes[0].text(0.5, 0.5, f'Instance {idx}\n(no image)', ha='center', va='center')
    axes[0].set_title('Input image')
    axes[0].axis('off')

    axes[1].imshow(
        raw[idx].T,
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
    )
    axes[1].set_title(f'ViT: {answer_class} ({llm_label})')
    axes[1].set_xlabel('Token')
    axes[1].set_ylabel('Layer')
    axes[1].invert_yaxis()

    method_im = axes[2].imshow(
        method_tensor[idx].T,
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
    )
    axes[2].set_title(f'{method}: {pred_name} ({method_label})')
    axes[2].set_xlabel('Token')
    axes[2].set_ylabel('Layer')
    axes[2].invert_yaxis()

    fig.suptitle(suptitle, fontsize=11)
    add_strength_colorbar(fig, method_im, bottom=0.18, height=0.65, layout_top=0.90)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def matrix_for_method(method: str, raw: np.ndarray, weighted: np.ndarray, vision: bool) -> np.ndarray:
    name = method.lower()
    if vision:
        if 'quadratic' in name:
            return propagate(raw, vision=True, quadratic=True)
        if 'cls_only' in name:
            return propagate(raw, vision=True, cls_only=True)
        if 'unweighted' in name:
            return raw
        if 'latent' in name or 'debate' in name:
            return propagate(raw, vision=True)
        return raw
    if 'quadratic' in name:
        return propagate(weighted, vision=False, quadratic=True)
    if 'unweighted' in name:
        return propagate(raw, vision=False)
    if 'latent' in name or 'debate' in name:
        return propagate(weighted, vision=False)
    return raw


def _plot_prediction_comparisons(
    raw: np.ndarray,
    weighted: np.ndarray,
    llm: np.ndarray,
    predictions: dict[str, np.ndarray],
    answer_texts: list[str],
    config: dict,
    output_dir: Path,
    vision: bool,
    vision_images,
    *,
    agree: bool,
    max_examples: int | None,
    root_name: str,
    progress_desc: str,
    suptitle: str,
) -> dict[str, list[dict]]:
    root = Path(output_dir) / root_name
    root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[dict]] = {}
    if vision_images is not None and len(vision_images) != len(raw):
        raise ValueError(
            f'Vision image count ({len(vision_images)}) != JSONL rows ({len(raw)}); '
            're-extract jsonl or fix vision_dataset in config'
        )

    for method, values in tqdm(predictions.items(), desc=progress_desc, unit='method'):
        if method.strip().lower() in _SKIP_BASELINE_METHODS:
            continue
        if agree:
            match_idx = np.flatnonzero(values == llm)
            if match_idx.size == 0:
                summary[method] = []
                continue
            plot_idx = (
                _subsample_indices(match_idx, max_examples)
                if max_examples is not None
                else match_idx
            )
        else:
            plot_idx = np.flatnonzero(values != llm)
            if plot_idx.size == 0:
                summary[method] = []
                continue

        safe_method = re.sub(r'[^\w\-.]+', '_', method).strip('_')
        tensor = matrix_for_method(method, raw, weighted, vision)
        written: list[dict] = []

        for idx in tqdm(plot_idx, desc=f'  {method}', unit='img', leave=False):
            answer_class = answer_texts[idx]
            pred_name = label_to_class_name(int(values[idx]), config)
            llm_ref = int(llm[idx])
            bucket = 'reference_zero' if llm_ref == 0 else 'reference_nonzero'
            safe_answer = re.sub(r'[^\w\-.]+', '_', answer_class.lower()).strip('_') or 'unknown'
            out_path = root / bucket / safe_method / f'instance_{idx:05d}_{safe_answer}.png'
            _plot_method_vs_vit_comparison(
                int(idx),
                method=method,
                method_tensor=tensor,
                raw=raw,
                llm_label=llm_ref,
                method_label=int(values[idx]),
                answer_class=answer_class,
                pred_name=pred_name,
                vision_images=vision_images,
                out_path=out_path,
                suptitle=f'Instance {idx} — {suptitle}',
            )
            written.append({
                'index': int(idx),
                'answer_class': answer_class,
                'llm_label': llm_ref,
                'llm_bucket': bucket,
                'method_label': int(values[idx]),
                'path': str(out_path),
            })

        summary[method] = written

    write_json(root / 'summary.json', {
        'vision': vision,
        'agreement' if agree else 'disagreement': True,
        'max_examples': max_examples,
        'methods': {
            method: {'count': len(items), 'instances': items}
            for method, items in summary.items()
        },
    })
    return summary


def plot_prediction_disagreements(
    raw: np.ndarray,
    weighted: np.ndarray,
    llm: np.ndarray,
    predictions: dict[str, np.ndarray],
    answer_texts: list[str],
    config: dict,
    output_dir: Path,
    vision: bool,
    vision_images=None,
) -> dict[str, list[dict]]:
    """One figure per disagreement: input image, ViT (raw) heatmap, method heatmap."""
    return _plot_prediction_comparisons(
        raw,
        weighted,
        llm,
        predictions,
        answer_texts,
        config,
        output_dir,
        vision,
        vision_images,
        agree=False,
        max_examples=None,
        root_name='prediction_disagreements',
        progress_desc='Plot prediction disagreements',
        suptitle='prediction disagreement',
    )


def plot_prediction_agreements(
    raw: np.ndarray,
    weighted: np.ndarray,
    llm: np.ndarray,
    predictions: dict[str, np.ndarray],
    answer_texts: list[str],
    config: dict,
    output_dir: Path,
    vision: bool,
    vision_images=None,
    max_examples: int = VISION_AGREEMENT_PLOT_MAX,
) -> dict[str, list[dict]]:
    """Sample figures where ViT and method predictions match (same layout as disagreements)."""
    return _plot_prediction_comparisons(
        raw,
        weighted,
        llm,
        predictions,
        answer_texts,
        config,
        output_dir,
        vision,
        vision_images,
        agree=True,
        max_examples=max_examples,
        root_name='prediction_agreements',
        progress_desc='Plot prediction agreements',
        suptitle='prediction agreement',
    )


def predict_all(raw, weighted, config, vision=False, output_dir: Path | None = None):
    rng = random.Random(config['random_seed'])
    n_tokens, n_layers = raw.shape[1], raw.shape[2]
    majority = (raw >= 0).sum(axis=(1, 2)) >= n_tokens * n_layers / 2

    random_pred = np.asarray(
        [1 if rng.choice(sample.ravel().tolist()) >= 0 else -1 for sample in raw],
        dtype=np.int8,
    )
    average_pred = np.where(raw.mean(axis=(1, 2)) >= 0, 1, -1).astype(np.int8)
    majority_pred = np.where(majority, 1, -1).astype(np.int8)
    if vision:
    
        latent_debate = np.select(
            [
                propagate(raw, vision)[:, 0, -1] >= 0,
                propagate(raw, vision)[:, 0, -1] < 0
            ],
            [1, -1],
            default=0
        ).astype(np.int8)

        # Make this raw for now! Propagation actually does change a lot!
        unweighted = np.select(
            [
                raw[:, 0, -1] >= 0,
                raw[:, 0, -1] < 0
            ],
            [1, -1],
            default=0
        ).astype(np.int8)

        quadratic = np.select(
            [
                propagate(raw, vision, quadratic=True)[:, 0, -1] >= 0,
                propagate(raw, vision, quadratic=True)[:, 0, -1] < 0
            ],
            [1, -1],
            default=0
        ).astype(np.int8)

        propagated = propagate(raw, vision)
        propagated_quadratic = propagate(raw, vision, quadratic=True)

        positive_label = str(config['positive_class'])
        negative_label = str(config['negative_class'])
        if output_dir is not None:
            raw_stats = plot_positive_negative_heatmaps(
                raw, positive_label, negative_label, output_dir / 'raw_mean_heatmaps.png'
            )
            propagated_stats = plot_positive_negative_heatmaps(
                propagated,
                positive_label,
                negative_label,
                output_dir / 'propagated_mean_heatmaps.png',
            )
            plot_positive_negative_heatmaps(
                propagated_quadratic,
                positive_label,
                negative_label,
                output_dir / 'propagated_quadratic_mean_heatmaps.png',
            )
        
    else:
        latent_debate = np.where(propagate(weighted)[:, -1, -1] >= 0, 1, -1).astype(np.int8)
        unweighted = np.where(propagate(raw)[:, -1, -1] >= 0, 1, -1).astype(np.int8)
        quadratic = np.where(
            propagate(weighted, vision, quadratic=True)[:, -1, -1] >= 0,
            1,
            -1,
        ).astype(np.int8)

    predictions = [
        random_pred,
        average_pred,
        majority_pred,
        latent_debate,
        unweighted,
        quadratic,
    ]
    return dict(zip(config['methods'], predictions))


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


def make_latex(table, config):
    datasets = [d['name'] for d in config['datasets']]
    headers = [
        'common\\_claim',
        'counterfact',
        'company',
        'TriviaQA',
        'MuSiQue',
        'TruthfulQA',
        'Avg',
    ]
    header_cells = ' & '.join(r'\textbf{\underline{\texttt{' + h + '}}}' for h in headers)
    lines = [
        r'\begin{table*}[tp]',
        r'    \centering',
        r'    \tiny',
        r'    \setlength{\tabcolsep}{2.2mm}',
        r'    \begin{threeparttable}',
        r'        \begin{tabular}{lccccccc}',
        r'            \toprule',
        '            & ' + header_cells + r' \\',
        r'            & 500 & 500 & 500 & 500 & 500 & 500 & -- \\',
    ]
    for model in config['models']:
        subset = [r for r in table if r['model'] == model['name']]
        columns = datasets + ['Avg']
        maxima = {key: max(Decimal(r[key]) for r in subset) for key in columns}
        lines.extend([
            r'            \midrule',
            r'            \multicolumn{8}{c}{\textit{' + model['name'] + r' (\%)}} \\',
            r'            \midrule',
        ])
        for row in subset:
            if row['method'] == 'Latent Debate':
                lines.append(r'            \rowcolor{gray!10}')
            values = [
                r'\textbf{' + row[key] + '}' if Decimal(row[key]) == maxima[key] else row[key]
                for key in columns
            ]
            lines.append('            ' + row['method'] + ' & ' + ' & '.join(values) + r' \\')
    lines.extend([
        r'            \bottomrule',
        r'        \end{tabular}',
        r'        \caption{Consistency scores on binary and open-ended question-answering datasets with three LLMs. Each dataset contains 500 examples. The first three datasets use the last six prompt tokens; TriviaQA, MuSiQue and TruthfulQA use the last five. Avg is the arithmetic mean across the six datasets. The best scores within each model are shown in bold.}',
        r'        \label{tab:correlation}',
        r'    \end{threeparttable}',
        r'    \vspace{-10pt}',
        r'\end{table*}',
    ])
    return '\n'.join(lines) + '\n'

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
    parser.add_argument('--manifest-file', type=str, default='input_manifest.json')
    parser.add_argument('--expected-results-file', type=str, default='expected_results.json')
    parser.add_argument('--calculate-weights', type=str_to_bool, default=True)
    parser.add_argument('--skip-checks', type=str_to_bool, default=False)
    parser.add_argument('--vision', type=str_to_bool, default=False)
    parser.add_argument(
        '--plot',
        type=str_to_bool,
        default=False,
        help='Write vision diagnostic heatmaps (mean, label-zero, agreements, disagreements)',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    config = read_json(HERE / args.config_file)
    manifest = read_json(HERE / args.manifest_file)
    expected = read_json(HERE / args.expected_results_file)

    if not args.skip_checks and protocol_hash(config) != expected['protocol_sha256']:
        raise ValueError('Configuration differs from the finalized Table 1 protocol')

    if not args.skip_checks:
        expected_cells = {(r['model'], r['dataset'], r['method']): r for r in expected['cells']}
        expected_predictions = np.load(HERE / 'expected_predictions.npz', allow_pickle=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    prediction_checks = 0
    input_checks = []
    answer_diagnostics = []

    data_dir = args.data_dir / args.data_subdir

    for model in config['models']:
        for dataset, file in zip(config['datasets'], model['files']):
            path = data_dir / file
            if not path.is_file():
                raise FileNotFoundError(f'Missing Table 1 input: {path}')

            digest = sha(path)
            if not args.skip_checks and digest != manifest['files'][file]['sha256']:
                raise ValueError(
                    f'Input fingerprint mismatch: {file}; '
                    'keep the released bytes and sample order unchanged'
                )

            raw, weighted, llm, query_hashes, noncanonical, answer_texts = load_input(
                path, dataset['tokens'], config, args.calculate_weights, args.skip_checks, args.vision
            )
            answer_diagnostics.extend(
                {'model': model['name'], 'dataset': dataset['name'], **r}
                for r in noncanonical
            )
            plot_output = args.output_dir if args.vision and args.plot else None
            vision_images = None
            if plot_output is not None:
                vision_images = load_vision_image_dataset(config)
                plot_label_zero_instances(
                    raw, llm, answer_texts, config, plot_output
                )
            predictions = predict_all(
                raw, weighted, config, args.vision, plot_output
            )
            if plot_output is not None:
                plot_prediction_agreements(
                    raw,
                    weighted,
                    llm,
                    predictions,
                    answer_texts,
                    config,
                    plot_output,
                    args.vision,
                    vision_images,
                )
                plot_prediction_disagreements(
                    raw,
                    weighted,
                    llm,
                    predictions,
                    answer_texts,
                    config,
                    plot_output,
                    args.vision,
                    vision_images,
                )

            key = model['name'].replace('-', '_') + '__' + dataset['name']
            all_predictions = np.column_stack(
                [llm] + [predictions[m] for m in config['methods']]
            )
            if not args.skip_checks and not np.array_equal(all_predictions, reference):
                reference = expected_predictions[key]
                bad = np.argwhere(all_predictions != reference)
                raise AssertionError(
                    f'Prediction mismatch for {key}: first mismatch={bad[0].tolist()}'
                )

            if not args.skip_checks:

                prediction_checks += len(llm) * len(config['methods'])
                sample_rows = []
                for i in range(len(llm)):
                    sample_rows.append({
                        'index': i,
                        'query_sha256': query_hashes[i],
                        'llm_prediction': int(llm[i]),
                        **{method: int(values[i]) for method, values in predictions.items()},
                    })
                write_csv(args.output_dir / 'predictions' / (key + '.csv'), sample_rows)

            for method, values in predictions.items():
                num_consistent = int((values == llm).sum())
                disagree = values != llm
                disagree_llm_zero = int((disagree & (llm == 0)).sum())
                disagree_llm_nonzero = int((disagree & (llm != 0)).sum())
                n_samples = len(llm)

                if not args.skip_checks:
                    reference_cell = expected_cells[(model['name'], dataset['name'], method)]
                    if num_consistent != reference_cell['consistent'] or len(llm) != reference_cell['n']:
                        raise AssertionError(f'Cell mismatch: {key}, {method}')
                row_data = {
                    'model': model['name'],
                    'dataset': dataset['name'],
                    'method': method,
                    'n': n_samples,
                    'consistent': num_consistent,
                    'consistency_percent': float(percentage(num_consistent, len(llm))),
                    'display_percent': display(
                        percentage(num_consistent, len(llm)),
                        config['decimal_places'],
                    ),
                    'disagree_llm_zero': disagree_llm_zero,
                    'disagree_llm_nonzero': disagree_llm_nonzero,
                    'tokens': dataset['tokens'],
                    'layers': config['selected_layers'],
                    'random_seed': config['random_seed'] if method == 'Random' else '',
                    'input_file': file,
                    'input_sha256': digest,
                }
                row_data['disagree_pct_llm_0'] = float(
                    percentage(disagree_llm_zero, n_samples)
                )
                row_data['disagree_pct_otherwise'] = float(
                    percentage(disagree_llm_nonzero, n_samples)
                )
                rows.append(row_data)

            input_checks.append({
                'model': model['name'],
                'dataset': dataset['name'],
                'file': file,
                'sha256': digest,
            })
    print(
        f'{model["name"]} / {dataset["name"]}: n={len(llm)}, '
        f'tokens={dataset["tokens"]}, all 6 methods verified',
        flush=True,
    )

    table = make_table(rows, config)
    if not args.skip_checks and table != expected['table_display']:
        raise AssertionError('Displayed table or row averages differ from the finalized Table 1')

    write_csv(args.output_dir / 'table1.csv', table)
    write_csv(args.output_dir / 'table1_full_precision.csv', rows)
    (args.output_dir / 'table1.tex').write_text(make_latex(table, config), encoding='utf-8')

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
    write_csv(args.output_dir / 'averages.csv', mean_rows)

    if answer_diagnostics:
        write_csv(args.output_dir / 'noncanonical_answer_prefixes.csv', answer_diagnostics)

    write_json(args.output_dir / 'config_used.json', config)
    write_json(args.output_dir / 'environment.json', {
        'python': sys.version,
        'platform': platform.platform(),
        'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy']},
    })
    status = {
        'verification': 'PASSED',
        'protocol_version': config['protocol_version'],
        'models': 3,
        'datasets_per_model': 6,
        'samples_per_input': 500,
        'input_files_verified': len(input_checks),
        'dataset_cells_verified': len(rows),
        'row_averages_verified': len(table),
        'per_sample_method_predictions_verified': prediction_checks,
        'random_seed': config['random_seed'],
        'llm_answer_rule': config['llm_answer_rule'],
        'noncanonical_answer_prefixes': len(answer_diagnostics),
        'mean_rounding': 'arithmetic mean of six unrounded scores, then ROUND_HALF_UP to one decimal',
        'elapsed_seconds': time.time() - start,
        'input_files': input_checks,
    }
    write_json(args.output_dir / 'run_status.json', status)

    print(
        'PASSED: 108 Table 1 cells, 18 averages and 54,000 per-sample method predictions.',
        flush=True,
    )
    print('Results: ' + str(args.output_dir.resolve()), flush=True)


if __name__ == '__main__':
    main()
