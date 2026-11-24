"""Vision diagnostic figures for Table 1 runs."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from propagate import propagate
except ImportError:
    from table1.propagate import propagate

VISION_PLOT_MAX_EXAMPLES = 10
VISION_AGREEMENT_PLOT_MAX = VISION_PLOT_MAX_EXAMPLES  # backwards-compatible alias
_VIT_PATCH_GRID = 14
_SKIP_BASELINE_METHODS = frozenset({'random', 'average', 'majority voting'})


def _json_sanitize(value):
    """Replace non-finite floats with null so sweep.json stays strict JSON."""
    if isinstance(value, dict):
        return {key: _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    return value


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_sanitize(value), ensure_ascii=False, indent=2, allow_nan=False)
        + '\n',
        encoding='utf-8',
    )


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


def _vision_row_to_array(row) -> np.ndarray:
    img = row.get('img') or row.get('image')
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _upsample_patch_grid(patch: np.ndarray, height: int, width: int) -> np.ndarray:
    from scipy.ndimage import zoom

    if patch.shape[0] == height and patch.shape[1] == width:
        return patch
    sy = height / patch.shape[0]
    sx = width / patch.shape[1]
    return zoom(patch, (sy, sx), order=1)


def _plot_instance_spatial_heatmaps(
    idx: int,
    raw: np.ndarray,
    method_tensor: np.ndarray,
    method: str,
    vision_images,
    out_path: Path,
    suptitle: str,
    *,
    patch_grid: int = 14,
) -> None:
    """14×14 patch overlays on the instance image: ViT raw row + method row × 12 layers."""
    import matplotlib.pyplot as plt

    n_tokens, n_layers = raw.shape[1], raw.shape[2]
    n_patches = patch_grid * patch_grid
    if n_tokens != n_patches + 1:
        return

    sample_raw = raw[idx]
    sample_method = method_tensor[idx]
    ref_image = _vision_row_to_array(vision_images[idx]) if vision_images is not None else None
    patch_indices = np.arange(1, n_patches + 1)
    rows = [('ViT (raw)', sample_raw), (method, sample_method)]

    vmin, vmax = -1.0, 1.0
    fig, axes = plt.subplots(2, n_layers, figsize=(2.4 * n_layers, 5.5), squeeze=False)
    mappable = None

    for row_i, (row_title, tensor) in enumerate(rows):
        for layer in range(n_layers):
            ax = axes[row_i, layer]
            patch_map = tensor[patch_indices, layer].reshape(patch_grid, patch_grid)
            if ref_image is not None:
                h, w = ref_image.shape[:2]
                ax.imshow(ref_image, aspect='equal')
                overlay = _upsample_patch_grid(patch_map, h, w)
                mappable = ax.imshow(
                    overlay,
                    cmap='coolwarm',
                    vmin=vmin,
                    vmax=vmax,
                    alpha=0.55,
                    aspect='equal',
                )
            else:
                mappable = ax.imshow(
                    patch_map,
                    cmap='coolwarm',
                    vmin=vmin,
                    vmax=vmax,
                    aspect='equal',
                )
            ax.set_title(f'L{layer}', fontsize=9)
            ax.axis('off')
        axes[row_i, 0].set_ylabel(row_title, fontsize=10, rotation=90, labelpad=32)

    if mappable is not None:
        add_strength_colorbar(fig, mappable, right=0.92, bottom=0.12, height=0.72, layout_top=0.92)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(suptitle, fontsize=11, y=0.98)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


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

    latent = propagate(raw)
    quadratic = propagate(raw, quadratic=True)
    cls_only = propagate(raw, cls_only=True)
    panels = [
        ('Raw', raw),
        ('Latent Debate', latent),
        ('Quadratic', quadratic),
        ('CLS only', cls_only),
    ]

    written = []
    vmin, vmax = -1, 1
    for idx in tqdm(zero_indices, desc='Plot label-zero instances', unit='img'):
        answer_class = answer_texts[idx]
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
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
    _write_json(summary_path, {
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


def matrix_for_method(method: str, raw: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    name = method.lower()
    if 'quadratic' in name:
        return propagate(raw, quadratic=True)
    if 'patch only' in name and 'cls' in name:
        return propagate(raw, cls_patch_only=True)
    if 'cls only' in name:
        return propagate(raw, cls_only=True)
    if name == 'raw' or 'unweighted' in name:
        return raw
    if name == 'weighted':
        weighted_input = raw if weights is None else raw * weights
        return propagate(weighted_input, weighted=True)
    if 'latent' in name or 'debate' in name:
        return propagate(raw)
    return raw


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
    ground_truth_name: str | None = None,
    ground_truth_label: int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    vmin, vmax = -1, 1
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    if vision_images is not None:
        row = vision_images[idx]
        img = row.get('img') or row.get('image')
        axes[0].imshow(np.asarray(img))
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
    if ground_truth_name:
        axes[1].set_title(
            f'ViT pred: {answer_class} ({llm_label})\n'
            f'GT label: {ground_truth_name} ({ground_truth_label})',
        )
    else:
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

    if vision_images is not None and raw.shape[1] == _VIT_PATCH_GRID * _VIT_PATCH_GRID + 1:
        spatial_path = out_path.with_name(f'{out_path.stem}_spatial{out_path.suffix}')
        _plot_instance_spatial_heatmaps(
            idx,
            raw,
            method_tensor,
            method,
            vision_images,
            spatial_path,
            suptitle,
        )


def _plot_prediction_comparisons(
    raw: np.ndarray,
    llm: np.ndarray,
    predictions: dict[str, np.ndarray],
    answer_texts: list[str],
    config: dict,
    output_dir: Path,
    vision_images,
    *,
    agree: bool,
    max_examples: int | None,
    root_name: str,
    progress_desc: str,
    suptitle: str,
    weights: np.ndarray | None = None,
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
            if max_examples is not None and plot_idx.size > max_examples:
                plot_idx = _subsample_indices(plot_idx, max_examples)

        safe_method = re.sub(r'[^\w\-.]+', '_', method).strip('_')
        tensor = matrix_for_method(method, raw, weights)
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

    _write_json(root / 'summary.json', {
        'agreement' if agree else 'disagreement': True,
        'max_examples': max_examples,
        'methods_plotted': sorted(predictions.keys()),
        'methods': {
            method: {'count': len(items), 'instances': items}
            for method, items in summary.items()
        },
    })
    return summary


def plot_vit_ground_truth_mismatches(
    raw: np.ndarray,
    vit_labels: np.ndarray,
    ground_truth: np.ndarray,
    ground_truth_texts: list[str],
    predictions: dict[str, np.ndarray],
    answer_texts: list[str],
    config: dict,
    output_dir: Path,
    vision_images=None,
    *,
    max_examples: int | None = VISION_PLOT_MAX_EXAMPLES,
    weights: np.ndarray | None = None,
) -> dict[str, list[dict]]:
    """Figures where ViT classifier head answer != dataset ground truth (JSONL label)."""
    mismatch_idx = np.flatnonzero(vit_labels != ground_truth)
    if mismatch_idx.size == 0:
        summary = {method: [] for method in predictions if method.lower() not in _SKIP_BASELINE_METHODS}
        root = Path(output_dir) / 'vit_gt_mismatches'
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / 'summary.json', {
            'vit_vs_ground_truth_mismatch': True,
            'count': 0,
            'methods': summary,
        })
        return summary

    root = Path(output_dir) / 'vit_gt_mismatches'
    root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[dict]] = {}
    if vision_images is not None and len(vision_images) != len(raw):
        raise ValueError(
            f'Vision image count ({len(vision_images)}) != JSONL rows ({len(raw)}); '
            're-extract jsonl or fix vision_dataset in config',
        )

    plot_idx = (
        _subsample_indices(mismatch_idx, max_examples)
        if max_examples is not None and mismatch_idx.size > max_examples
        else mismatch_idx
    )

    for method, values in tqdm(predictions.items(), desc='Plot ViT≠GT mismatches', unit='method'):
        if method.strip().lower() in _SKIP_BASELINE_METHODS:
            continue
        safe_method = re.sub(r'[^\w\-.]+', '_', method).strip('_')
        tensor = matrix_for_method(method, raw, weights)
        written: list[dict] = []

        for idx in tqdm(plot_idx, desc=f'  {method}', unit='img', leave=False):
            vit_name = answer_texts[idx]
            gt_name = ground_truth_texts[idx]
            vit_ref = int(vit_labels[idx])
            gt_ref = int(ground_truth[idx])
            pred_name = label_to_class_name(int(values[idx]), config)
            safe_vit = re.sub(r'[^\w\-.]+', '_', vit_name.lower()).strip('_') or 'unknown'
            safe_gt = re.sub(r'[^\w\-.]+', '_', gt_name.lower()).strip('_') or 'unknown'
            out_path = root / safe_method / f'instance_{idx:05d}_vit_{safe_vit}_gt_{safe_gt}.png'
            _plot_method_vs_vit_comparison(
                int(idx),
                method=method,
                method_tensor=tensor,
                raw=raw,
                llm_label=vit_ref,
                method_label=int(values[idx]),
                answer_class=vit_name,
                pred_name=pred_name,
                vision_images=vision_images,
                out_path=out_path,
                suptitle=(
                    f'Instance {idx} — ViT head ≠ dataset ground truth '
                    f'(pred {vit_name} vs GT {gt_name})'
                ),
                ground_truth_name=gt_name,
                ground_truth_label=gt_ref,
            )
            written.append({
                'index': int(idx),
                'vit_answer_class': vit_name,
                'vit_label': vit_ref,
                'ground_truth_class': gt_name,
                'ground_truth_label': gt_ref,
                'method_label': int(values[idx]),
                'path': str(out_path),
            })
        summary[method] = written

    _write_json(root / 'summary.json', {
        'vit_vs_ground_truth_mismatch': True,
        'max_examples': max_examples,
        'total_mismatches': int(mismatch_idx.size),
        'methods_plotted': sorted(predictions.keys()),
        'methods': {
            method: {'count': len(items), 'instances': items}
            for method, items in summary.items()
        },
    })
    return summary


def plot_prediction_disagreements(
    raw: np.ndarray,
    llm: np.ndarray,
    predictions: dict[str, np.ndarray],
    answer_texts: list[str],
    config: dict,
    output_dir: Path,
    vision_images=None,
    max_examples: int | None = VISION_PLOT_MAX_EXAMPLES,
    weights: np.ndarray | None = None,
) -> dict[str, list[dict]]:
    """One figure per disagreement: input image, ViT (raw) heatmap, method heatmap."""
    return _plot_prediction_comparisons(
        raw,
        llm,
        predictions,
        answer_texts,
        config,
        output_dir,
        vision_images,
        agree=False,
        max_examples=max_examples,
        root_name='prediction_disagreements',
        progress_desc='Plot prediction disagreements',
        suptitle='prediction disagreement',
        weights=weights,
    )


def _mean_ablated_strength(bucket: dict, *, attackers: bool) -> list:
    if attackers:
        return bucket.get(
            'mean_final_strength_negatives',
            bucket.get('mean_final_strength_positives', []),
        )
    return bucket['mean_final_strength_positives']


def _plot_ablated_final_strength_lines(ax, sweep: dict, *, show_legend: bool) -> None:
    above = sweep['ablate_above']
    below = sweep['ablate_below']
    ax.axhline(
        0.0,
        color='0.25',
        linestyle='--',
        linewidth=1.5,
        zorder=0,
        label='Zero strength' if show_legend else None,
    )
    ax.plot(
        above['thresholds'],
        _mean_ablated_strength(above, attackers=False),
        'o-',
        label='Ablate supporters > threshold',
    )
    ax.plot(
        below['thresholds'],
        _mean_ablated_strength(below, attackers=True),
        's-',
        label='Ablate attackers < threshold',
    )
    ax.set_xlabel('Ablation threshold')
    ax.set_ylabel('Mean ablated CLS final strength')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=8)


def _load_pair_sweep(repo_root: Path, dataset: str, pair: str) -> dict | None:
    sweep_path = (
        repo_root / 'vit_results' / dataset / pair / 'cls_patch_supporter_ablation' / 'sweep.json'
    )
    if sweep_path.is_file():
        return json.loads(sweep_path.read_text(encoding='utf-8'))

    config_path = repo_root / 'table1' / dataset / f'{pair}.json'
    data_path = repo_root / 'data' / dataset / f'{pair}.jsonl'
    if not config_path.is_file() or not data_path.is_file():
        return None

    import sys

    table1_dir = repo_root / 'table1'
    table1_dir_str = str(table1_dir)
    if table1_dir_str not in sys.path:
        sys.path.insert(0, table1_dir_str)
    import run_table1  # noqa: WPS433

    config = run_table1.read_json(config_path)
    raw, _, _, _, _, _ = run_table1.load_input(data_path, config)
    return run_table1.analyze_cls_patch_supporter_ablation(raw, output_dir=None, quiet=True)


def plot_cls_patch_ablation_strength_grid(
    repo_root: Path,
    dataset: str,
) -> Path | None:
    """Combine per-pair ablated final strength curves for one vision dataset."""
    import math

    import matplotlib.pyplot as plt

    repo_root = Path(repo_root)
    dataset_dir = repo_root / 'vit_results' / dataset
    pair_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    pairs: list[tuple[str, dict]] = []
    for pair_dir in pair_dirs:
        sweep = _load_pair_sweep(repo_root, dataset, pair_dir.name)
        if sweep is not None:
            pairs.append((pair_dir.name, sweep))

    if not pairs:
        print(f'No ablation sweeps for {dataset}; skip combined strength figure.', flush=True)
        return None

    n = len(pairs)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), squeeze=False)

    for index, (pair_name, sweep) in enumerate(pairs):
        row, col = divmod(index, ncols)
        ax = axes[row][col]
        _plot_ablated_final_strength_lines(ax, sweep, show_legend=(index == 0))
        ax.set_title(pair_name)
        ax.tick_params(labelsize=8)

    for index in range(n, nrows * ncols):
        row, col = divmod(index, ncols)
        axes[row][col].axis('off')

    fig.suptitle(f'{dataset}: ablated final strength vs threshold', fontsize=13)
    fig.tight_layout()
    out = dataset_dir / 'ablated_final_strength_all_pairs.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out.relative_to(repo_root)}', flush=True)
    return out


def plot_cls_patch_supporter_ablation(sweep: dict, output_dir: Path) -> None:
    """Save flip rate and mean ablated CLS strength vs ablation threshold."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir) / 'cls_patch_supporter_ablation'
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / 'sweep.json', sweep)

    above = sweep['ablate_above']
    below = sweep['ablate_below']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        above['thresholds'],
        above['flip_rates'],
        'o-',
        label='Ablate supporters > threshold',
    )
    ax.plot(
        below['thresholds'],
        below['flip_rates'],
        's-',
        label='Ablate attackers < threshold',
    )
    ax.set_xlabel('Ablation threshold')
    ax.set_ylabel('Flip rate (CLS-patch-positive samples)')
    ax.set_title('CLS patch only: prediction flip rate vs ablation threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    flip_path = output_dir / 'flip_rate_vs_threshold.png'
    fig.tight_layout()
    plt.savefig(flip_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_ablated_final_strength_lines(ax, sweep, show_legend=True)
    ax.set_title('CLS patch only: ablated final strength vs ablation threshold')
    strength_path = output_dir / 'ablated_final_strength_vs_threshold.png'
    fig.tight_layout()
    plt.savefig(strength_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_prediction_agreements(
    raw: np.ndarray,
    llm: np.ndarray,
    predictions: dict[str, np.ndarray],
    answer_texts: list[str],
    config: dict,
    output_dir: Path,
    vision_images=None,
    max_examples: int | None = VISION_PLOT_MAX_EXAMPLES,
    weights: np.ndarray | None = None,
) -> dict[str, list[dict]]:
    """Sample figures where ViT and method predictions match (same layout as disagreements)."""
    return _plot_prediction_comparisons(
        raw,
        llm,
        predictions,
        answer_texts,
        config,
        output_dir,
        vision_images,
        agree=True,
        max_examples=max_examples,
        root_name='prediction_agreements',
        progress_desc='Plot prediction agreements',
        suptitle='prediction agreement',
        weights=weights,
    )
