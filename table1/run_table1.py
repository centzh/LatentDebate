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
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from scipy.special import softmax

HERE = Path(__file__).resolve().parent


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


def load_input(path, tokens, config, calculate_weights=True, skip_checks=False):
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
    query_hashes = []
    noncanonical = []
    layers = config['selected_layers']

    for index, record in enumerate(records):
        # Query metadata includes the string, tokenized query, layer-wise token
        # probabilities, and token scores used as QBAF initial strengths/weights.
        query = record['result']['query']
        p = np.asarray(query['query_probs'], dtype=np.float64)
        if p.ndim != 2 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f'Invalid probability array: {path.name}, row {index}')

        # Last prompt tokens, from layer 0 up to (not including) the output layer.
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

        answer = record['result']['answer']['answer'].strip()
        # Preserve the exact original Table 1 target mapping, including nonstandard
        # generated answers. Do not silently replace the already-finalized protocol.
        label = 1 if answer.startswith('T') else -1
        if not skip_checks and not (answer.startswith('True') or answer.startswith('False')):
            noncanonical.append({
                'input_file': path.name,
                'index': index,
                'original_answer': answer,
                'mapped_llm_prediction': label,
            })

        raw.append(selected)
        if calculate_weights:
            weights.append(weight)
        answers.append(label)
        query_hashes.append(hashlib.sha256(query['query'].encode('utf-8')).hexdigest())

    raw = np.asarray(raw, dtype=np.float64)
    answers = np.asarray(answers, dtype=np.int8)
    if calculate_weights:
        weighted = raw * np.asarray(weights)[:, :, None]
        return raw, weighted, answers, query_hashes, noncanonical
    else:
        return raw, raw, answers, query_hashes, noncanonical

def propagate(initial, quadratic=False):
    """Original topology or within-layer quadratic topology, probability semantics."""
    num_records, tokens, layers = initial.shape
    final = np.zeros_like(initial)

    for layer in range(layers):
        for token in range(tokens):
            energy = np.zeros(num_records, dtype=np.float64)

            if quadratic:
                # Aggregate from all preceding tokens on the same layer.
                energy += final[:, :token, layer].sum(axis=1)
            elif token:
                # Linear topology: previous token on the same layer.
                energy += final[:, token - 1, layer]

            # Last-position tokens also receive the last-position node from the layer below.
            if token == tokens - 1 and layer:
                energy += final[:, token, layer - 1]

            influence = 1 - np.tanh(np.abs(energy))
            final[:, token, layer] = np.tanh(energy) + influence * initial[:, token, layer]

    return final


def predict_all(raw, weighted, config):
    rng = random.Random(config['random_seed'])
    n_tokens, n_layers = raw.shape[1], raw.shape[2]
    majority = (raw >= 0).sum(axis=(1, 2)) >= n_tokens * n_layers / 2

    random_pred = np.asarray(
        [1 if rng.choice(sample.ravel().tolist()) >= 0 else -1 for sample in raw],
        dtype=np.int8,
    )
    average_pred = np.where(raw.mean(axis=(1, 2)) >= 0, 1, -1).astype(np.int8)
    majority_pred = np.where(majority, 1, -1).astype(np.int8)
    latent_debate = np.where(propagate(weighted)[:, -1, -1] >= 0, 1, -1).astype(np.int8)
    unweighted = np.where(propagate(raw)[:, -1, -1] >= 0, 1, -1).astype(np.int8)
    quadratic = np.where(
        propagate(weighted, quadratic=True)[:, -1, -1] >= 0,
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

            raw, weighted, llm, query_hashes, noncanonical = load_input(
                path, dataset['tokens'], config, args.calculate_weights, args.skip_checks
            )
            answer_diagnostics.extend(
                {'model': model['name'], 'dataset': dataset['name'], **r}
                for r in noncanonical
            )
            predictions = predict_all(raw, weighted, config)

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
                count = int((values == llm).sum())
                if not args.skip_checks:
                    reference_cell = expected_cells[(model['name'], dataset['name'], method)]
                    if count != reference_cell['consistent'] or len(llm) != reference_cell['n']:
                        raise AssertionError(f'Cell mismatch: {key}, {method}')
                rows.append({
                    'model': model['name'],
                    'dataset': dataset['name'],
                    'method': method,
                    'n': len(llm),
                    'consistent': count,
                    'consistency_percent': float(percentage(count, len(llm))),
                    'display_percent': display(
                        percentage(count, len(llm)),
                        config['decimal_places'],
                    ),
                    'tokens': dataset['tokens'],
                    'layers': config['selected_layers'],
                    'random_seed': config['random_seed'] if method == 'Random' else '',
                    'input_file': file,
                    'input_sha256': digest,
                })

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
