# Vision Table 1

Commands run from the repository root. Table 1 does not write plots.

`extract` writes the JSONL. `score` runs Table 1 from that JSONL. `run` does both: it extracts, then scores.

`--weighted` selects the attention-weighted config (`dumplings_gyoza_weighted`). Omit it for the plain config.

`score` and `run` refresh `vit_results/<dataset>/table1.csv` and `table1_average.csv`. Each file has an unweighted half (`avg_unweighted`, then those pairs) and a weighted half (`avg_weighted`, then those pairs). Averages are computed inside each half only. `table1.csv` is agreement with the ViT head. `table1_average.csv` is that number as a percent of Raw on the same column.

## One class pair

```bash
uv run vision.py extract food101 dumplings_gyoza
uv run vision.py score food101 dumplings_gyoza
uv run vision.py run food101 dumplings_gyoza          # extract, then score
```

Weighted extract:

```bash
uv run vision.py extract food101 dumplings_gyoza --weighted
uv run vision.py score food101 dumplings_gyoza --weighted
uv run vision.py run food101 dumplings_gyoza --weighted   # extract, then score
```

## Every class pair in a dataset

```bash
uv run vision.py extract food101
uv run vision.py score food101
uv run vision.py run food101          # extract, then score every pair
```

Weighted configs only:

```bash
uv run vision.py extract food101 --weighted
uv run vision.py score food101 --weighted
uv run vision.py run food101 --weighted   # extract, then score every weighted pair
```

`cifar10` works the same way as `food101`.

## Plots

Each figure has the image in the left column. The rows are raw strengths, the tensor that method consumes, and that method's output. At most 10 instances per method. Agreements, disagreements, and ViT-versus-ground-truth mismatches are all written.

```bash
uv run vision.py plot food101 caesar_salad_caprese_salad
uv run vision.py plot food101 caesar_salad_caprese_salad --weighted
uv run vision.py plot food101
uv run vision.py plot food101 --weighted
```

`--compare` draws raw, the unweighted output, and the weighted output on one figure. It needs both JSONL files.

```bash
uv run vision.py plot food101 caesar_salad_caprese_salad --compare
uv run vision.py plot food101 --compare
```

Compared figures go in `vit_results/<dataset>/<pair>_compared/`.

## Files

| Step | Path |
|------|------|
| Extract config | `config/vit/<dataset>/<pair>.yaml` |
| JSONL | `data/vit/<dataset>/<pair>.jsonl` |
| Table 1 config | `table1/vit/<dataset>/<pair>.json` |
| One pair | `vit_results/<dataset>/<pair>/` |
| Both halves | `vit_results/<dataset>/table1.csv`, `table1_average.csv` |
