# Table 1

Run from the release root:

```bash
python table1/run_table1.py --output-dir results/table1
```

This runner evaluates three models, six datasets and six methods on 500 samples
per cell. Common Claim, CounterFact and Company use the last six prompt tokens;
TriviaQA, MuSiQue and TruthfulQA use the last five. Random uses seed 0, reset for
each input cache. The other methods are deterministic. All displayed values,
averages and per-sample predictions are checked against the bundled references.

Configuration is in `config.json` (release Table 1). Per-dataset vision protocols live under `cifar10/` and `food101/` (e.g. `food101/cheesecake_chocolate_cake.json`). Pass one with `--config-file`. Input hashes for the release are in `input_manifest.json`.

