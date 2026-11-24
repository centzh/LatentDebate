# Latent Debate Reproducibility Code

This release contains the code and cached model outputs used to reproduce the
paper tables. Identical caches are stored once under `data/` and shared by all
table runners. Every CPU runner verifies input SHA-256 fingerprints and checks
its computed results against the bundled full-precision references.

## Contents

- `VISION_TABLE1.md`: ViT/Food-101/CIFAR-10 vision Table 1 (extract, score, collate, plots).
- `table1/`: Table 1.
- `table2/`: Table 2 and the SelfCheckGPT/Semantic Entropy GPU implementations.
- `table3_table4/`: Tables 3 and 4.
- `appendix_A1_A2_A4_A6/`: Tables A1, A2, A4 and A6.
- `appendix_A5/`: Table A5.
- `appendix_A3_gpu/`: Table A3 GPU pipeline.
- `data/`: shared probability, Tuned Lens and SAPLMA caches.

## CPU setup

Python 3.11 is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Reproduce all CPU tables

From the release root:

```bash
python run_all.py
```

Outputs are written under `results/`. A mismatch in an input hash or expected
result stops the run with an error.

## Run one table group

```bash
python table1/run_table1.py --output-dir results/table1
python table2/run.py --output-dir results/table2
python table3_table4/run.py --output-dir results/table3_table4
python appendix_A1_A2_A4_A6/run.py --output-dir results/appendix_A1_A2_A4_A6
python appendix_A5/run.py --output-dir results/appendix_A5
```

The fixed protocols, seeds and train/test splits are recorded in each table
folder's `config.json`. Machine-readable target values are stored in
`expected_results.json`, and paper-ready values are stored in the corresponding
`.tex` files.

## GPU code

- SelfCheckGPT: `table2/gpu/selfcheckgpt/run_selfcheckgpt.py`
- Semantic Entropy: `table2/gpu/semantic_entropy/`
- Table A3: `appendix_A3_gpu/`

Install the requirements provided beside each GPU implementation before use.

