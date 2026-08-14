# GPT reimplementation

A from-scratch GPT implementation, built up step by step.

## Contents

- `data_pipeline.py` — character-level tokenizer (`stoi`/`itos`), train/val split, and `get_batch` for sampling `(x, y)` training batches.
- `input.txt` — the Tiny Shakespeare corpus (~1.1 MB) used as training data.

## Usage

```bash
pip install torch
python data_pipeline.py
```

Prints a decoded input/target batch pair to sanity-check the pipeline.
