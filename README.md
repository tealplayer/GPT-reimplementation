# GPT reimplementation

A from-scratch GPT implementation, built up step by step.

Based on:

1. `Attention is All You Need` - Vaswani et al. 2017
2. `A Mathematical Framework for Transformer Circuits` - Elhage et al. 2021

## Contents

- `gpt-reimplementation.ipynb` — the full notebook: character-level tokenizer, train/val split and batching, multi-head self-attention, transformer blocks, the training loop, sampling, and attention-pattern analysis.
- `input.txt` — the Tiny Shakespeare corpus (~1.1 MB) used as training data (the notebook also downloads it directly).

## Usage

```bash
pip install torch matplotlib
jupyter notebook gpt-reimplementation.ipynb
```

Run the cells top to bottom to train the model and generate sample text.
