# Embeddings

Exploring word embeddings using GloVe vectors — analogies, visualization, debiasing, and concept interpolation.

**[Read the interactive article](https://timvieira.github.io/steering-embeddings/)**

## Overview

- **`embedding.py`** — Core `Embeddings` class for loading GloVe vectors, computing analogies (`king - man + woman = queen`), finding similar words via KD-tree, MDS-based visualization, and a subspace debiasing method that projects out gender-correlated directions.

- **`search.py`** — A* search over the embedding space to find interpretable paths between two concepts (experimental).

- **`FATE.ipynb`** — Walkthrough notebook covering word analogies, gender bias in embeddings, and debiasing with visualizations.

## Setup

1. Download [GloVe 6B vectors](https://nlp.stanford.edu/projects/glove/) and place `glove.6B.100d.txt` in `data/`.

2. Install dependencies:
   ```
   pip install numpy scipy plotly arsenal
   ```

3. Build the compressed vectors:
   ```
   python embedding.py
   ```
   This creates `vecs.npz` from the GloVe text file for faster loading.

## Usage

```python
from embedding import Embeddings, normalize_rows, load_vecs
from arsenal import Alphabet
import numpy as np

with np.load('vecs.npz') as data:
    emb = Embeddings(normalize_rows(data['vec']), Alphabet(data['voc']))

emb.analogy('man :: woman -> king')   # queen
emb.most_similar(emb('dog'), n=5)     # nearest neighbors
```

Interactive mode:
```
python embedding.py -i
```
