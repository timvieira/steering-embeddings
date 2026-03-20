# Steering Word Embeddings

Interactive article exploring word vectors, analogies, and subspace projection.
Uses GloVe embeddings, runs entirely in the browser.

## Project structure

- `article/index.html` — main article (Distill template, prose, inline JS for visualization setup)
- `article/js/embeddings.js` — embedding operations: loading, nearest neighbors, analogies, MDS, steering
- `article/js/viz.js` — all visualization code: D3 plots (1D/2D/projected-3D), Three.js hero, steering animation
- `article/css/style.css` — styles (mostly overridden by inline styles in index.html)
- `article/data/glove-{small,medium,large}.bin` — binary GloVe vectors (10K/50K/400K words)
- `article/export_vectors.py` — script to produce the binary files from raw GloVe
- `article/test.mjs` — Puppeteer smoke tests
- `TODO.md` — task tracking

## Running tests

Always run tests before presenting work:

```bash
# Start server (leave running)
python3 -m http.server 8768 --directory article &

# Run smoke tests
node article/test.mjs http://localhost:8768/index.html
```

Tests require puppeteer installed at `/tmp/node_modules/puppeteer`.
The test suite loads the full 50K vocabulary and takes ~60s.

## Key conventions

- Direction arrows on plots use two methods depending on the data:
  - **Pair-difference SVD**: for paired concepts (gender, degree) — SVD on covariance of pair differences
  - **PCA**: for ordered sequences (numbers) — top principal component of the word vectors
  - SVD eigenvectors have arbitrary sign; always orient by dotting with a known reference direction
- `buildVizWithDirection()` in index.html handles both modes via `directionMode: 'pairs'` (default) or `'pca'`
- Anchor points for direction arrows are synthetic 100-D vectors included in the MDS distance matrix
- `hiddenPoints` in viz.js suppresses circles (r=0) for anchor points but keeps labels
- The hero visualization uses Three.js; all inline plots use D3 SVG with projected 3D rotation
- Multiple agents may edit this repo concurrently — prefer editing different files to avoid conflicts
