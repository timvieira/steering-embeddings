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
- `EmbeddingViz` has two embedding references:
  - `emb` — used for MDS coordinate computation (may be an extended embedding with anchor points)
  - `searchEmb` — used for click-to-expand neighbor search (should be the full 50K vocabulary)
  - When `buildVizWithDirection` creates an `extEmb`, always pass `searchEmb: emb` (the global one)

## Deploying

The deploy repo is a GitHub Pages site at `/tmp/steering-embeddings` (remote: `git@github.com:timvieira/steering-embeddings.git`). To deploy:

```bash
# Copy article files (one-way sync, never edit the deploy repo directly)
cp article/index.html /tmp/steering-embeddings/
cp article/js/* /tmp/steering-embeddings/js/
cp article/css/* /tmp/steering-embeddings/css/
cp article/img/* /tmp/steering-embeddings/img/

# Commit and push
cd /tmp/steering-embeddings
git add -A && git commit -m "Sync from main repo" && git push
```

Note: `data/` (binary vectors) are already in the deploy repo and rarely change. Only copy them if `export_vectors.py` has been re-run.

## Workflow rules

- **Always run tests** (`node article/test.mjs`) before committing. If tests fail, fix before committing.
- **Update TODO.md** after completing each task — mark `[x]` with a brief explanation of what was done.
- **Re-read files before editing** if another agent may have modified them. Check `git status` and `git log` frequently.
- **Never edit deploy repos directly** — only edit in this main repo. If a deploy repo exists, copy files out one-way.
- **Cache-busting**: JS module imports in index.html use `Date.now()` query strings so browsers load fresh code.
- **Distill template v1 quirks**:
  - The `dt-banner` ("awaiting review") is hidden via CSS: `dt-banner { display: none !important; }`
  - Front-matter with authors but no affiliations causes a harmless JS error (suppressed)
  - Math uses KaTeX loaded separately (not bundled in v1); use `$...$` and `$$...$$` delimiters
  - `<dt-cite key="...">` for inline citations; bibliography in `<script type="text/bibliography">`
- **D3 zoom vs orbit**: In 3D projected mode, D3 zoom is NOT attached (would conflict with orbit drag). When reusing an SVG from 2D, stale zoom listeners must be removed: `svg.on('.zoom', null)`.
- **The user values**: understated writing (no dorky capitalization, no cliché quotes), smoke-tested code, and thorough TODO tracking. Don't present work as done without verifying it works.
