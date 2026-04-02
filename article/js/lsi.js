/**
 * LSI-style context expansion and UI.
 * Extracted from embeddings.js and index.html.
 *
 * Dependencies (Embeddings, EmbeddingViz) are passed as arguments to avoid
 * static import chains that conflict with cache-busting dynamic imports.
 */

/**
 * LSI-style context expansion: find words from the full vocabulary that
 * belong in the same low-dimensional subspace as the seed words.
 *
 * Two-stage retrieval (pseudo-relevance feedback style):
 * 1. Initial retrieval: find top candidates by cosine similarity to centroid.
 * 2. Re-rank by subspace alignment: reconstruction ratio = ||proj_B(w - μ)||² / ||w - μ||²
 *
 * @param {Embeddings} emb - The embedding to search
 * @param {string[]} seedWords - Seed words defining the subspace
 * @param {object} opts - { k: subspace dims, n: number of results }
 * @returns {{ results: [{word, score}], eigenvalues: number[] }}
 */
function lsiExpand(emb, seedWords, { k = 5, n = 20 } = {}) {
  const d = emb.dims;
  const seeds = seedWords.filter(w => emb.has(w));
  if (seeds.length < 2) return { results: [], eigenvalues: [] };

  // Centroid of seed vectors
  const centroid = new Float64Array(d);
  const seedVecs = seeds.map(w => emb.vec(w));
  for (const v of seedVecs) for (let i = 0; i < d; i++) centroid[i] += v[i];
  for (let i = 0; i < d; i++) centroid[i] /= seeds.length;

  // Covariance matrix of centered seed vectors
  const C = new Float64Array(d * d);
  for (const v of seedVecs) {
    for (let i = 0; i < d; i++)
      for (let j = 0; j < d; j++)
        C[i * d + j] += (v[i] - centroid[i]) * (v[j] - centroid[j]);
  }
  const Cmat = [];
  for (let i = 0; i < d; i++) {
    Cmat.push([]);
    for (let j = 0; j < d; j++) Cmat[i].push(C[i * d + j]);
  }
  const { u: svdU, q: svdQ } = SVDJS.SVD(Cmat);

  // Sort eigenvalues descending (svd-js doesn't guarantee order)
  const eigOrder = svdQ.map((v, i) => i).sort((a, b) => svdQ[b] - svdQ[a]);
  const sortedEigs = eigOrder.map(i => svdQ[i]);

  // Top-k eigenvectors (only those with nonzero eigenvalues)
  const kActual = Math.min(k, seeds.length - 1, d);
  const basis = [];
  for (let ki = 0; ki < kActual; ki++) {
    if (sortedEigs[ki] < 1e-10) break;
    const col = new Float64Array(d);
    const origIdx = eigOrder[ki];
    for (let i = 0; i < d; i++) col[i] = svdU[i][origIdx];
    basis.push(col);
  }

  const eigenvalues = sortedEigs.slice(0, kActual);

  // Stage 1: centroid NN — get a generous candidate pool
  const centroidNormalized = new Float32Array(d);
  for (let i = 0; i < d; i++) centroidNormalized[i] = centroid[i];
  // Normalize in place (use static method from the Embeddings class via prototype chain)
  const norm = Math.sqrt(centroidNormalized.reduce((s, v) => s + v * v, 0));
  if (norm > 0) for (let i = 0; i < d; i++) centroidNormalized[i] /= norm;
  const poolSize = Math.max(n * 10, 200);
  const candidates = emb.mostSimilar(centroidNormalized, poolSize, new Set(seeds));

  // Stage 2: re-rank by reconstruction ratio
  const results = [];
  for (const word of candidates) {
    const v = emb.vec(word);
    if (!v) continue;

    let fullNormSq = 0;
    let projNormSq = 0;
    for (let i = 0; i < d; i++) {
      const vi = v[i] - centroid[i];
      fullNormSq += vi * vi;
    }
    for (const b of basis) {
      let bDot = 0;
      for (let i = 0; i < d; i++) bDot += (v[i] - centroid[i]) * b[i];
      projNormSq += bDot * bDot;
    }
    const reconRatio = fullNormSq > 0 ? projNormSq / fullNormSq : 0;
    results.push({ word, score: reconRatio });
  }
  results.sort((a, b) => b.score - a.score);
  return { results: results.slice(0, n), eigenvalues };
}


/**
 * Set up the LSI appendix UI: textarea, eigenvalue histogram, comparison table, MDS plot.
 *
 * @param {Embeddings} emb - The embedding to search
 * @param {Array} vizInstances - Array to push new EmbeddingViz instances onto
 */
function setupLSI(emb, vizInstances, { Embeddings, EmbeddingViz }) {
  const seedInput = document.getElementById('lsi-seeds');
  const nInput = document.getElementById('lsi-n');
  const statusEl = document.getElementById('lsi-status');
  const eigEl = document.getElementById('lsi-eigenvalues');
  const compEl = document.getElementById('lsi-comparison');
  const plotEl = document.getElementById('plot-lsi');
  const eigenEl = document.getElementById('eigen-lsi');
  let currentK = 3;

  function runLSI() {
    const text = seedInput.value.trim().toLowerCase();
    if (!text) { statusEl.textContent = ''; eigEl.innerHTML = ''; compEl.innerHTML = ''; plotEl.innerHTML = ''; eigenEl.innerHTML = ''; return; }

    const seeds = text.split(/\s+/).filter(Boolean);
    const missing = seeds.filter(w => !emb.has(w));
    const valid = seeds.filter(w => emb.has(w));

    if (valid.length < 2) {
      statusEl.innerHTML = '<span class="error">Need at least 2 words in vocabulary.</span>';
      if (missing.length > 0) statusEl.innerHTML += ` Not found: ${missing.join(', ')}`;
      eigEl.innerHTML = ''; compEl.innerHTML = ''; plotEl.innerHTML = ''; eigenEl.innerHTML = '';
      return;
    }

    const k = Math.min(currentK, valid.length - 1);
    const n = parseInt(nInput.value) || 20;

    const { results: lsiResults, eigenvalues } = lsiExpand(emb, valid, { k, n });

    let status = `${valid.length} seed words`;
    if (missing.length > 0) status += ` (not found: ${missing.join(', ')})`;
    statusEl.textContent = status;

    // --- Eigenvalue histogram (clickable to set k) ---
    eigEl.innerHTML = '';
    const { eigenvalues: fullEigs } = lsiExpand(emb, valid, { k: valid.length - 1, n: 0 });
    const allEigs = (fullEigs.length > 0 ? fullEigs : eigenvalues).filter(e => e > 1e-10);
    if (allEigs.length > 0) {
      const eigTotal = allEigs.reduce((a, b) => a + b, 0);
      const barW = Math.min(400, (eigEl.clientWidth || 400));
      const barH = 90;
      const margin = { left: 5, right: 5, top: 5, bottom: 20 };
      const innerW = barW - margin.left - margin.right;
      const innerH = barH - margin.top - margin.bottom;
      const maxEig = Math.max(...allEigs);
      const colW = Math.min(28, (innerW / allEigs.length) - 2);
      const gap = Math.max(2, (innerW - colW * allEigs.length) / (allEigs.length - 1 || 1));

      const svg = d3.select(eigEl).append('svg')
        .attr('width', barW).attr('height', barH)
        .style('cursor', 'pointer');
      const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

      const cols = g.selectAll('g.eig-col').data(allEigs).enter().append('g')
        .attr('class', 'eig-col')
        .attr('transform', (d, i) => `translate(${i * (colW + gap)}, 0)`)
        .style('cursor', 'pointer');

      cols.append('rect')
        .attr('width', colW).attr('height', innerH + 16)
        .attr('fill', 'transparent');

      const bars = cols.append('rect')
        .attr('class', 'bar')
        .attr('y', d => innerH - (maxEig > 0 ? Math.max(3, d / maxEig * innerH) : 3))
        .attr('width', colW)
        .attr('height', d => maxEig > 0 ? Math.max(3, d / maxEig * innerH) : 3)
        .attr('fill', (d, i) => i < k ? '#4a6a8a' : '#ddd')
        .attr('rx', 2);

      let cumSum = 0;
      const labels = cols.append('text')
        .attr('x', colW / 2)
        .attr('y', innerH + 14)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .attr('fill', (d, i) => i < k ? '#4a6a8a' : '#aaa')
        .text(d => {
          cumSum += Math.max(0, d);
          return eigTotal > 0 ? `${(cumSum / eigTotal * 100).toFixed(0)}%` : '';
        });

      cols.on('click', function(event, d) {
        const idx = allEigs.indexOf(d);
        currentK = idx + 1;
        bars.attr('fill', (d, i) => i < currentK ? '#4a6a8a' : '#ddd');
        labels.attr('fill', (d, i) => i < currentK ? '#4a6a8a' : '#aaa');
        captionEl.html(`k = ${currentK}: top ${currentK} of ${allEigs.length} dimensions capture <b>${(allEigs.slice(0, currentK).reduce((a, b) => a + b, 0) / eigTotal * 100).toFixed(1)}%</b> of seed variance <span style="color:#bbb">(click bars to change k)</span>`);
        runLSIResults(valid, currentK, n);
      });

      const kCum = allEigs.slice(0, k).reduce((a, b) => a + b, 0);
      const captionEl = d3.select(eigEl).append('div')
        .style('font-size', '12px').style('color', '#999').style('margin-top', '2px');
      captionEl.html(`k = ${k}: top ${k} of ${allEigs.length} dimensions capture <b>${(kCum / eigTotal * 100).toFixed(1)}%</b> of seed variance <span style="color:#bbb">(click bars to change k)</span>`);
    }

    runLSIResults(valid, k, n);
  }

  function runLSIResults(valid, k, n) {
    const { results: lsiResults } = lsiExpand(emb, valid, { k, n });

    const centroid = new Float32Array(emb.dims);
    for (const w of valid) {
      const v = emb.vec(w);
      for (let i = 0; i < emb.dims; i++) centroid[i] += v[i];
    }
    for (let i = 0; i < emb.dims; i++) centroid[i] /= valid.length;
    Embeddings.normalize(centroid);
    const nnResults = emb.mostSimilar(centroid, n, new Set(valid));

    const thStyle = 'padding: 4px 10px; text-align: left; border-bottom: 1px solid #ddd; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #555;';
    let html = `<table style="border-collapse: collapse; font-size: 14px; width: 100%; max-width: 560px;">`;
    html += `<thead><tr><th style="${thStyle}">#</th>`;
    html += `<th style="${thStyle}">LSI expansion</th>`;
    html += `<th style="${thStyle}">Nearest to centroid</th></tr></thead><tbody>`;
    const nnSet = new Set(nnResults);
    const lsiSet = new Set(lsiResults.map(r => r.word));
    for (let i = 0; i < Math.max(lsiResults.length, nnResults.length); i++) {
      const lsiWord = lsiResults[i]?.word || '';
      const r = lsiResults[i];
      const nnWord = nnResults[i] || '';
      const lsiStyle = lsiWord && !nnSet.has(lsiWord) ? ' color: #4a6a8a; font-weight: 600;' : '';
      const nnStyle = nnWord && !lsiSet.has(nnWord) ? ' color: #5e8c61; font-weight: 600;' : '';
      const detail = r ? ` <span style="color:#999;font-size:10px" title="reconstruction ratio">${(r.score * 100).toFixed(1)}%</span>` : '';
      html += `<tr>`;
      html += `<td style="padding: 2px 10px; border-bottom: 1px solid #f0f0f0; color: #999;">${i + 1}</td>`;
      html += `<td style="padding: 2px 10px; border-bottom: 1px solid #f0f0f0;${lsiStyle}">${lsiWord}${detail}</td>`;
      html += `<td style="padding: 2px 10px; border-bottom: 1px solid #f0f0f0;${nnStyle}">${nnWord}</td>`;
      html += `</tr>`;
    }
    html += '</tbody></table>';
    html += '<div style="font-size: 12px; color: #999; margin-top: 4px;"><span style="color: #4a6a8a; font-weight: 600;">Blue</span> = LSI only, <span style="color: #5e8c61; font-weight: 600;">orange</span> = centroid NN only</div>';
    compEl.innerHTML = html;

    // MDS plot of seeds + expanded words
    plotEl.innerHTML = '';
    eigenEl.innerHTML = '';
    const lsiWords = lsiResults.map(r => r.word).filter(w => emb.has(w));
    const allWords = [...valid, ...lsiWords.filter(w => !valid.includes(w))];
    if (allWords.length >= 3) {
      vizInstances.push(new EmbeddingViz({
        emb, searchEmb: emb,
        words: allWords,
        groups: [valid, lsiWords.filter(w => !valid.includes(w))],
        plotEl, eigenEl,
        highlights: Array.from({ length: valid.length }, (_, i) => i),
      }));
    }
  }

  let debounceTimer = null;
  function debouncedRun() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(runLSI, 300);
  }
  seedInput.addEventListener('input', debouncedRun);
  nInput.addEventListener('input', debouncedRun);

  runLSI();
}

export { lsiExpand, setupLSI };
