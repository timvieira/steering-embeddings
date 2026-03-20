/**
 * Word embedding operations: loading, nearest neighbors, analogies, MDS, steering.
 * All computation runs client-side in the browser.
 */

class Embeddings {
  constructor(words, vectors, dims) {
    this.words = words;            // string[]
    this.vectors = vectors;        // Float32Array (flat, row-major)
    this.dims = dims;              // number of dimensions per word
    this.numWords = words.length;

    // Build word -> index lookup
    this.wordIndex = new Map();
    for (let i = 0; i < words.length; i++) {
      this.wordIndex.set(words[i], i);
    }
  }

  has(word) {
    return this.wordIndex.has(word);
  }

  // Add a word with its vector (copies from source embedding if provided)
  addWord(word, sourceEmb) {
    if (this.has(word)) return;
    const srcVec = sourceEmb ? sourceEmb.vec(word) : null;
    if (!srcVec) return;
    // Expand the vectors array
    const newVecs = new Float32Array(this.vectors.length + this.dims);
    newVecs.set(this.vectors);
    newVecs.set(srcVec, this.vectors.length);
    this.vectors = newVecs;
    this.words.push(word);
    this.wordIndex.set(word, this.numWords);
    this.numWords++;
  }

  // Get the vector for a word (returns a Float32Array view)
  vec(word) {
    const idx = this.wordIndex.get(word);
    if (idx === undefined) return null;
    return this.vectors.subarray(idx * this.dims, (idx + 1) * this.dims);
  }

  // Euclidean distance between two vectors
  static dist(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const d = a[i] - b[i];
      sum += d * d;
    }
    return Math.sqrt(sum);
  }

  // Normalize a vector in place, return it
  static normalize(v) {
    let norm = 0;
    for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < v.length; i++) v[i] /= norm;
    return v;
  }

  // Find n most similar words to a vector, excluding specified words
  mostSimilar(queryVec, n = 5, exclude = new Set()) {
    const scores = [];
    for (let i = 0; i < this.numWords; i++) {
      if (exclude.has(this.words[i])) continue;
      const wv = this.vectors.subarray(i * this.dims, (i + 1) * this.dims);
      // cosine similarity (vectors assumed normalized)
      let dot = 0;
      for (let j = 0; j < this.dims; j++) dot += queryVec[j] * wv[j];
      scores.push({ word: this.words[i], score: dot });
    }
    scores.sort((a, b) => b.score - a.score);
    return scores.slice(0, n).map(s => s.word);
  }

  // Solve analogy: a is to b as c is to ?
  analogy(a, b, c, n = 1) {
    const va = this.vec(a), vb = this.vec(b), vc = this.vec(c);
    if (!va || !vb || !vc) return [];
    // result ≈ vb - va + vc
    const query = new Float32Array(this.dims);
    for (let i = 0; i < this.dims; i++) query[i] = vb[i] - va[i] + vc[i];
    Embeddings.normalize(query);
    return this.mostSimilar(query, n, new Set([a, b, c]));
  }

  // Compute pairwise distance matrix for a list of words
  distanceMatrix(words) {
    const n = words.length;
    const D = new Float64Array(n * n);
    const vecs = words.map(w => this.vec(w));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < i; j++) {
        const d = Embeddings.dist(vecs[i], vecs[j]);
        D[i * n + j] = d;
        D[j * n + i] = d;
      }
    }
    return { matrix: D, n };
  }

  // Steer (debias) embeddings by projecting out a subspace.
  // pairs: array of [word1, word2] pairs defining the subspace
  // K: number of dimensions to project out
  steer(pairs, K = 10) {
    // Collect the pair vectors
    const pairVecs = [];
    for (const [w1, w2] of pairs) {
      const v1 = this.vec(w1), v2 = this.vec(w2);
      if (v1 && v2) pairVecs.push([v1, v2]);
    }

    // Compute covariance matrix: sum of np.cov for each pair.
    // For a pair [v1, v2], np.cov gives the outer product of the
    // difference: (v1-v2)(v1-v2)^T / (N-1), where N=2, so / 1.
    // We sum these across all pairs, matching the Python:
    //   C += np.cov(self.vec[self.dom.encode_many(g)].T)
    const d = this.dims;
    const C = new Float64Array(d * d);
    for (const [v1, v2] of pairVecs) {
      const diff = new Float64Array(d);
      for (let i = 0; i < d; i++) diff[i] = v1[i] - v2[i];
      // np.cov with 2 observations and ddof=1: cov = diff * diff^T / 2
      // But we also add the mean-centered outer products which equal this.
      // More precisely: mean = (v1+v2)/2, centered = ±diff/2
      // cov[i,j] = 2 * (diff[i]/2)*(diff[j]/2) / 1 = diff[i]*diff[j] / 2
      for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
          C[i * d + j] += diff[i] * diff[j] / 2;
        }
      }
    }

    // SVD of covariance matrix to find top K basis vectors
    const Cmat = [];
    for (let i = 0; i < d; i++) {
      Cmat.push([]);
      for (let j = 0; j < d; j++) {
        Cmat[i].push(C[i * d + j]);
      }
    }
    const { u: svdU } = SVDJS.SVD(Cmat);
    const basis = [];
    for (let k = 0; k < K; k++) {
      const v = new Float64Array(d);
      for (let i = 0; i < d; i++) v[i] = svdU[i][k];
      basis.push(v);
    }

    // Project out the basis from all vectors
    const newVecs = new Float32Array(this.numWords * d);
    for (let w = 0; w < this.numWords; w++) {
      const offset = w * d;
      // Copy original
      for (let i = 0; i < d; i++) newVecs[offset + i] = this.vectors[offset + i];
      // Subtract projection onto each basis vector
      for (const b of basis) {
        let dot = 0;
        for (let i = 0; i < d; i++) dot += newVecs[offset + i] * b[i];
        for (let i = 0; i < d; i++) newVecs[offset + i] -= dot * b[i];
      }
      // Re-normalize
      let norm = 0;
      for (let i = 0; i < d; i++) norm += newVecs[offset + i] * newVecs[offset + i];
      norm = Math.sqrt(norm);
      if (norm > 0) for (let i = 0; i < d; i++) newVecs[offset + i] /= norm;
    }

    return new Embeddings(this.words, newVecs, d);
  }
}


/**
 * MDS: given a distance matrix, compute low-dimensional coordinates.
 * Uses svd-js for exact SVD (must be loaded as SVDJS global).
 * Returns { coords: number[][], eigenvalues: number[] }
 */
function mds(distMatrix, n, dimensions = 2) {
  // Double-center the squared distance matrix
  const D2 = [];
  for (let i = 0; i < n; i++) {
    D2.push([]);
    for (let j = 0; j < n; j++) {
      D2[i].push(-0.5 * distMatrix[i * n + j] * distMatrix[i * n + j]);
    }
  }

  // Row means, column means, grand mean
  const rowMeans = new Array(n).fill(0);
  const colMeans = new Array(n).fill(0);
  let grandMean = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      rowMeans[i] += D2[i][j];
      colMeans[j] += D2[i][j];
      grandMean += D2[i][j];
    }
  }
  for (let i = 0; i < n; i++) { rowMeans[i] /= n; colMeans[i] /= n; }
  grandMean /= (n * n);

  // Double-centered matrix B (as 2D array for svd-js)
  const B = [];
  for (let i = 0; i < n; i++) {
    B.push([]);
    for (let j = 0; j < n; j++) {
      B[i].push(D2[i][j] - rowMeans[i] - colMeans[j] + grandMean);
    }
  }

  // SVD via svd-js: B = U * diag(q) * V^T
  const { u, q } = SVDJS.SVD(B);

  // Eigenvalues are the singular values (B is symmetric PSD, so singular values = eigenvalues)
  const eigenvalues = q.slice(0, Math.min(dimensions, n));

  // Coordinates: X_ik = sqrt(eigenvalue_k) * U_ik
  const coords = [];
  for (let i = 0; i < n; i++) {
    const point = [];
    for (let k = 0; k < dimensions; k++) {
      point.push(u[i][k] * Math.sqrt(Math.max(0, q[k])));
    }
    coords.push(point);
  }

  // Return all singular values for variance reporting
  return { coords, eigenvalues: q };
}


/**
 * Load embeddings from a binary file with a progress callback.
 * Returns a Promise<Embeddings>.
 */
async function loadEmbeddings(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);

  const contentLength = response.headers.get('content-length');
  const total = contentLength ? parseInt(contentLength) : 0;
  const reader = response.body.getReader();

  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (onProgress && total) onProgress(received / total);
  }

  // Concatenate chunks
  const buffer = new ArrayBuffer(received);
  const view = new Uint8Array(buffer);
  let offset = 0;
  for (const chunk of chunks) {
    view.set(chunk, offset);
    offset += chunk.length;
  }

  // Parse header
  const dataView = new DataView(buffer);
  let pos = 0;
  const numWords = dataView.getUint32(pos, true); pos += 4;
  const numDims = dataView.getUint32(pos, true); pos += 4;

  // Parse vocabulary
  const words = [];
  const decoder = new TextDecoder('utf-8');
  for (let i = 0; i < numWords; i++) {
    const wordLen = dataView.getUint16(pos, true); pos += 2;
    const wordBytes = new Uint8Array(buffer, pos, wordLen);
    words.push(decoder.decode(wordBytes));
    pos += wordLen;
  }

  // Parse vectors (copy to ensure alignment)
  const vecBytes = new Uint8Array(buffer, pos, numWords * numDims * 4);
  const vectors = new Float32Array(vecBytes.buffer.slice(pos, pos + numWords * numDims * 4));

  // Normalize vectors
  for (let i = 0; i < numWords; i++) {
    let norm = 0;
    const off = i * numDims;
    for (let j = 0; j < numDims; j++) norm += vectors[off + j] * vectors[off + j];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let j = 0; j < numDims; j++) vectors[off + j] /= norm;
  }

  if (onProgress) onProgress(1);
  return new Embeddings(words, vectors, numDims);
}

export { Embeddings, mds, loadEmbeddings };
