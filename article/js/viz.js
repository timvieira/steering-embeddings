/**
 * Embedding visualizations using D3 (2D) and Three.js (3D).
 * Unified component for paths, analogies, and steering comparisons.
 */
import { mds } from './embeddings.js';

const COLORS = {
  point: '#5778a4',
  arrow: 'rgba(87, 120, 164, 0.5)',
  highlight: '#e49444',
  crossGroup: 'rgba(228, 148, 68, 0.3)',
  eigenActive: '#5778a4',
  eigenInactive: '#ddd',
};

/**
 * Compute MDS for a set of words at dimensions 1, 2, and 3.
 * Returns { coords1, coords2, coords3, eigenvalues, varianceExplained }
 */
function computeAllMDS(emb, words) {
  const { matrix, n } = emb.distanceMatrix(words);
  const r1 = mds(matrix, n, 1);
  const r2 = mds(matrix, n, 2);
  const r3 = mds(matrix, n, 3);
  // All eigenvalues from the full SVD (r3 has all of them)
  const allEig = r3.eigenvalues;
  const eigSum = allEig.reduce((a, b) => a + Math.max(0, b), 0);
  // Variance explained: sum of top-k eigenvalues / sum of all eigenvalues
  // This matches the Python: S[:dimensions].sum() / S.sum()
  function varianceForDims(k) {
    if (eigSum <= 0) return 100;
    let topK = 0;
    for (let i = 0; i < k && i < allEig.length; i++) topK += Math.max(0, allEig[i]);
    return topK / eigSum * 100;
  }
  return {
    coords: { 1: r1.coords, 2: r2.coords, 3: r3.coords },
    eigenvalues: allEig,
    variance: { 1: varianceForDims(1), 2: varianceForDims(2), 3: varianceForDims(3) },
  };
}


/**
 * MDS Eigenvalue selector widget (SVG in the margin).
 */
function createEigenSelector(container, eigenvalues, activeDims, onChange) {
  // Only show top 3 eigenvalues (for 1D/2D/3D switching)
  const topEig = eigenvalues.slice(0, 3);
  const width = 80, height = 60, barWidth = 20, gap = 4;
  const maxVal = Math.max(...topEig.map(v => Math.max(0, v)));

  const svg = d3.select(container).append('svg')
    .attr('width', width).attr('height', height + 20)
    .style('cursor', 'pointer');

  const bars = svg.selectAll('rect.bar')
    .data(topEig)
    .enter().append('rect')
    .attr('class', 'bar')
    .attr('x', (d, i) => i * (barWidth + gap))
    .attr('y', d => height - (maxVal > 0 ? Math.max(0, d) / maxVal * height : 0))
    .attr('width', barWidth)
    .attr('height', d => maxVal > 0 ? Math.max(0, d) / maxVal * height : 0)
    .attr('fill', (d, i) => i < activeDims ? COLORS.eigenActive : COLORS.eigenInactive)
    .attr('rx', 2);

  // Labels
  svg.selectAll('text.label')
    .data(topEig)
    .enter().append('text')
    .attr('class', 'label')
    .attr('x', (d, i) => i * (barWidth + gap) + barWidth / 2)
    .attr('y', height + 14)
    .attr('text-anchor', 'middle')
    .attr('font-size', '10px')
    .attr('fill', '#666')
    .text((d, i) => `${i + 1}D`);

  // Click handler
  svg.selectAll('rect.bar').on('click', function(event, d) {
    const idx = topEig.indexOf(d);
    const newDims = idx + 1;
    // Update colors
    bars.attr('fill', (d, i) => i < newDims ? COLORS.eigenActive : COLORS.eigenInactive);
    onChange(newDims);
  });

  return { update(dims) {
    bars.attr('fill', (d, i) => i < dims ? COLORS.eigenActive : COLORS.eigenInactive);
  }};
}


/**
 * 2D scatter plot with arrows using D3 SVG.
 */
function render2D(container, words, coords, arrows, options = {}) {
  const {
    highlights = [], crossGroupLines = [], width = 620, height = 450,
    neighborWords = new Set(), neighborLinks = [],
    animate = false, prevCoords = null, prevWords = null,
  } = options;
  const margin = { top: 30, right: 30, bottom: 30, left: 30 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;
  const dur = animate ? 600 : 0;
  const highlightSet = new Set(highlights);

  // Scales (always based on final coords)
  const xs = coords.map(c => c[0]);
  const ys = coords.map(c => c[1]);
  const pad = 0.1;
  const xRange = [Math.min(...xs), Math.max(...xs)];
  const yRange = [Math.min(...ys), Math.max(...ys)];
  const xPad = (xRange[1] - xRange[0]) * pad || 0.1;
  const yPad = (yRange[1] - yRange[0]) * pad || 0.1;
  const xScale = d3.scaleLinear()
    .domain([xRange[0] - xPad, xRange[1] + xPad]).range([0, w]);
  const yScale = d3.scaleLinear()
    .domain([yRange[0] - yPad, yRange[1] + yPad]).range([h, 0]);

  // Map previous words to their old MDS coords for animation start positions
  const prevWordIdx = prevWords ? new Map(prevWords.map((ww, i) => [ww, i])) : null;
  function startXY(i) {
    if (!animate) return [xScale(coords[i][0]), yScale(coords[i][1])];
    if (prevCoords && prevWordIdx) {
      const pi = prevWordIdx.get(words[i]);
      if (pi !== undefined) {
        // Existing word: start at old position mapped through new scale
        return [xScale(prevCoords[pi][0]), yScale(prevCoords[pi][1])];
      }
      // New word: start at parent's final position
      const link = neighborLinks.find(l => l.child === i);
      if (link) {
        return [xScale(coords[link.parent][0]), yScale(coords[link.parent][1])];
      }
    }
    return [xScale(coords[i][0]), yScale(coords[i][1])];
  }

  // Unique container ID for arrow marker references
  const cid = container.id || 'plot';

  // Reuse SVG for animated updates, otherwise rebuild
  let svg = d3.select(container).select('svg.plot');
  let g;
  if (svg.empty() || !animate) {
    d3.select(container).selectAll('svg.plot').remove();
    svg = d3.select(container).append('svg')
      .attr('class', 'plot')
      .attr('width', width).attr('height', height);
    const defs = svg.append('defs');
    for (const [suffix, color] of [['', COLORS.arrow], ['-red', COLORS.highlight]]) {
      defs.append('marker')
        .attr('id', `arrow${suffix}-${cid}`)
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 8).attr('refY', 5)
        .attr('markerWidth', 6).attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 Z').attr('fill', color);
    }
    // Zoom container wraps the main group so pan/zoom transforms apply to all content
    const zoomG = svg.append('g').attr('class', 'zoom-container');
    g = zoomG.append('g').attr('class', 'main')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Pan + zoom via d3.zoom, but don't intercept clicks on words/circles
    const zoom = d3.zoom()
      .scaleExtent([0.5, 5])
      .filter((event) => {
        // Allow scroll-to-zoom always; only allow drag from background (not words/circles)
        if (event.type === 'wheel') return true;
        if (event.type === 'mousedown' || event.type === 'pointerdown') {
          const tag = event.target.tagName;
          return tag !== 'circle' && tag !== 'text';
        }
        return true;
      })
      .on('zoom', (event) => { zoomG.attr('transform', event.transform); });
    svg.call(zoom);
    svg.style('cursor', 'grab');
    svg.on('mousedown.cursor', () => svg.style('cursor', 'grabbing'));
    svg.on('mouseup.cursor', () => svg.style('cursor', 'grab'));
  } else {
    g = svg.select('g.main');
  }

  // --- Neighbor links (dashed lines from parent to child) ---
  const linkData = neighborLinks.filter(l => l.child < coords.length && l.parent < coords.length);
  const nlinks = g.selectAll('line.neighbor-link').data(linkData, d => `${d.parent}-${d.child}`);
  nlinks.exit().remove();
  const nlinksEnter = nlinks.enter().append('line').attr('class', 'neighbor-link')
    .attr('stroke', '#ccc').attr('stroke-width', 1).attr('stroke-dasharray', '3,2');
  nlinksEnter.merge(nlinks).transition().duration(dur)
    .attr('x1', d => xScale(coords[d.parent][0])).attr('y1', d => yScale(coords[d.parent][1]))
    .attr('x2', d => xScale(coords[d.child][0])).attr('y2', d => yScale(coords[d.child][1]));

  // --- Cross-group lines ---
  const cgData = crossGroupLines.filter(([i, j]) => i < coords.length && j < coords.length);
  const cg = g.selectAll('line.cross-group').data(cgData, d => `cg-${d[0]}-${d[1]}`);
  cg.exit().remove();
  const cgEnter = cg.enter().append('line').attr('class', 'cross-group')
    .attr('stroke', COLORS.crossGroup).attr('stroke-width', 1).attr('stroke-dasharray', '4,3');
  cgEnter.merge(cg).transition().duration(dur)
    .attr('x1', d => xScale(coords[d[0]][0])).attr('y1', d => yScale(coords[d[0]][1]))
    .attr('x2', d => xScale(coords[d[1]][0])).attr('y2', d => yScale(coords[d[1]][1]));

  // --- Arrows ---
  const arrowData = arrows.filter(a => a.from < coords.length && a.to < coords.length);
  const arrowSel = g.selectAll('line.arrow').data(arrowData, (d, i) => `a-${d.from}-${d.to}`);
  arrowSel.exit().remove();
  const arrowEnter = arrowSel.enter().append('line').attr('class', 'arrow')
    .attr('stroke-width', 1.5)
    .attr('stroke', d => d.color === 'red' ? COLORS.highlight : COLORS.arrow)
    .attr('marker-end', d => `url(#arrow${d.color === 'red' ? '-red' : ''}-${cid})`);
  arrowEnter.merge(arrowSel).transition().duration(dur)
    .attr('x1', d => xScale(coords[d.from][0])).attr('y1', d => yScale(coords[d.from][1]))
    .attr('x2', d => xScale(coords[d.to][0])).attr('y2', d => yScale(coords[d.to][1]));

  // --- Points (keyed by word for stable enter/update/exit) ---
  const pointData = coords.map((c, i) => ({ c, i, word: words[i] }));
  const circles = g.selectAll('circle.point').data(pointData, d => d.word);
  circles.exit().transition().duration(dur).attr('r', 0).remove();
  const circlesEnter = circles.enter().append('circle').attr('class', 'point')
    .attr('cx', d => startXY(d.i)[0])
    .attr('cy', d => startXY(d.i)[1])
    .attr('r', 0)
    .style('cursor', 'pointer');
  circlesEnter.merge(circles).transition().duration(dur)
    .attr('cx', d => xScale(d.c[0]))
    .attr('cy', d => yScale(d.c[1]))
    .attr('r', d => highlightSet.has(d.i) ? 5 : 3.5)
    .attr('fill', d => highlightSet.has(d.i) ? COLORS.highlight :
      neighborWords.has(d.word) ? '#999' : COLORS.point);

  // --- Labels (keyed by word) ---
  const labels = g.selectAll('text.word-label').data(pointData, d => d.word);
  labels.exit().transition().duration(dur).style('opacity', 0).remove();
  const labelsEnter = labels.enter().append('text').attr('class', 'word-label')
    .attr('x', d => startXY(d.i)[0])
    .attr('y', d => startXY(d.i)[1] - 8)
    .attr('text-anchor', 'middle')
    .style('opacity', 0)
    .text(d => d.word);
  labelsEnter.merge(labels).transition().duration(dur)
    .attr('x', d => xScale(d.c[0]))
    .attr('y', d => yScale(d.c[1]) - 8)
    .attr('font-size', d => neighborWords.has(d.word) ? '10px' : '11px')
    .attr('fill', d => neighborWords.has(d.word) ? '#666' : '#333')
    .style('opacity', 1);

  // --- Click handlers ---
  if (options.onClick) {
    g.selectAll('circle.point').style('cursor', 'pointer')
      .on('click', (event, d) => options.onClick(d.i, d.word));
    g.selectAll('text.word-label').style('cursor', 'pointer')
      .on('click', (event, d) => options.onClick(d.i, d.word));
  }
}


/**
 * 3D scatter plot with arrows using Three.js.
 */
function render3D(container, words, coords, arrows, options = {}) {
  const { highlights = [], crossGroupLines = [], width = 620, height = 450 } = options;

  // Clear
  const el = typeof container === 'string' ? document.getElementById(container) : container;
  el.querySelectorAll('canvas').forEach(c => c.remove());

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.domElement.style.border = '1px solid #e0e0e0';
  renderer.domElement.style.borderRadius = '4px';
  renderer.domElement.style.cursor = 'grab';
  renderer.domElement.addEventListener('pointerdown', () => { renderer.domElement.style.cursor = 'grabbing'; });
  renderer.domElement.addEventListener('pointerup', () => { renderer.domElement.style.cursor = 'grab'; });
  el.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  // Scale coords to fit nicely
  const flat = coords.flat();
  const maxAbs = Math.max(...flat.map(Math.abs)) || 1;
  const scale = 2 / maxAbs;

  const highlightSet = new Set(highlights);

  // Points (store meshes for raycasting)
  const clickableMeshes = [];
  for (let i = 0; i < words.length; i++) {
    const [x, y, z] = coords[i].map(v => v * scale);
    const isHL = highlightSet.has(i);
    const geo = new THREE.SphereGeometry(isHL ? 0.06 : 0.04, 16, 16);
    const mat = new THREE.MeshBasicMaterial({
      color: isHL ? COLORS.highlight : COLORS.point
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, y, z);
    mesh.userData = { index: i, word: words[i] };
    scene.add(mesh);
    clickableMeshes.push(mesh);

    // Label (sprite)
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 512; canvas.height = 128;
    ctx.font = 'bold 48px sans-serif';
    ctx.fillStyle = '#333';
    ctx.textAlign = 'center';
    ctx.fillText(words[i], 256, 80);
    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.position.set(x, y + 0.15, z);
    sprite.scale.set(0.8, 0.2, 1);
    scene.add(sprite);
  }

  // Arrows
  for (const { from, to, color } of arrows) {
    const c = coords;
    const start = new THREE.Vector3(...c[from].map(v => v * scale));
    const end = new THREE.Vector3(...c[to].map(v => v * scale));
    const dir = new THREE.Vector3().subVectors(end, start);
    const len = dir.length();
    if (len === 0) continue;

    const arrowColor = color === 'red' ? COLORS.highlight : COLORS.point;
    const headLen = Math.min(0.08, len * 0.15);
    const headWidth = headLen * 0.5;
    const arrow = new THREE.ArrowHelper(
      dir.normalize(), start, len, arrowColor, headLen, headWidth
    );
    scene.add(arrow);
  }

  // Cross-group lines
  for (const [i, j] of crossGroupLines) {
    const points = [
      new THREE.Vector3(...coords[i].map(v => v * scale)),
      new THREE.Vector3(...coords[j].map(v => v * scale)),
    ];
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineDashedMaterial({
      color: COLORS.crossGroup, dashSize: 0.05, gapSize: 0.03
    });
    const line = new THREE.Line(geo, mat);
    line.computeLineDistances();
    scene.add(line);
  }

  // Camera position
  camera.position.set(3, 2, 3);
  camera.lookAt(0, 0, 0);

  // Auto-orbit
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.0;

  // Pause auto-orbit on interaction, resume after 3s
  let pauseTimeout = null;
  function pauseOrbit() {
    controls.autoRotate = false;
    clearTimeout(pauseTimeout);
    pauseTimeout = setTimeout(() => { controls.autoRotate = true; }, 3000);
  }
  renderer.domElement.addEventListener('pointerdown', pauseOrbit);
  renderer.domElement.addEventListener('wheel', pauseOrbit);

  // Click-to-expand via raycasting
  if (options.onClick) {
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let pointerDownPos = null;

    renderer.domElement.addEventListener('pointerdown', (e) => {
      pointerDownPos = { x: e.clientX, y: e.clientY };
    });

    renderer.domElement.addEventListener('pointerup', (e) => {
      // Only treat as click if pointer didn't move much (not a drag)
      if (!pointerDownPos) return;
      const dx = e.clientX - pointerDownPos.x;
      const dy = e.clientY - pointerDownPos.y;
      if (Math.sqrt(dx * dx + dy * dy) > 5) return;

      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(clickableMeshes);
      if (hits.length > 0) {
        const { index, word } = hits[0].object.userData;
        options.onClick(index, word);
      }
    });
  }

  // Animate
  let animId = null;
  function animate() {
    animId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Return handle for cleanup and external access
  return { scene, camera, renderer, controls, el,
    destroy() { cancelAnimationFrame(animId); renderer.dispose(); clearTimeout(pauseTimeout); }
  };
}


/**
 * 1D strip plot using D3.
 */
function render1D(container, words, coords, arrows, options = {}) {
  const { highlights = [], width = 620, height = 120 } = options;
  const margin = { top: 30, right: 30, bottom: 30, left: 30 };
  const w = width - margin.left - margin.right;

  d3.select(container).selectAll('svg.plot').remove();

  const svg = d3.select(container).append('svg')
    .attr('class', 'plot')
    .attr('width', width).attr('height', height);

  const zoomG = svg.append('g').attr('class', 'zoom-container');
  const g = zoomG.append('g')
    .attr('transform', `translate(${margin.left},${height / 2})`);

  const zoom = d3.zoom()
    .scaleExtent([0.5, 5])
    .on('zoom', (event) => { zoomG.attr('transform', event.transform); });
  svg.call(zoom);
  svg.style('cursor', 'grab');
  svg.on('mousedown.cursor', () => svg.style('cursor', 'grabbing'));
  svg.on('mouseup.cursor', () => svg.style('cursor', 'grab'));

  const xs = coords.map(c => c[0]);
  const pad = 0.1;
  const xRange = [Math.min(...xs), Math.max(...xs)];
  const xPad = (xRange[1] - xRange[0]) * pad || 0.1;
  const xScale = d3.scaleLinear()
    .domain([xRange[0] - xPad, xRange[1] + xPad]).range([0, w]);

  const highlightSet = new Set(highlights);

  const pointData = coords.map((c, i) => ({ c, i, word: words[i] }));

  g.selectAll('circle')
    .data(pointData)
    .enter().append('circle')
    .attr('cx', d => xScale(d.c[0]))
    .attr('cy', 0)
    .attr('r', d => highlightSet.has(d.i) ? 5 : 3)
    .attr('fill', d => highlightSet.has(d.i) ? COLORS.highlight : COLORS.point)
    .style('cursor', options.onClick ? 'pointer' : 'default')
    .on('click', options.onClick ? (event, d) => { event.stopPropagation(); options.onClick(d.i, d.word); } : null);

  g.selectAll('text.word-label')
    .data(pointData)
    .enter().append('text')
    .attr('class', 'word-label')
    .attr('x', d => xScale(d.c[0]))
    .attr('y', -12)
    .attr('text-anchor', 'middle')
    .style('cursor', options.onClick ? 'pointer' : 'default')
    .on('click', options.onClick ? (event, d) => { event.stopPropagation(); options.onClick(d.i, d.word); } : null)
    .attr('font-size', '11px')
    .attr('fill', '#333')
    .text(d => d.word);
}


/**
 * Main visualization wrapper with MDS dimension switching.
 */
class EmbeddingViz {
  constructor(config) {
    this.emb = config.emb;
    this.words = config.words;
    this.groups = config.groups || [config.words];
    this.plotEl = config.plotEl;
    this.eigenEl = config.eigenEl;
    this.arrows = config.arrows || [];
    this.highlights = config.highlights || [];
    this.crossGroupLines = config.crossGroupLines || [];
    this.connectGroups = config.connectGroups || false;
    this.dims = config.initialDims || 2;
    this.neighborWords = new Set();   // words added via click expansion
    this.neighborLinks = [];          // { parent, child } index pairs
    this._prevCoords = null;
    this._prevWords = null;

    // Compute MDS at all dimensions
    this.mdsData = computeAllMDS(this.emb, this.words);

    // Build cross-group connections if requested
    if (this.connectGroups && this.groups.length > 1) {
      const wordIdx = new Map(this.words.map((w, i) => [w, i]));
      const n = Math.min(...this.groups.map(g => g.length));
      for (let i = 0; i < n; i++) {
        for (let g = 0; g < this.groups.length - 1; g++) {
          const a = wordIdx.get(this.groups[g][i]);
          const b = wordIdx.get(this.groups[g + 1][i]);
          if (a !== undefined && b !== undefined) {
            this.crossGroupLines.push([a, b]);
          }
        }
      }
    }

    // Build arrows from groups (consecutive words in each group)
    if (this.arrows.length === 0) {
      const wordIdx = new Map(this.words.map((w, i) => [w, i]));
      for (const group of this.groups) {
        for (let i = 0; i < group.length - 1; i++) {
          const a = wordIdx.get(group[i]);
          const b = wordIdx.get(group[i + 1]);
          if (a !== undefined && b !== undefined) {
            this.arrows.push({ from: a, to: b });
          }
        }
      }
    }

    // Create eigen selector
    if (this.eigenEl) {
      this.eigenSelector = createEigenSelector(
        this.eigenEl,
        this.mdsData.eigenvalues,
        this.dims,
        (newDims) => { this.dims = newDims; this.render(); }
      );
    }

    this.render();
  }

  render(animate = false) {
    const coords = this.mdsData.coords[this.dims];
    const opts = {
      highlights: this.highlights,
      crossGroupLines: this.crossGroupLines,
      neighborWords: this.neighborWords,
      neighborLinks: this.neighborLinks,
      animate,
      prevCoords: this._prevCoords,
      prevWords: this._prevWords,
    };

    const el = typeof this.plotEl === 'string'
      ? document.getElementById(this.plotEl)
      : this.plotEl;

    // For animated 2D updates, keep the SVG and just update data
    if (!animate || this.dims !== 2) {
      el.innerHTML = '';
      const caption = document.createElement('div');
      caption.className = 'variance-caption';
      caption.textContent = `${this.dims}D MDS captures ${this.mdsData.variance[this.dims].toFixed(1)}% of variance`;
      el.appendChild(caption);
    } else {
      const caption = el.querySelector('.variance-caption');
      if (caption) caption.textContent = `${this.dims}D MDS captures ${this.mdsData.variance[this.dims].toFixed(1)}% of variance`;
    }

    // Click handler: add neighbors, recompute MDS, animate
    const self = this;
    opts.onClick = (idx, word) => {
      const vec = self.emb.vec(word);
      if (!vec) return;
      const existing = new Set(self.words);
      const neighbors = self.emb.mostSimilar(vec, 5, existing);
      if (neighbors.length === 0) return;

      // Save current state for animation start positions
      self._prevCoords = self.mdsData.coords[self.dims];
      self._prevWords = [...self.words];

      // Add neighbors to word list
      const wordIdx = new Map(self.words.map((w, i) => [w, i]));
      const parentIdx = wordIdx.get(word);
      for (const nw of neighbors) {
        if (!existing.has(nw) && self.emb.has(nw)) {
          self.words.push(nw);
          self.neighborWords.add(nw);
          self.neighborLinks.push({ parent: parentIdx, child: self.words.length - 1 });
          existing.add(nw);
        }
      }

      // Recompute MDS with expanded word set and animate
      self.mdsData = computeAllMDS(self.emb, self.words);
      self.render(true);
    };

    if (this.dims === 1) {
      render1D(el, this.words, coords, this.arrows, opts);
    } else if (this.dims === 2) {
      render2D(el, this.words, coords, this.arrows, opts);
    } else {
      render3D(el, this.words, coords, this.arrows, opts);
    }

    if (this.eigenSelector) this.eigenSelector.update(this.dims);
  }
}

/**
 * Hero 3D visualization: shows original and steered positions with trails.
 * Words are shown at their steered positions (bright) with ghost dots at
 * original positions and thin trails connecting them.
 *
 * wordData: array of { word, origCoord: [x,y,z], steeredCoord: [x,y,z], group: string }
 */
function renderHero3D(container, wordData, options = {}) {
  const { width = 800, height = 500 } = options;

  const el = typeof container === 'string' ? document.getElementById(container) : container;
  el.querySelectorAll('canvas').forEach(c => c.remove());

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);
  const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  el.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.8;
  // Visible border so users know where the interactive canvas is
  renderer.domElement.style.border = '1px solid #e0e0e0';
  renderer.domElement.style.borderRadius = '4px';
  renderer.domElement.style.cursor = 'grab';
  renderer.domElement.addEventListener('pointerdown', () => { renderer.domElement.style.cursor = 'grabbing'; });
  renderer.domElement.addEventListener('pointerup', () => { renderer.domElement.style.cursor = 'grab'; });

  let pauseTimeout = null;
  function pauseOrbit() {
    controls.autoRotate = false;
    clearTimeout(pauseTimeout);
    pauseTimeout = setTimeout(() => { controls.autoRotate = true; }, 3000);
  }
  renderer.domElement.addEventListener('pointerdown', pauseOrbit);

  // Scale coords
  const allCoords = wordData.flatMap(d => [...d.origCoord, ...d.steeredCoord]);
  const maxAbs = Math.max(...allCoords.map(Math.abs)) || 1;
  const scale = 2 / maxAbs;

  // Group colors
  const groupNames = [...new Set(wordData.map(d => d.group))];
  const groupPalette = ['#5778a4', '#e49444', '#6a9f58', '#b07aa1', '#d1615d', '#85b6b2'];
  const groupColor = {};
  groupNames.forEach((g, i) => { groupColor[g] = groupPalette[i % groupPalette.length]; });

  for (const d of wordData) {
    const ox = d.origCoord.map(v => v * scale);
    const sx = d.steeredCoord.map(v => v * scale);
    const color = new THREE.Color(groupColor[d.group]);

    // Ghost dot (original position, faded)
    const ghostGeo = new THREE.SphereGeometry(0.03, 12, 12);
    const ghostMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.55 });
    const ghost = new THREE.Mesh(ghostGeo, ghostMat);
    ghost.position.set(...ox);
    scene.add(ghost);

    // Trail (original → steered)
    const trailGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(...ox), new THREE.Vector3(...sx)
    ]);
    const trailMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.45 });
    scene.add(new THREE.Line(trailGeo, trailMat));

    // Steered dot (bright)
    const dotGeo = new THREE.SphereGeometry(0.04, 16, 16);
    const dotMat = new THREE.MeshBasicMaterial({ color });
    const dot = new THREE.Mesh(dotGeo, dotMat);
    dot.position.set(...sx);
    scene.add(dot);

    // Label at steered position
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 512; canvas.height = 128;
    ctx.font = 'bold 44px sans-serif';
    ctx.fillStyle = groupColor[d.group];
    ctx.textAlign = 'center';
    ctx.fillText(d.word, 256, 80);
    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.position.set(sx[0], sx[1] + 0.14, sx[2]);
    sprite.scale.set(0.7, 0.175, 1);
    scene.add(sprite);
  }

  camera.position.set(3, 2, 3);
  camera.lookAt(0, 0, 0);

  let animId = null;
  function animate() {
    animId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  return { scene, camera, renderer, controls, el,
    destroy() { cancelAnimationFrame(animId); renderer.dispose(); clearTimeout(pauseTimeout); }
  };
}

/**
 * 2D animated steering visualization.
 * Shows words moving from original to steered positions with fading trails.
 */
function renderSteering2D(container, wordData, options = {}) {
  const { width = 620, height = 450 } = options;
  const margin = { top: 30, right: 30, bottom: 40, left: 30 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  const el = typeof container === 'string' ? document.getElementById(container) : container;
  d3.select(el).selectAll('svg.plot').remove();
  d3.select(el).selectAll('.steer-controls').remove();

  const allX = wordData.flatMap(d => [d.origCoord[0], d.steeredCoord[0]]);
  const allY = wordData.flatMap(d => [d.origCoord[1], d.steeredCoord[1]]);
  const pad = 0.1;
  const xRange = [Math.min(...allX), Math.max(...allX)];
  const yRange = [Math.min(...allY), Math.max(...allY)];
  const xPad = (xRange[1] - xRange[0]) * pad || 0.1;
  const yPad = (yRange[1] - yRange[0]) * pad || 0.1;
  const xScale = d3.scaleLinear().domain([xRange[0] - xPad, xRange[1] + xPad]).range([0, w]);
  const yScale = d3.scaleLinear().domain([yRange[0] - yPad, yRange[1] + yPad]).range([h, 0]);

  const groupNames = [...new Set(wordData.map(d => d.group))];
  const groupPalette = ['#5778a4', '#e49444', '#6a9f58', '#b07aa1', '#d1615d', '#85b6b2'];
  const groupColor = {};
  groupNames.forEach((g, i) => { groupColor[g] = groupPalette[i % groupPalette.length]; });

  const svg = d3.select(el).append('svg')
    .attr('class', 'plot').attr('width', width).attr('height', height);

  const zoomG = svg.append('g').attr('class', 'zoom-container');
  const g = zoomG.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  const zoom = d3.zoom()
    .scaleExtent([0.5, 5])
    .on('zoom', (event) => { zoomG.attr('transform', event.transform); });
  svg.call(zoom);
  svg.style('cursor', 'grab');
  svg.on('mousedown.cursor', () => svg.style('cursor', 'grabbing'));
  svg.on('mouseup.cursor', () => svg.style('cursor', 'grab'));

  const trails = g.selectAll('line.trail').data(wordData).enter().append('line')
    .attr('class', 'trail')
    .attr('x1', d => xScale(d.origCoord[0])).attr('y1', d => yScale(d.origCoord[1]))
    .attr('x2', d => xScale(d.origCoord[0])).attr('y2', d => yScale(d.origCoord[1]))
    .attr('stroke', d => groupColor[d.group]).attr('stroke-width', 1.5).attr('opacity', 0);

  g.selectAll('circle.ghost').data(wordData).enter().append('circle')
    .attr('class', 'ghost')
    .attr('cx', d => xScale(d.origCoord[0])).attr('cy', d => yScale(d.origCoord[1]))
    .attr('r', 3).attr('fill', d => groupColor[d.group]).attr('opacity', 0);

  const dots = g.selectAll('circle.word').data(wordData).enter().append('circle')
    .attr('class', 'word')
    .attr('cx', d => xScale(d.origCoord[0])).attr('cy', d => yScale(d.origCoord[1]))
    .attr('r', 4).attr('fill', d => groupColor[d.group]);

  const labels = g.selectAll('text.word-label').data(wordData).enter().append('text')
    .attr('class', 'word-label')
    .attr('x', d => xScale(d.origCoord[0])).attr('y', d => yScale(d.origCoord[1]) - 8)
    .attr('text-anchor', 'middle').attr('font-size', '11px').attr('fill', '#333')
    .text(d => d.word);

  const controls = d3.select(el).append('div').attr('class', 'steer-controls')
    .style('margin-top', '8px').style('display', 'flex').style('gap', '8px').style('align-items', 'center');

  const playBtn = controls.append('button')
    .style('background', COLORS.point).style('color', 'white').style('border', 'none')
    .style('border-radius', '4px').style('padding', '5px 14px').style('font-size', '13px')
    .style('cursor', 'pointer').text('▶ Steer');

  const resetBtn = controls.append('button')
    .style('background', '#ddd').style('color', '#333').style('border', 'none')
    .style('border-radius', '4px').style('padding', '5px 14px').style('font-size', '13px')
    .style('cursor', 'pointer').text('Reset');

  const statusText = controls.append('span')
    .style('font-size', '12px').style('color', '#999').text('Original embeddings');

  let steered = false;

  playBtn.on('click', () => {
    if (steered) return;
    steered = true;
    statusText.text('Steering...');
    g.selectAll('circle.ghost').transition().duration(300).attr('opacity', 0.3);
    trails.transition().duration(1500).ease(d3.easeCubicInOut)
      .attr('x2', d => xScale(d.steeredCoord[0])).attr('y2', d => yScale(d.steeredCoord[1]))
      .attr('opacity', 0.4);
    dots.transition().duration(1500).ease(d3.easeCubicInOut)
      .attr('cx', d => xScale(d.steeredCoord[0])).attr('cy', d => yScale(d.steeredCoord[1]));
    labels.transition().duration(1500).ease(d3.easeCubicInOut)
      .attr('x', d => xScale(d.steeredCoord[0])).attr('y', d => yScale(d.steeredCoord[1]) - 8);
    setTimeout(() => statusText.text('Steered embeddings'), 1500);
  });

  resetBtn.on('click', () => {
    steered = false;
    statusText.text('Original embeddings');
    g.selectAll('circle.ghost').transition().duration(300).attr('opacity', 0);
    trails.transition().duration(800).ease(d3.easeCubicInOut)
      .attr('x2', d => xScale(d.origCoord[0])).attr('y2', d => yScale(d.origCoord[1]))
      .attr('opacity', 0);
    dots.transition().duration(800).ease(d3.easeCubicInOut)
      .attr('cx', d => xScale(d.origCoord[0])).attr('cy', d => yScale(d.origCoord[1]));
    labels.transition().duration(800).ease(d3.easeCubicInOut)
      .attr('x', d => xScale(d.origCoord[0])).attr('y', d => yScale(d.origCoord[1]) - 8);
  });
}

export { EmbeddingViz, computeAllMDS, render2D, render3D, render1D, renderHero3D, renderSteering2D };
