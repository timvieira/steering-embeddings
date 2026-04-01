/**
 * Embedding visualizations using D3 for all rendering (1D, 2D, projected 3D).
 * Three.js is only used for the hero visualization.
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
 * Project 3D coordinates to 2D via rotation matrix.
 * angle: rotation around Y axis (radians)
 * tilt: rotation around X axis (radians)
 */
function project3Dto2D(coords3D, angle = 0, tilt = 0.4) {
  const cosA = Math.cos(angle), sinA = Math.sin(angle);
  const cosT = Math.cos(tilt), sinT = Math.sin(tilt);
  return coords3D.map(([x, y, z]) => {
    // Rotate around Y axis
    const x1 = x * cosA + z * sinA;
    const z1 = -x * sinA + z * cosA;
    // Rotate around X axis (tilt)
    const y1 = y * cosT - z1 * sinT;
    return [x1, y1];
  });
}

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


/** Get responsive width from a container element, with fallback. */
function getResponsiveWidth(container, fallback = 620) {
  const el = typeof container === 'string' ? document.getElementById(container) : container;
  if (!el) return fallback;
  const w = el.clientWidth || el.getBoundingClientRect().width;
  return w > 100 ? Math.min(w, 900) : fallback;
}

/**
 * MDS Eigenvalue selector widget (SVG in the margin).
 */
function createEigenSelector(container, eigenvalues, activeDims, onChange) {
  const topEig = eigenvalues.slice(0, 3);
  const maxVal = Math.max(...topEig.map(v => Math.max(0, v)));
  const barH = 20, barW = 8, gap = 2;
  const svgW = topEig.length * (barW + gap), svgH = barH + 12;

  const row = d3.select(container).append('div')
    .style('display', 'inline-flex')
    .style('align-items', 'flex-end')
    .style('gap', '3px')
    .style('user-select', 'none')
    .style('background', 'rgba(255,255,255,0.85)')
    .style('padding', '2px 4px')
    .style('border-radius', '3px');

  const svg = row.append('svg')
    .attr('width', svgW).attr('height', svgH)
    .style('cursor', 'pointer');

  svg.append('title').text('Click to switch dimensions');

  const bars = svg.selectAll('rect.bar')
    .data(topEig)
    .enter().append('rect')
    .attr('class', 'bar')
    .attr('x', (d, i) => i * (barW + gap))
    .attr('y', d => barH - (maxVal > 0 ? Math.max(0, d) / maxVal * barH : 0))
    .attr('width', barW)
    .attr('height', d => maxVal > 0 ? Math.max(0, d) / maxVal * barH : 0)
    .attr('fill', (d, i) => i < activeDims ? '#aac4de' : '#e8e8e8')
    .attr('rx', 1);

  svg.selectAll('text.label')
    .data(topEig)
    .enter().append('text')
    .attr('x', (d, i) => i * (barW + gap) + barW / 2)
    .attr('y', barH + 10)
    .attr('text-anchor', 'middle')
    .attr('font-size', '8px')
    .attr('fill', '#ccc')
    .text((d, i) => `${i + 1}`);

  bars.on('click', function(event, d) {
    const idx = topEig.indexOf(d);
    const newDims = idx + 1;
    bars.attr('fill', (d, i) => i < newDims ? '#aac4de' : '#e8e8e8');
    onChange(newDims);
  });

  const varianceText = row.append('span')
    .style('font-size', '9px')
    .style('color', '#ccc')
    .style('margin-left', '2px');

  return {
    update(dims, variance) {
      bars.attr('fill', (d, i) => i < dims ? '#aac4de' : '#e8e8e8');
      if (variance !== undefined) {
        varianceText.text(`${variance.toFixed(1)}%`);
      }
    }
  };
}


/**
 * 2D scatter plot with arrows using D3 SVG.
 */
function render2D(container, words, coords, arrows, options = {}) {
  const containerEl = typeof container === 'string' ? document.getElementById(container) : container;
  const defaultW = getResponsiveWidth(containerEl);
  const {
    highlights = [], crossGroupLines = [], width = defaultW, height = Math.round(defaultW * 0.55),
    neighborWords = new Set(), neighborLinks = [],
    animate = false, prevCoords = null, prevWords = null,
    fixedDomain = null,  // optional: [min, max] for both axes (for stable transitions)
    fixedDomainX = null, // optional: separate X domain (overrides fixedDomain)
    fixedDomainY = null, // optional: separate Y domain (overrides fixedDomain)
    disableZoom = false,  // true in 3D mode (orbit handles interaction)
    hiddenPoints = new Set(),  // indices of points to hide (no circle or label)
  } = options;
  // Detect 1D mode early: all y-coords are ~0 (set by _getCoords2D for dims===1)
  const is1D = coords.every(c => Math.abs(c[1]) < 1e-9);
  const margin = { top: is1D ? 40 : 0, right: 10, bottom: is1D ? 40 : 0, left: 10 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;
  const dur = animate ? 600 : 0;
  const highlightSet = new Set(highlights);

  // Scales — support separate X/Y domains for equal-aspect mode
  let xScale, yScale;
  const domX = fixedDomainX || fixedDomain;
  const domY = fixedDomainY || fixedDomain;
  if (domX || domY) {
    xScale = d3.scaleLinear().domain(domX || [-1.15, 1.15]).range([0, w]);
    yScale = d3.scaleLinear().domain(domY || [-1.15, 1.15]).range([h, 0]);
  } else {
    const xs = coords.map(c => c[0]);
    const ys = coords.map(c => c[1]);
    const pad = 0.1;
    const xRange = [Math.min(...xs), Math.max(...xs)];
    const yRange = [Math.min(...ys), Math.max(...ys)];
    const xPad = (xRange[1] - xRange[0]) * pad || 0.1;
    const yPad = (yRange[1] - yRange[0]) * pad || 0.1;
    xScale = d3.scaleLinear().domain([xRange[0] - xPad, xRange[1] + xPad]).range([0, w]);
    yScale = d3.scaleLinear().domain([yRange[0] - yPad, yRange[1] + yPad]).range([h, 0]);
  }

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

    // Pan + zoom via d3.zoom — but NOT in 3D mode (orbit handles drag there)
    if (!disableZoom) {
      const zoom = d3.zoom()
        .scaleExtent([0.5, 5])
        .filter((event) => {
          if (event.type === 'wheel') return true;
          if (event.type === 'mousedown' || event.type === 'pointerdown') {
            const tag = event.target.tagName;
            return tag !== 'circle' && tag !== 'text';
          }
          return true;
        })
        .on('zoom', (event) => { zoomG.attr('transform', event.transform); });
      svg.call(zoom);
    }
    svg.style('cursor', 'grab');
    svg.on('mousedown.cursor', () => svg.style('cursor', 'grabbing'));
    svg.on('mouseup.cursor', () => svg.style('cursor', 'grab'));
  } else {
    g = svg.select('g.main');
    // If switching to 3D on a reused SVG, remove the D3 zoom that was attached in 2D
    if (disableZoom) {
      svg.on('.zoom', null);  // remove all d3.zoom event listeners
      // Reset the zoom-container transform (pan/zoom may have shifted it)
      svg.select('g.zoom-container').attr('transform', null);
    }
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
    .attr('stroke-width', d => d.thick ? 2.5 : 1.5)
    .attr('stroke', d => d.color === 'red' ? COLORS.highlight : COLORS.arrow)
    .attr('stroke-dasharray', d => d.dashed ? '8,4' : null)
    .attr('opacity', d => d.dashed ? 0.8 : 1)
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
    .attr('r', d => hiddenPoints.has(d.i) ? 0 : highlightSet.has(d.i) ? 5 : 3.5)
    .attr('fill', d => highlightSet.has(d.i) ? COLORS.highlight :
      neighborWords.has(d.word) ? '#999' : COLORS.point);

  // --- Labels (keyed by word) ---
  // In 1D mode, alternate labels above/below the line (sorted by x) to reduce overlap.
  // Above: rotate -90° text-anchor:start (text up). Below: rotate 90° text-anchor:start (text down).
  // All use transform for smooth animated transitions between dimensions.
  const labelSide = {};  // word → 'above' | 'below'
  if (is1D) {
    const sorted = pointData.slice().sort((a, b) => a.c[0] - b.c[0]);
    sorted.forEach((d, i) => { labelSide[d.word] = i % 2 === 0 ? 'above' : 'below'; });
  }
  function labelTransformFor(d) {
    const lx = xScale(d.c[0]);
    if (!is1D) return `translate(${lx}, ${yScale(d.c[1]) - 8}) rotate(0)`;
    const below = labelSide[d.word] === 'below';
    const ly = yScale(d.c[1]) + (below ? 6 : -6);
    return `translate(${lx}, ${ly}) rotate(${below ? 90 : -90})`;
  }
  const labelAnchor = is1D ? 'start' : 'middle';

  const labels = g.selectAll('text.word-label').data(pointData, d => d.word);
  labels.exit().transition().duration(dur).style('opacity', 0).remove();

  // Migrate any existing labels from x/y positioning to transform (no visual change)
  if (animate) {
    labels.each(function() {
      const el = d3.select(this);
      const ox = parseFloat(el.attr('x')) || 0;
      const oy = parseFloat(el.attr('y')) || 0;
      if (ox !== 0 || oy !== 0) {
        el.attr('x', 0).attr('y', 0)
          .attr('transform', `translate(${ox}, ${oy}) rotate(0)`);
      }
    });
  }

  const labelsEnter = labels.enter().append('text').attr('class', 'word-label')
    .attr('text-anchor', labelAnchor)
    .style('opacity', 0)
    .attr('transform', d => {
      const [sx, sy] = startXY(d.i);
      return `translate(${sx}, ${sy - 8}) rotate(0)`;
    })
    .text(d => d.word);
  labelsEnter.merge(labels)
    .attr('text-anchor', labelAnchor)
    .transition().duration(dur)
    .attr('transform', d => labelTransformFor(d))
    .attr('font-size', d => hiddenPoints.has(d.i) ? '12px' : neighborWords.has(d.word) ? '10px' : '11px')
    .attr('font-weight', d => hiddenPoints.has(d.i) ? 'bold' : 'normal')
    .attr('font-style', d => hiddenPoints.has(d.i) ? 'italic' : 'normal')
    .attr('fill', d => hiddenPoints.has(d.i) ? COLORS.highlight : neighborWords.has(d.word) ? '#666' : '#333')
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
  const containerEl = typeof container === 'string' ? document.getElementById(container) : container;
  const defaultW = getResponsiveWidth(containerEl);
  const { highlights = [], crossGroupLines = [], width = defaultW, height = Math.round(defaultW * 0.72),
    neighborWords = new Set(), neighborLinks = [] } = options;

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
    const isNeighbor = neighborWords.has(words[i]);
    const pointColor = isHL ? COLORS.highlight : isNeighbor ? '#999' : COLORS.point;
    const geo = new THREE.SphereGeometry(isHL ? 0.06 : 0.04, 16, 16);
    const mat = new THREE.MeshBasicMaterial({ color: pointColor });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, y, z);
    mesh.userData = { index: i, word: words[i] };
    scene.add(mesh);
    clickableMeshes.push(mesh);

    // Label (sprite)
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 512; canvas.height = 128;
    ctx.font = isNeighbor ? '40px sans-serif' : 'bold 48px sans-serif';
    ctx.fillStyle = isNeighbor ? '#666' : '#333';
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

  // Neighbor links (dashed lines from parent to child)
  for (const { parent, child } of neighborLinks) {
    if (parent >= coords.length || child >= coords.length) continue;
    const points = [
      new THREE.Vector3(...coords[parent].map(v => v * scale)),
      new THREE.Vector3(...coords[child].map(v => v * scale)),
    ];
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineDashedMaterial({ color: '#ccc', dashSize: 0.04, gapSize: 0.02 });
    const line = new THREE.Line(geo, mat);
    line.computeLineDistances();
    scene.add(line);
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
  const containerEl = typeof container === 'string' ? document.getElementById(container) : container;
  const defaultW = getResponsiveWidth(containerEl);
  const { highlights = [], width = defaultW, height = 100, neighborWords = new Set(), neighborLinks = [] } = options;
  const margin = { top: 0, right: 10, bottom: 0, left: 10 };
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

  // Neighbor links (dashed lines from parent to child)
  const linkData = neighborLinks.filter(l => l.child < coords.length && l.parent < coords.length);
  g.selectAll('line.neighbor-link').data(linkData).enter().append('line')
    .attr('class', 'neighbor-link')
    .attr('x1', d => xScale(coords[d.parent][0])).attr('y1', 0)
    .attr('x2', d => xScale(coords[d.child][0])).attr('y2', 0)
    .attr('stroke', '#ccc').attr('stroke-width', 1).attr('stroke-dasharray', '3,2');

  const pointData = coords.map((c, i) => ({ c, i, word: words[i] }));

  g.selectAll('circle')
    .data(pointData)
    .enter().append('circle')
    .attr('cx', d => xScale(d.c[0]))
    .attr('cy', 0)
    .attr('r', d => highlightSet.has(d.i) ? 5 : 3)
    .attr('fill', d => highlightSet.has(d.i) ? COLORS.highlight :
      neighborWords.has(d.word) ? '#999' : COLORS.point)
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
    .attr('font-size', d => neighborWords.has(d.word) ? '10px' : '11px')
    .attr('fill', d => neighborWords.has(d.word) ? '#666' : '#333')
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
    this.hiddenPoints = config.hiddenPoints || new Set();
    this.connectGroups = config.connectGroups || false;
    this.dims = config.initialDims || 2;
    this.searchEmb = config.searchEmb || config.emb;  // full vocab for neighbor search
    this.equalAspect = config.equalAspect !== undefined ? config.equalAspect : true;
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

    this._rotationAngle = 0;
    this._tiltAngle = 0.4;  // initial tilt (same as project3Dto2D default)
    this._rotationAnim = null;

    // Create eigen selector (clear any previous one)
    if (this.eigenEl) {
      this.eigenEl.innerHTML = '';
      this.eigenSelector = createEigenSelector(
        this.eigenEl,
        this.mdsData.eigenvalues,
        this.dims,
        (newDims) => {
          this._prevCoords = this._currentCoords2D;
          this._prevWords = [...this.words];
          const prevDims = this.dims;
          this.dims = newDims;
          this._stopRotation();

          // When switching to 3D, find rotation angle that best matches current 2D layout
          if (newDims === 3 && prevDims <= 2 && this._prevCoords) {
            this._findBestRotation();
          }

          this.render(true);
        }
      );
    }

    this.render();
  }

  // Find rotation angle where projected 3D best matches current 2D layout
  _findBestRotation() {
    if (!this._prevCoords || !this.mdsData.coords[3]) return;
    const raw3D = this.mdsData.coords[3];
    const maxR = Math.max(...raw3D.map(([x, y, z]) => Math.sqrt(x*x + y*y + z*z))) || 1;
    const norm3D = raw3D.map(([x, y, z]) => [x / maxR, y / maxR, z / maxR]);
    const prev = this._prevCoords;
    const n = Math.min(prev.length, norm3D.length);

    // Normalize prev coords to [-1,1] for comparison
    const prevAll = prev.flatMap(c => [Math.abs(c[0]), Math.abs(c[1])]);
    const pm = Math.max(...prevAll) || 1;
    const prevN = prev.map(c => [c[0] / pm, c[1] / pm]);

    // Score: sum of squared distances after optimal uniform scaling.
    function score(a, t) {
      const p = project3Dto2D(norm3D, a, t);
      let dot = 0, pp = 0;
      for (let i = 0; i < n; i++) {
        dot += p[i][0] * prevN[i][0] + p[i][1] * prevN[i][1];
        pp += p[i][0] * p[i][0] + p[i][1] * p[i][1];
      }
      const s = pp > 0 ? dot / pp : 1;
      let d = 0;
      for (let i = 0; i < n; i++) {
        d += (s * p[i][0] - prevN[i][0]) ** 2 + (s * p[i][1] - prevN[i][1]) ** 2;
      }
      return d;
    }

    // Coarse search: 72 angles (5° steps) × full tilt range [-π/2, π/2]
    // 72 angles covers X-flip (angle + π). Wide tilt range covers Y-flip.
    let bestA = 0, bestT = 0.4, bestD = Infinity;
    for (let ai = 0; ai < 72; ai++) {
      const a = ai * Math.PI / 36;
      for (let ti = -7; ti <= 7; ti++) {
        const t = ti * (Math.PI / 14);  // ~±π/2
        const d = score(a, t);
        if (d < bestD) { bestD = d; bestA = a; bestT = t; }
      }
    }

    // Fine search: refine ±3° around best angle, ±0.05 around best tilt
    const fineStep = Math.PI / 180;
    for (let da = -3; da <= 3; da++) {
      for (let dt = -3; dt <= 3; dt++) {
        const a = bestA + da * fineStep;
        const t = bestT + dt * 0.02;
        const d = score(a, t);
        if (d < bestD) { bestD = d; bestA = a; bestT = t; }
      }
    }

    this._rotationAngle = bestA;
    this._tiltAngle = bestT;
  }

  // Get current 2D coordinates (projecting 3D if needed).
  // For 3D, normalizes to bounding sphere so scale is stable during rotation.
  _getCoords2D() {
    const raw = this.mdsData.coords[this.dims];
    if (this.dims === 1) {
      const maxAbs = Math.max(...raw.map(c => Math.abs(c[0]))) || 1;
      return raw.map(c => [c[0] / maxAbs, 0]);
    }
    if (this.dims === 2) {
      // Always normalize to bounding circle — keeps scale consistent with 3D for smooth transitions
      const maxR = Math.max(...raw.map(([x, y]) => Math.sqrt(x*x + y*y))) || 1;
      return raw.map(([x, y]) => [x / maxR, y / maxR]);
    }
    // 3D: project and normalize to bounding sphere
    const maxR = Math.max(...raw.map(([x, y, z]) => Math.sqrt(x*x + y*y + z*z))) || 1;
    const normalized = raw.map(([x, y, z]) => [x / maxR, y / maxR, z / maxR]);
    return project3Dto2D(normalized, this._rotationAngle, this._tiltAngle);
  }

  _stopRotation() {
    if (this._rotationAnim) {
      cancelAnimationFrame(this._rotationAnim);
      this._rotationAnim = null;
    }
  }

  _startRotation() {
    if (this.dims !== 3) return;
    const self = this;
    const el = typeof this.plotEl === 'string'
      ? document.getElementById(this.plotEl) : this.plotEl;
    const svg = d3.select(el).select('svg.plot');
    const g = svg.select('g.main');
    if (g.empty()) return;

    // Use same normalized coords and fixed domain as _getCoords2D and render()
    const raw3D = this.mdsData.coords[3];
    const maxR = Math.max(...raw3D.map(([x, y, z]) => Math.sqrt(x*x + y*y + z*z))) || 1;
    const normalized = raw3D.map(([x, y, z]) => [x / maxR, y / maxR, z / maxR]);

    const margin = { top: 0, right: 10, bottom: 0, left: 10 };
    const plotW = getResponsiveWidth(el);
    const plotH = Math.round(plotW * 0.55);
    const w = plotW - margin.left - margin.right;
    const h = plotH - margin.top - margin.bottom;
    // Same domain logic as render() — expand the wider axis for equal aspect
    const baseDom = [-1.08, 1.08];
    let domX = baseDom, domY = baseDom;
    if (this.equalAspect) {
      const baseSpan = baseDom[1] - baseDom[0];
      const mid = (baseDom[0] + baseDom[1]) / 2;
      if (w > h) {
        const s = baseSpan * (w / h);
        domX = [mid - s / 2, mid + s / 2];
      } else {
        const s = baseSpan * (h / w);
        domY = [mid - s / 2, mid + s / 2];
      }
    }
    const xScale = d3.scaleLinear().domain(domX).range([0, w]);
    const yScale = d3.scaleLinear().domain(domY).range([h, 0]);

    // Drag-to-orbit: horizontal drag controls rotation angle
    let dragging = false;
    let dragStartX = 0, dragStartY = 0;
    let dragStartAngle = 0, dragStartTilt = 0;
    let autoRotate = true;
    let resumeTimeout = null;

    const svgNode = svg.node();
    svgNode.addEventListener('pointerdown', (e) => {
      if (e.target.tagName === 'circle' || e.target.tagName === 'text') return;
      dragging = true;
      dragStartX = e.clientX;
      dragStartY = e.clientY;
      dragStartAngle = self._rotationAngle;
      dragStartTilt = self._tiltAngle;
      autoRotate = false;
      clearTimeout(resumeTimeout);
      svgNode.style.cursor = 'grabbing';
      svgNode.setPointerCapture(e.pointerId);
    });

    svgNode.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - dragStartX;
      const dy = e.clientY - dragStartY;
      self._rotationAngle = dragStartAngle + dx * 0.01;
      self._tiltAngle = Math.max(-Math.PI / 2, Math.min(Math.PI / 2,
        dragStartTilt + dy * 0.01));
    });

    svgNode.addEventListener('pointerup', () => {
      if (!dragging) return;
      dragging = false;
      svgNode.style.cursor = 'grab';
      resumeTimeout = setTimeout(() => { autoRotate = true; }, 3000);
    });

    function tick() {
      if (autoRotate && !dragging) self._rotationAngle += 0.005;
      const projected = project3Dto2D(normalized, self._rotationAngle, self._tiltAngle);
      self._currentCoords2D = projected;

      // Update positions directly (no transition — this is continuous rotation)
      g.selectAll('circle.point')
        .attr('cx', d => xScale(projected[d.i][0]))
        .attr('cy', d => yScale(projected[d.i][1]));
      g.selectAll('text.word-label')
        .attr('transform', d => `translate(${xScale(projected[d.i][0])}, ${yScale(projected[d.i][1]) - 8}) rotate(0)`);

      // Update arrows
      g.selectAll('line.arrow')
        .attr('x1', d => xScale(projected[d.from][0]))
        .attr('y1', d => yScale(projected[d.from][1]))
        .attr('x2', d => xScale(projected[d.to][0]))
        .attr('y2', d => yScale(projected[d.to][1]));

      // Update neighbor links
      g.selectAll('line.neighbor-link')
        .attr('x1', d => xScale(projected[d.parent][0]))
        .attr('y1', d => yScale(projected[d.parent][1]))
        .attr('x2', d => xScale(projected[d.child][0]))
        .attr('y2', d => yScale(projected[d.child][1]));

      // Update cross-group lines
      g.selectAll('line.cross-group')
        .attr('x1', d => xScale(projected[d[0]][0]))
        .attr('y1', d => yScale(projected[d[0]][1]))
        .attr('x2', d => xScale(projected[d[1]][0]))
        .attr('y2', d => yScale(projected[d[1]][1]));

      self._rotationAnim = requestAnimationFrame(tick);
    }
    self._rotationAnim = requestAnimationFrame(tick);
  }

  _makeOnClick() {
    const self = this;
    return (idx, word) => {
      const vec = self.searchEmb.vec(word);
      if (!vec) return;
      const existing = new Set(self.words);
      const neighbors = self.searchEmb.mostSimilar(vec, 5, existing);
      if (neighbors.length === 0) return;

      self._stopRotation();
      self._prevCoords = self._currentCoords2D;
      self._prevWords = [...self.words];

      const wordIdx = new Map(self.words.map((w, i) => [w, i]));
      const parentIdx = wordIdx.get(word);
      for (const nw of neighbors) {
        if (!existing.has(nw) && self.searchEmb.has(nw)) {
          // Add neighbor's vector to self.emb if it's not already there
          if (!self.emb.has(nw)) {
            self.emb.addWord(nw, self.searchEmb);
          }
          self.words.push(nw);
          self.neighborWords.add(nw);
          self.neighborLinks.push({ parent: parentIdx, child: self.words.length - 1 });
          existing.add(nw);
        }
      }

      self.mdsData = computeAllMDS(self.emb, self.words);
      self.render(true);
    };
  }

  render(animate = false) {
    this._stopRotation();
    const coords2D = this._getCoords2D();
    this._currentCoords2D = coords2D;

    const el = typeof this.plotEl === 'string'
      ? document.getElementById(this.plotEl) : this.plotEl;

    // For animated updates, keep the SVG; otherwise rebuild
    if (!animate) {
      el.innerHTML = '';
    }

    // Fixed domain for all dims — coords are normalized to [-1,1],
    // so a consistent domain prevents zoom jumps during dimension transitions.
    const baseDomain = [-1.08, 1.08];

    // In 1D mode, use a compact height — the data is a single line
    const plotWidth = getResponsiveWidth(el);
    const plotHeight = this.dims === 1
      ? Math.min(160, Math.round(plotWidth * 0.25))
      : Math.round(plotWidth * 0.55);

    // Equal aspect: expand the domain on the longer axis so 1 data-unit = same pixels on both axes
    let fixedDomainX = baseDomain, fixedDomainY = baseDomain;
    const margin = { top: 30, right: 30, bottom: 30, left: 30 };
    const innerW = plotWidth - margin.left - margin.right;
    const innerH = plotHeight - margin.top - margin.bottom;
    if (this.equalAspect && this.dims >= 2) {
      const baseSpan = baseDomain[1] - baseDomain[0];  // 2.3
      if (innerW > innerH) {
        const expandedSpan = baseSpan * (innerW / innerH);
        const mid = (baseDomain[0] + baseDomain[1]) / 2;
        fixedDomainX = [mid - expandedSpan / 2, mid + expandedSpan / 2];
      } else {
        const expandedSpan = baseSpan * (innerH / innerW);
        const mid = (baseDomain[0] + baseDomain[1]) / 2;
        fixedDomainY = [mid - expandedSpan / 2, mid + expandedSpan / 2];
      }
    }

    const opts = {
      highlights: this.highlights,
      crossGroupLines: this.crossGroupLines,
      neighborWords: this.neighborWords,
      neighborLinks: this.neighborLinks,
      animate,
      prevCoords: this._prevCoords,
      prevWords: this._prevWords,
      onClick: this._makeOnClick(),
      fixedDomainX, fixedDomainY,
      disableZoom: this.dims === 3,
      hiddenPoints: this.hiddenPoints,
      width: plotWidth,
      height: plotHeight,
    };

    render2D(el, this.words, coords2D, this.arrows, opts);

    if (this.eigenSelector) this.eigenSelector.update(this.dims, this.mdsData.variance[this.dims]);

    // Start rotation for 3D
    if (this.dims === 3) {
      // Wait for transition to finish before starting rotation
      setTimeout(() => this._startRotation(), animate ? 700 : 0);
    }
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

  camera.position.set(2, 1.3, 2);
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
 * Compute MDS at dimensions 1, 2, 3 from a raw distance matrix.
 * Returns the same shape as computeAllMDS: { coords, eigenvalues, variance }.
 */
function computeAllMDSFromMatrix(distMatrix, n) {
  const r1 = mds(distMatrix, n, 1);
  const r2 = mds(distMatrix, n, 2);
  const r3 = mds(distMatrix, n, 3);
  const allEig = r3.eigenvalues;
  const eigSum = allEig.reduce((a, b) => a + Math.max(0, b), 0);
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
 * SteeringViz: dimension-switchable steering visualization with eigenvalue bars.
 *
 * Takes a pre-built joint distance matrix (over [steered_0..n, orig_0..n])
 * and wordData array, computes MDS at 1/2/3 dims, and renders with
 * Steer/Reset buttons, trails, ghost dots, arrows, and dimension switching.
 *
 * config: {
 *   plotEl,        // DOM element or ID for the SVG plot
 *   eigenEl,       // DOM element or ID for the eigenvalue selector (optional)
 *   wordData,      // array of { word, group } — one entry per unique word
 *   distMatrix,    // Float64Array, joint distance matrix of size (2n)x(2n)
 *   n,             // number of unique words (half of distMatrix dimension)
 *   arrows,        // optional array of { from, to } index pairs
 *   initialDims,   // 1, 2, or 3 (default 2)
 * }
 */
class SteeringViz {
  constructor(config) {
    this.plotEl = typeof config.plotEl === 'string'
      ? document.getElementById(config.plotEl) : config.plotEl;
    this.eigenEl = config.eigenEl
      ? (typeof config.eigenEl === 'string' ? document.getElementById(config.eigenEl) : config.eigenEl)
      : null;
    this.wordData = config.wordData;  // { word, group }[]
    this.arrows = config.arrows || [];
    this.n = config.n;                // number of words
    this.dims = config.initialDims || 2;
    this.steered = false;

    // Compute MDS at all dimensions from the joint distance matrix
    const nn = 2 * this.n;
    this.mdsData = computeAllMDSFromMatrix(config.distMatrix, nn);

    // Assign coords to wordData at each dimension level
    this._assignCoords();

    // Rotation state for 3D
    this._rotationAngle = 0;
    this._tiltAngle = 0.4;
    this._rotationAnim = null;
    this._prevCoords = null;

    // Create eigen selector
    if (this.eigenEl) {
      this.eigenEl.innerHTML = '';
      this.eigenSelector = createEigenSelector(
        this.eigenEl,
        this.mdsData.eigenvalues,
        this.dims,
        (newDims) => {
          this._prevCoords = this._currentProjected;
          const prevDims = this.dims;
          this.dims = newDims;
          this._stopRotation();
          if (newDims === 3 && prevDims <= 2 && this._prevCoords) {
            this._findBestRotation();
          }
          this.render(true);
        }
      );
    }

    this.render();
  }

  _assignCoords() {
    const n = this.n;
    for (let dim = 1; dim <= 3; dim++) {
      const coords = this.mdsData.coords[dim];
      for (let i = 0; i < n; i++) {
        if (!this.wordData[i]._coords) this.wordData[i]._coords = {};
        this.wordData[i]._coords[dim] = {
          steered: coords[i],
          orig: coords[n + i],
        };
      }
    }
  }

  /** Get projected 2D coords for current dim, for both orig and steered sets.
   *  Returns { orig: [[x,y],...], steered: [[x,y],...] }
   */
  _getProjected() {
    const n = this.n;
    const dim = this.dims;
    const origRaw = this.wordData.map(d => d._coords[dim].orig);
    const steeredRaw = this.wordData.map(d => d._coords[dim].steered);

    if (dim === 1) {
      const all = [...origRaw, ...steeredRaw].map(c => Math.abs(c[0]));
      const maxAbs = Math.max(...all) || 1;
      return {
        orig: origRaw.map(c => [c[0] / maxAbs, 0]),
        steered: steeredRaw.map(c => [c[0] / maxAbs, 0]),
      };
    }
    if (dim === 2) {
      const all = [...origRaw, ...steeredRaw];
      const maxR = Math.max(...all.map(([x, y]) => Math.sqrt(x*x + y*y))) || 1;
      return {
        orig: origRaw.map(([x, y]) => [x / maxR, y / maxR]),
        steered: steeredRaw.map(([x, y]) => [x / maxR, y / maxR]),
      };
    }
    // 3D: normalize to bounding sphere then project
    const all = [...origRaw, ...steeredRaw];
    const maxR = Math.max(...all.map(([x, y, z]) => Math.sqrt(x*x + y*y + z*z))) || 1;
    const origNorm = origRaw.map(([x, y, z]) => [x / maxR, y / maxR, z / maxR]);
    const steeredNorm = steeredRaw.map(([x, y, z]) => [x / maxR, y / maxR, z / maxR]);
    return {
      orig: project3Dto2D(origNorm, this._rotationAngle, this._tiltAngle),
      steered: project3Dto2D(steeredNorm, this._rotationAngle, this._tiltAngle),
    };
  }

  _findBestRotation() {
    if (!this._prevCoords || !this.mdsData.coords[3]) return;
    // Use all points (orig + steered) in the current steered state for matching
    const raw3D = this.mdsData.coords[3];
    const n = this.n;
    // Use the set of points currently visible (steered or orig depending on state)
    const pointsRaw = this.steered
      ? raw3D.slice(0, n)   // steered positions
      : raw3D.slice(n);     // orig positions
    const maxR = Math.max(...[...raw3D.slice(0, n), ...raw3D.slice(n)]
      .map(([x, y, z]) => Math.sqrt(x*x + y*y + z*z))) || 1;
    const norm3D = pointsRaw.map(([x, y, z]) => [x / maxR, y / maxR, z / maxR]);
    const prev = this._prevCoords;
    const nn = Math.min(prev.length, norm3D.length);

    const prevAll = prev.flatMap(c => [Math.abs(c[0]), Math.abs(c[1])]);
    const pm = Math.max(...prevAll) || 1;
    const prevN = prev.map(c => [c[0] / pm, c[1] / pm]);

    function score(a, t) {
      const p = project3Dto2D(norm3D, a, t);
      let dot = 0, pp = 0;
      for (let i = 0; i < nn; i++) {
        dot += p[i][0] * prevN[i][0] + p[i][1] * prevN[i][1];
        pp += p[i][0] * p[i][0] + p[i][1] * p[i][1];
      }
      const s = pp > 0 ? dot / pp : 1;
      let d = 0;
      for (let i = 0; i < nn; i++) {
        d += (s * p[i][0] - prevN[i][0]) ** 2 + (s * p[i][1] - prevN[i][1]) ** 2;
      }
      return d;
    }

    let bestA = 0, bestT = 0.4, bestD = Infinity;
    for (let ai = 0; ai < 72; ai++) {
      const a = ai * Math.PI / 36;
      for (let ti = -7; ti <= 7; ti++) {
        const t = ti * (Math.PI / 14);
        const d = score(a, t);
        if (d < bestD) { bestD = d; bestA = a; bestT = t; }
      }
    }
    const fineStep = Math.PI / 180;
    for (let da = -3; da <= 3; da++) {
      for (let dt = -3; dt <= 3; dt++) {
        const a = bestA + da * fineStep;
        const t = bestT + dt * 0.02;
        const d = score(a, t);
        if (d < bestD) { bestD = d; bestA = a; bestT = t; }
      }
    }
    this._rotationAngle = bestA;
    this._tiltAngle = bestT;
  }

  _stopRotation() {
    if (this._rotationAnim) {
      cancelAnimationFrame(this._rotationAnim);
      this._rotationAnim = null;
    }
  }

  _startRotation() {
    if (this.dims !== 3) return;
    const self = this;
    const el = this.plotEl;
    const svg = d3.select(el).select('svg.plot');
    const g = svg.select('g.main');
    if (g.empty()) return;

    const raw3D = this.mdsData.coords[3];
    const n = this.n;
    const allRaw = [...raw3D.slice(0, n), ...raw3D.slice(n)];
    const maxR = Math.max(...allRaw.map(([x, y, z]) => Math.sqrt(x*x + y*y + z*z))) || 1;
    const origNorm = raw3D.slice(n, 2*n).map(([x, y, z]) => [x / maxR, y / maxR, z / maxR]);
    const steeredNorm = raw3D.slice(0, n).map(([x, y, z]) => [x / maxR, y / maxR, z / maxR]);

    const margin = { top: 0, right: 10, bottom: 0, left: 10 };
    const plotW = getResponsiveWidth(el);
    const plotH = Math.round(plotW * 0.55);
    const w = plotW - margin.left - margin.right;
    const h = plotH - margin.top - margin.bottom;
    const baseDom = [-1.08, 1.08];
    let domX = baseDom, domY = baseDom;
    const baseSpan = baseDom[1] - baseDom[0];
    const mid = (baseDom[0] + baseDom[1]) / 2;
    if (w > h) {
      const s = baseSpan * (w / h);
      domX = [mid - s / 2, mid + s / 2];
    } else {
      const s = baseSpan * (h / w);
      domY = [mid - s / 2, mid + s / 2];
    }
    const xScale = d3.scaleLinear().domain(domX).range([0, w]);
    const yScale = d3.scaleLinear().domain(domY).range([h, 0]);

    let dragging = false;
    let dragStartX = 0, dragStartY = 0;
    let dragStartAngle = 0, dragStartTilt = 0;
    let autoRotate = true;
    let resumeTimeout = null;

    const svgNode = svg.node();
    svgNode.addEventListener('pointerdown', (e) => {
      if (e.target.tagName === 'circle' || e.target.tagName === 'text') return;
      dragging = true;
      dragStartX = e.clientX;
      dragStartY = e.clientY;
      dragStartAngle = self._rotationAngle;
      dragStartTilt = self._tiltAngle;
      autoRotate = false;
      clearTimeout(resumeTimeout);
      svgNode.style.cursor = 'grabbing';
      svgNode.setPointerCapture(e.pointerId);
    });
    svgNode.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - dragStartX;
      const dy = e.clientY - dragStartY;
      self._rotationAngle = dragStartAngle + dx * 0.01;
      self._tiltAngle = Math.max(-Math.PI / 2, Math.min(Math.PI / 2,
        dragStartTilt + dy * 0.01));
    });
    svgNode.addEventListener('pointerup', () => {
      if (!dragging) return;
      dragging = false;
      svgNode.style.cursor = 'grab';
      resumeTimeout = setTimeout(() => { autoRotate = true; }, 3000);
    });

    const wordData = this.wordData;
    const arrows = this.arrows;
    const steered = this.steered;

    function tick() {
      if (autoRotate && !dragging) self._rotationAngle += 0.005;
      const projOrig = project3Dto2D(origNorm, self._rotationAngle, self._tiltAngle);
      const projSteered = project3Dto2D(steeredNorm, self._rotationAngle, self._tiltAngle);
      const activeProj = steered ? projSteered : projOrig;
      self._currentProjected = activeProj;

      g.selectAll('circle.word')
        .attr('cx', (d, i) => xScale(activeProj[i][0]))
        .attr('cy', (d, i) => yScale(activeProj[i][1]));
      g.selectAll('text.word-label')
        .attr('x', (d, i) => xScale(activeProj[i][0]))
        .attr('y', (d, i) => yScale(activeProj[i][1]) - 8);
      g.selectAll('circle.ghost')
        .attr('cx', (d, i) => xScale(projOrig[i][0]))
        .attr('cy', (d, i) => yScale(projOrig[i][1]));
      g.selectAll('line.trail')
        .attr('x1', (d, i) => xScale(projOrig[i][0]))
        .attr('y1', (d, i) => yScale(projOrig[i][1]))
        .attr('x2', (d, i) => xScale(steered ? projSteered[i][0] : projOrig[i][0]))
        .attr('y2', (d, i) => yScale(steered ? projSteered[i][1] : projOrig[i][1]));
      g.selectAll('line.ghost-arrow')
        .attr('x1', d => xScale(projOrig[d.from][0]))
        .attr('y1', d => yScale(projOrig[d.from][1]))
        .attr('x2', d => xScale(projOrig[d.to][0]))
        .attr('y2', d => yScale(projOrig[d.to][1]));
      g.selectAll('line.active-arrow')
        .attr('x1', d => xScale(activeProj[d.from][0]))
        .attr('y1', d => yScale(activeProj[d.from][1]))
        .attr('x2', d => xScale(activeProj[d.to][0]))
        .attr('y2', d => yScale(activeProj[d.to][1]));

      self._rotationAnim = requestAnimationFrame(tick);
    }
    self._rotationAnim = requestAnimationFrame(tick);
  }

  render(animate = false) {
    this._stopRotation();
    const el = this.plotEl;
    const projected = this._getProjected();
    // Current visible coords: if steered, use steered; else orig
    const activeCoords = this.steered ? projected.steered : projected.orig;
    this._currentProjected = activeCoords;

    const defaultW = getResponsiveWidth(el);
    const plotWidth = defaultW;
    const plotHeight = this.dims === 1
      ? Math.min(200, Math.round(plotWidth * 0.35))
      : Math.round(plotWidth * 0.55);

    const margin = { top: 0, right: 10, bottom: 0, left: 10 };
    const w = plotWidth - margin.left - margin.right;
    const h = plotHeight - margin.top - margin.bottom;

    // Fixed domain for stable dimension transitions
    const baseDomain = [-1.08, 1.08];
    let fixedDomainX = baseDomain, fixedDomainY = baseDomain;
    if (this.dims >= 2) {
      const baseSpan = baseDomain[1] - baseDomain[0];
      const mid = (baseDomain[0] + baseDomain[1]) / 2;
      if (w > h) {
        const expandedSpan = baseSpan * (w / h);
        fixedDomainX = [mid - expandedSpan / 2, mid + expandedSpan / 2];
      } else {
        const expandedSpan = baseSpan * (h / w);
        fixedDomainY = [mid - expandedSpan / 2, mid + expandedSpan / 2];
      }
    }

    const xScale = d3.scaleLinear().domain(fixedDomainX).range([0, w]);
    const yScale = d3.scaleLinear().domain(fixedDomainY).range([h, 0]);

    const groupNames = [...new Set(this.wordData.map(d => d.group))];
    const groupPalette = ['#5778a4', '#e49444', '#6a9f58', '#b07aa1', '#d1615d', '#85b6b2'];
    const groupColor = {};
    groupNames.forEach((g, i) => { groupColor[g] = groupPalette[i % groupPalette.length]; });

    const dur = animate ? 600 : 0;
    const wordData = this.wordData;
    const arrows = this.arrows;
    const n = this.n;

    // Rebuild SVG
    d3.select(el).selectAll('svg.plot').remove();
    d3.select(el).selectAll('.steer-controls').remove();

    const svg = d3.select(el).append('svg')
      .attr('class', 'plot').attr('width', plotWidth).attr('height', plotHeight);

    const zoomG = svg.append('g').attr('class', 'zoom-container');
    const g = zoomG.append('g')
      .attr('class', 'main')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Pan+zoom only in 1D/2D
    if (this.dims < 3) {
      const zoom = d3.zoom()
        .scaleExtent([0.5, 5])
        .on('zoom', (event) => { zoomG.attr('transform', event.transform); });
      svg.call(zoom);
    }
    svg.style('cursor', 'grab');
    svg.on('mousedown.cursor', () => svg.style('cursor', 'grabbing'));
    svg.on('mouseup.cursor', () => svg.style('cursor', 'grab'));

    // Trails
    const trails = g.selectAll('line.trail').data(wordData).enter().append('line')
      .attr('class', 'trail')
      .attr('x1', (d, i) => xScale(projected.orig[i][0]))
      .attr('y1', (d, i) => yScale(projected.orig[i][1]))
      .attr('x2', (d, i) => xScale(this.steered ? projected.steered[i][0] : projected.orig[i][0]))
      .attr('y2', (d, i) => yScale(this.steered ? projected.steered[i][1] : projected.orig[i][1]))
      .attr('stroke', d => groupColor[d.group]).attr('stroke-width', 1.5)
      .attr('opacity', this.steered ? 0.4 : 0);

    // Ghost dots
    g.selectAll('circle.ghost').data(wordData).enter().append('circle')
      .attr('class', 'ghost')
      .attr('cx', (d, i) => xScale(projected.orig[i][0]))
      .attr('cy', (d, i) => yScale(projected.orig[i][1]))
      .attr('r', 3).attr('fill', d => groupColor[d.group])
      .attr('opacity', this.steered ? 0.3 : 0);

    // Ghost arrows
    const ghostArrows = g.selectAll('line.ghost-arrow').data(arrows).enter().append('line')
      .attr('class', 'ghost-arrow')
      .attr('x1', d => xScale(projected.orig[d.from][0]))
      .attr('y1', d => yScale(projected.orig[d.from][1]))
      .attr('x2', d => xScale(projected.orig[d.to][0]))
      .attr('y2', d => yScale(projected.orig[d.to][1]))
      .attr('stroke', '#ccc').attr('stroke-width', 1).attr('stroke-dasharray', '4,3')
      .attr('opacity', this.steered ? 0.3 : 0);

    // Active arrows
    const activeArrows = g.selectAll('line.active-arrow').data(arrows).enter().append('line')
      .attr('class', 'active-arrow')
      .attr('x1', d => xScale(activeCoords[d.from][0]))
      .attr('y1', d => yScale(activeCoords[d.from][1]))
      .attr('x2', d => xScale(activeCoords[d.to][0]))
      .attr('y2', d => yScale(activeCoords[d.to][1]))
      .attr('stroke', COLORS.arrow).attr('stroke-width', 1.5);

    // Word dots
    const dots = g.selectAll('circle.word').data(wordData).enter().append('circle')
      .attr('class', 'word')
      .attr('cx', (d, i) => xScale(activeCoords[i][0]))
      .attr('cy', (d, i) => yScale(activeCoords[i][1]))
      .attr('r', 4).attr('fill', d => groupColor[d.group]);

    // Labels
    const labels = g.selectAll('text.word-label').data(wordData).enter().append('text')
      .attr('class', 'word-label')
      .attr('x', (d, i) => xScale(activeCoords[i][0]))
      .attr('y', (d, i) => yScale(activeCoords[i][1]) - 8)
      .attr('text-anchor', 'middle').attr('font-size', '11px').attr('fill', '#333')
      .text(d => d.word);

    // Steer/Reset controls
    const controls = d3.select(el).append('div').attr('class', 'steer-controls')
      .style('margin-top', '8px').style('display', 'flex').style('gap', '8px').style('align-items', 'center');

    const toggleBtn = controls.append('button')
      .style('background', this.steered ? '#ddd' : COLORS.point)
      .style('color', this.steered ? '#333' : 'white')
      .style('border', 'none')
      .style('border-radius', '4px').style('padding', '5px 14px').style('font-size', '13px')
      .style('cursor', 'pointer').text(this.steered ? 'Reset' : '▶ Steer');

    const statusText = controls.append('span')
      .style('font-size', '12px').style('color', '#999')
      .text(this.steered ? 'Steered embeddings' : 'Original embeddings');

    const self = this;
    toggleBtn.on('click', () => {
      if (!self.steered) {
        self.steered = true;
        toggleBtn.text('Reset').style('background', '#ddd').style('color', '#333');
        statusText.text('Steering...');

        if (self.dims === 3) {
          // In 3D, the rotation loop handles position updates; just toggle state
          g.selectAll('circle.ghost').transition().duration(300).attr('opacity', 0.3);
          ghostArrows.transition().duration(300).attr('opacity', 0.3);
          // trails and dots will update on next rotation tick
          setTimeout(() => statusText.text('Steered embeddings'), 300);
        } else {
          const proj = self._getProjected();
          g.selectAll('circle.ghost').transition().duration(300).attr('opacity', 0.3);
          ghostArrows.transition().duration(300).attr('opacity', 0.3);
          trails.transition().duration(1500).ease(d3.easeCubicInOut)
            .attr('x2', (d, i) => xScale(proj.steered[i][0]))
            .attr('y2', (d, i) => yScale(proj.steered[i][1]))
            .attr('opacity', 0.4);
          dots.transition().duration(1500).ease(d3.easeCubicInOut)
            .attr('cx', (d, i) => xScale(proj.steered[i][0]))
            .attr('cy', (d, i) => yScale(proj.steered[i][1]));
          labels.transition().duration(1500).ease(d3.easeCubicInOut)
            .attr('x', (d, i) => xScale(proj.steered[i][0]))
            .attr('y', (d, i) => yScale(proj.steered[i][1]) - 8);
          activeArrows.transition().duration(1500).ease(d3.easeCubicInOut)
            .attr('x1', d => xScale(proj.steered[d.from][0]))
            .attr('y1', d => yScale(proj.steered[d.from][1]))
            .attr('x2', d => xScale(proj.steered[d.to][0]))
            .attr('y2', d => yScale(proj.steered[d.to][1]));
          setTimeout(() => statusText.text('Steered embeddings'), 1500);
        }
      } else {
        self.steered = false;
        toggleBtn.text('▶ Steer').style('background', COLORS.point).style('color', 'white');
        statusText.text('Original embeddings');

        if (self.dims === 3) {
          g.selectAll('circle.ghost').transition().duration(300).attr('opacity', 0);
          ghostArrows.transition().duration(300).attr('opacity', 0);
        } else {
          const proj = self._getProjected();
          g.selectAll('circle.ghost').transition().duration(300).attr('opacity', 0);
          ghostArrows.transition().duration(300).attr('opacity', 0);
          trails.transition().duration(800).ease(d3.easeCubicInOut)
            .attr('x2', (d, i) => xScale(proj.orig[i][0]))
            .attr('y2', (d, i) => yScale(proj.orig[i][1]))
            .attr('opacity', 0);
          dots.transition().duration(800).ease(d3.easeCubicInOut)
            .attr('cx', (d, i) => xScale(proj.orig[i][0]))
            .attr('cy', (d, i) => yScale(proj.orig[i][1]));
          labels.transition().duration(800).ease(d3.easeCubicInOut)
            .attr('x', (d, i) => xScale(proj.orig[i][0]))
            .attr('y', (d, i) => yScale(proj.orig[i][1]) - 8);
          activeArrows.transition().duration(800).ease(d3.easeCubicInOut)
            .attr('x1', d => xScale(proj.orig[d.from][0]))
            .attr('y1', d => yScale(proj.orig[d.from][1]))
            .attr('x2', d => xScale(proj.orig[d.to][0]))
            .attr('y2', d => yScale(proj.orig[d.to][1]));
        }
      }
    });

    // Update eigen selector
    if (this.eigenSelector) {
      this.eigenSelector.update(this.dims, this.mdsData.variance[this.dims]);
    }

    // Start 3D rotation
    if (this.dims === 3) {
      setTimeout(() => this._startRotation(), animate ? 700 : 0);
    }
  }
}


/**
 * 2D animated steering visualization (legacy stateless version).
 * Shows words moving from original to steered positions with fading trails.
 */
function renderSteering2D(container, wordData, options = {}) {
  const el = typeof container === 'string' ? document.getElementById(container) : container;
  const defaultW = getResponsiveWidth(el);
  const { width = defaultW, height = Math.round(defaultW * 0.55), arrows = [] } = options;
  const margin = { top: 0, right: 10, bottom: 0, left: 10 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  d3.select(el).selectAll('svg.plot').remove();
  d3.select(el).selectAll('.steer-controls').remove();

  const allX = wordData.flatMap(d => [d.origCoord[0], d.steeredCoord[0]]);
  const allY = wordData.flatMap(d => [d.origCoord[1], d.steeredCoord[1]]);
  const pad = 0.1;
  const xMin = Math.min(...allX), xMax = Math.max(...allX);
  const yMin = Math.min(...allY), yMax = Math.max(...allY);
  const xSpan = (xMax - xMin) || 0.2, ySpan = (yMax - yMin) || 0.2;
  const xPad = xSpan * pad, yPad = ySpan * pad;
  // Equal scaling: expand the smaller axis to match the aspect ratio
  const dataW = xSpan + 2 * xPad, dataH = ySpan + 2 * yPad;
  const aspect = w / h;
  let domW = dataW, domH = dataH;
  if (dataW / dataH > aspect) {
    domH = domW / aspect;
  } else {
    domW = domH * aspect;
  }
  const cxDom = (xMin + xMax) / 2, cyDom = (yMin + yMax) / 2;
  const xScale = d3.scaleLinear().domain([cxDom - domW / 2, cxDom + domW / 2]).range([0, w]);
  const yScale = d3.scaleLinear().domain([cyDom - domH / 2, cyDom + domH / 2]).range([h, 0]);

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

  // Ghost arrows (original positions, shown after steer)
  const wordIdx = new Map(wordData.map((d, i) => [d.word, i]));
  const ghostArrows = g.selectAll('line.ghost-arrow').data(arrows).enter().append('line')
    .attr('class', 'ghost-arrow')
    .attr('x1', d => xScale(wordData[d.from]?.origCoord[0])).attr('y1', d => yScale(wordData[d.from]?.origCoord[1]))
    .attr('x2', d => xScale(wordData[d.to]?.origCoord[0])).attr('y2', d => yScale(wordData[d.to]?.origCoord[1]))
    .attr('stroke', '#ccc').attr('stroke-width', 1).attr('stroke-dasharray', '4,3').attr('opacity', 0);

  // Active arrows (start at original, animate to steered)
  const activeArrows = g.selectAll('line.active-arrow').data(arrows).enter().append('line')
    .attr('class', 'active-arrow')
    .attr('x1', d => xScale(wordData[d.from]?.origCoord[0])).attr('y1', d => yScale(wordData[d.from]?.origCoord[1]))
    .attr('x2', d => xScale(wordData[d.to]?.origCoord[0])).attr('y2', d => yScale(wordData[d.to]?.origCoord[1]))
    .attr('stroke', COLORS.arrow).attr('stroke-width', 1.5);

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

  const toggleBtn = controls.append('button')
    .style('background', COLORS.point).style('color', 'white').style('border', 'none')
    .style('border-radius', '4px').style('padding', '5px 14px').style('font-size', '13px')
    .style('cursor', 'pointer').text('▶ Steer');

  const statusText = controls.append('span')
    .style('font-size', '12px').style('color', '#999').text('Original embeddings');

  let steered = false;

  toggleBtn.on('click', () => {
    if (!steered) {
      steered = true;
      toggleBtn.text('Reset').style('background', '#ddd').style('color', '#333');
      statusText.text('Steering...');
      g.selectAll('circle.ghost').transition().duration(300).attr('opacity', 0.3);
      ghostArrows.transition().duration(300).attr('opacity', 0.3);
      trails.transition().duration(1500).ease(d3.easeCubicInOut)
        .attr('x2', d => xScale(d.steeredCoord[0])).attr('y2', d => yScale(d.steeredCoord[1]))
        .attr('opacity', 0.4);
      dots.transition().duration(1500).ease(d3.easeCubicInOut)
        .attr('cx', d => xScale(d.steeredCoord[0])).attr('cy', d => yScale(d.steeredCoord[1]));
      labels.transition().duration(1500).ease(d3.easeCubicInOut)
        .attr('x', d => xScale(d.steeredCoord[0])).attr('y', d => yScale(d.steeredCoord[1]) - 8);
      activeArrows.transition().duration(1500).ease(d3.easeCubicInOut)
        .attr('x1', d => xScale(wordData[d.from]?.steeredCoord[0])).attr('y1', d => yScale(wordData[d.from]?.steeredCoord[1]))
        .attr('x2', d => xScale(wordData[d.to]?.steeredCoord[0])).attr('y2', d => yScale(wordData[d.to]?.steeredCoord[1]));
      setTimeout(() => statusText.text('Steered embeddings'), 1500);
    } else {
      steered = false;
      toggleBtn.text('▶ Steer').style('background', COLORS.point).style('color', 'white');
      statusText.text('Original embeddings');
      g.selectAll('circle.ghost').transition().duration(300).attr('opacity', 0);
      ghostArrows.transition().duration(300).attr('opacity', 0);
      trails.transition().duration(800).ease(d3.easeCubicInOut)
        .attr('x2', d => xScale(d.origCoord[0])).attr('y2', d => yScale(d.origCoord[1]))
        .attr('opacity', 0);
      dots.transition().duration(800).ease(d3.easeCubicInOut)
        .attr('cx', d => xScale(d.origCoord[0])).attr('cy', d => yScale(d.origCoord[1]));
      labels.transition().duration(800).ease(d3.easeCubicInOut)
        .attr('x', d => xScale(d.origCoord[0])).attr('y', d => yScale(d.origCoord[1]) - 8);
      activeArrows.transition().duration(800).ease(d3.easeCubicInOut)
        .attr('x1', d => xScale(wordData[d.from]?.origCoord[0])).attr('y1', d => yScale(wordData[d.from]?.origCoord[1]))
        .attr('x2', d => xScale(wordData[d.to]?.origCoord[0])).attr('y2', d => yScale(wordData[d.to]?.origCoord[1]));
    }
  });
}

/**
 * Animated subspace identification walkthrough.
 * Steps: Pairs → Differences → Direction → Projection → Steer.
 *
 * wordData: array of { word, coord: [x,y] }
 * options.pairs: array of { from, to } indices into wordData
 * options.directionLabel: label for the direction arrow
 */
function renderSubspaceAnimation(container, wordData, options = {}) {
  const el = typeof container === 'string' ? document.getElementById(container) : container;
  if (!el) return;
  const defaultW = getResponsiveWidth(el);
  const { width = defaultW, height = Math.round(defaultW * 0.55) } = options;
  const { pairs = [], directionLabel = 'direction' } = options;
  const { mdsData = null, eigenContainer = null } = options;
  const margin = { top: 0, right: 10, bottom: 30, left: 10 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;

  // Mutable state shared across rebuilds
  let dims = 2;
  let rotationAngle = 0, tiltAngle = 0.4;
  let rotationAnim = null;

  function stopRotation() {
    if (rotationAnim) { cancelAnimationFrame(rotationAnim); rotationAnim = null; }
  }

  // Compute direction + steered coords in the given dimensionality
  function computeDerived(ndims, rawCoords) {
    for (let i = 0; i < wordData.length; i++) {
      wordData[i].coord = rawCoords[i].slice(0, ndims);
    }
    const centroid = [];
    for (let k = 0; k < ndims; k++) centroid.push(d3.mean(wordData, d => d.coord[k]));

    const dd = new Array(ndims).fill(0);
    for (const p of pairs) {
      for (let k = 0; k < ndims; k++) dd[k] += wordData[p.to].coord[k] - wordData[p.from].coord[k];
    }
    const dLen = Math.sqrt(dd.reduce((s, v) => s + v * v, 0)) || 1;
    const dirVec = dd.map(v => v / dLen);

    for (const d of wordData) {
      const v = d.coord.map((c, k) => c - centroid[k]);
      const proj = v.reduce((s, vi, k) => s + vi * dirVec[k], 0);
      d.steeredCoord = d.coord.map((c, k) => c - proj * dirVec[k]);
    }
    return { centroid, dirVec };
  }

  // Build the full scene; called on init and when switching dims.
  function buildScene() {
    stopRotation();
    d3.select(el).selectAll('svg.plot').remove();
    d3.select(el).selectAll('.anim-controls').remove();

    const rawCoords = mdsData ? mdsData.coords[dims] : wordData.map(d => d.coord);
    const { centroid, dirVec } = computeDerived(dims, rawCoords);

    // Direction arrow length
    const maxPairLen = Math.max(...pairs.map(p => {
      const diff = wordData[p.to].coord.map((v, k) => v - wordData[p.from].coord[k]);
      return Math.sqrt(diff.reduce((s, v) => s + v * v, 0));
    })) || 1;
    const dirArrowLen = maxPairLen * 0.8 * 1.6;

    // --- Scales ---
    let xScale, yScale;
    let norm3D = null, normSteered3D = null, normCentroid3D = null;
    let normDirEnd3D = null, normDiffEnds3D = null;

    if (dims === 3) {
      // Collect every 3D point that will be drawn, normalize together
      const allPts = [];
      for (const d of wordData) { allPts.push(d.coord); allPts.push(d.steeredCoord); }
      allPts.push(centroid);
      allPts.push(centroid.map((c, k) => c + dirVec[k] * dirArrowLen));
      for (const p of pairs) {
        const diff = wordData[p.to].coord.map((v, k) => v - wordData[p.from].coord[k]);
        allPts.push(centroid.map((c, k) => c + diff[k]));
      }
      const maxR = Math.max(...allPts.map(([x, y, z]) => Math.sqrt(x*x + y*y + z*z))) || 1;
      const nf = pt => pt.map(v => v / maxR);
      norm3D = wordData.map(d => nf(d.coord));
      normSteered3D = wordData.map(d => nf(d.steeredCoord));
      normCentroid3D = nf(centroid);
      normDirEnd3D = nf(centroid.map((c, k) => c + dirVec[k] * dirArrowLen));
      normDiffEnds3D = pairs.map(p => {
        const diff = wordData[p.to].coord.map((v, k) => v - wordData[p.from].coord[k]);
        return nf(centroid.map((c, k) => c + diff[k]));
      });

      const baseDom = [-1.15, 1.15];
      let domX = baseDom, domY = baseDom;
      const baseSpan = baseDom[1] - baseDom[0];
      const mid = (baseDom[0] + baseDom[1]) / 2;
      if (w > h) { const s = baseSpan * (w / h); domX = [mid - s/2, mid + s/2]; }
      else { const s = baseSpan * (h / w); domY = [mid - s/2, mid + s/2]; }
      xScale = d3.scaleLinear().domain(domX).range([0, w]);
      yScale = d3.scaleLinear().domain(domY).range([h, 0]);
    } else {
      // 2D: data-driven domain with equal scaling
      const allX = wordData.flatMap(d => [d.coord[0], d.steeredCoord[0]]);
      const allY = wordData.flatMap(d => [d.coord[1], d.steeredCoord[1]]);
      allX.push(centroid[0] + dirVec[0] * dirArrowLen);
      allY.push(centroid[1] + dirVec[1] * dirArrowLen);
      for (const p of pairs) {
        allX.push(centroid[0] + wordData[p.to].coord[0] - wordData[p.from].coord[0]);
        allY.push(centroid[1] + wordData[p.to].coord[1] - wordData[p.from].coord[1]);
      }
      const pad = 0.12;
      const xMin = Math.min(...allX), xMax = Math.max(...allX);
      const yMin = Math.min(...allY), yMax = Math.max(...allY);
      const xSpan = (xMax - xMin) || 0.2, ySpan = (yMax - yMin) || 0.2;
      const dataW = xSpan + 2 * xSpan * pad, dataH = ySpan + 2 * ySpan * pad;
      const aspect = w / h;
      let domW = dataW, domH = dataH;
      if (dataW / dataH > aspect) domH = domW / aspect;
      else domW = domH * aspect;
      const cxDom = (xMin + xMax) / 2, cyDom = (yMin + yMax) / 2;
      xScale = d3.scaleLinear().domain([cxDom - domW/2, cxDom + domW/2]).range([0, w]);
      yScale = d3.scaleLinear().domain([cyDom - domH/2, cyDom + domH/2]).range([h, 0]);
    }

    // --- Projection helpers (3D only) ---
    function getProjected() {
      return {
        words: project3Dto2D(norm3D, rotationAngle, tiltAngle),
        steered: project3Dto2D(normSteered3D, rotationAngle, tiltAngle),
        centroid: project3Dto2D([normCentroid3D], rotationAngle, tiltAngle)[0],
        dirEnd: project3Dto2D([normDirEnd3D], rotationAngle, tiltAngle)[0],
        diffs: project3Dto2D(normDiffEnds3D, rotationAngle, tiltAngle),
      };
    }

    function wordXY(d, i) {
      if (dims === 2) return [xScale(d.coord[0]), yScale(d.coord[1])];
      const p = project3Dto2D([norm3D[i]], rotationAngle, tiltAngle)[0];
      return [xScale(p[0]), yScale(p[1])];
    }

    // --- SVG setup ---
    const svg = d3.select(el).append('svg')
      .attr('class', 'plot').attr('width', width).attr('height', height);
    if (dims === 3) svg.style('cursor', 'grab');
    const cid = el.id || 'anim';
    const defs = svg.append('defs');
    for (const [suffix, color] of [['', COLORS.arrow], ['-dir', '#c0392b']]) {
      defs.append('marker')
        .attr('id', `arrow${suffix}-${cid}`)
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 8).attr('refY', 5)
        .attr('markerWidth', 6).attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 Z').attr('fill', color);
    }

    const mainG = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // --- Create all SVG elements ---
    const pairArrows = mainG.selectAll('line.pair-arrow').data(pairs).enter().append('line')
      .attr('class', 'pair-arrow')
      .attr('x1', d => wordXY(wordData[d.from], d.from)[0])
      .attr('y1', d => wordXY(wordData[d.from], d.from)[1])
      .attr('x2', d => wordXY(wordData[d.to], d.to)[0])
      .attr('y2', d => wordXY(wordData[d.to], d.to)[1])
      .attr('stroke', COLORS.arrow).attr('stroke-width', 1.5)
      .attr('marker-end', `url(#arrow-${cid})`);

    const transArrows = mainG.selectAll('line.trans-arrow').data(pairs).enter().append('line')
      .attr('class', 'trans-arrow')
      .attr('x1', d => wordXY(wordData[d.from], d.from)[0])
      .attr('y1', d => wordXY(wordData[d.from], d.from)[1])
      .attr('x2', d => wordXY(wordData[d.to], d.to)[0])
      .attr('y2', d => wordXY(wordData[d.to], d.to)[1])
      .attr('stroke', COLORS.arrow).attr('stroke-width', 1.5)
      .attr('marker-end', `url(#arrow-${cid})`)
      .attr('opacity', 0);

    // Direction arrow
    const cXY = dims === 2 ? [xScale(centroid[0]), yScale(centroid[1])]
      : (() => { const p = project3Dto2D([normCentroid3D], rotationAngle, tiltAngle)[0]; return [xScale(p[0]), yScale(p[1])]; })();
    const deXY = dims === 2
      ? [xScale(centroid[0] + dirVec[0] * dirArrowLen), yScale(centroid[1] + dirVec[1] * dirArrowLen)]
      : (() => { const p = project3Dto2D([normDirEnd3D], rotationAngle, tiltAngle)[0]; return [xScale(p[0]), yScale(p[1])]; })();
    const dirArrow = mainG.append('line')
      .attr('x1', cXY[0]).attr('y1', cXY[1]).attr('x2', deXY[0]).attr('y2', deXY[1])
      .attr('stroke', '#c0392b').attr('stroke-width', 2.5)
      .attr('stroke-dasharray', '8,4')
      .attr('marker-end', `url(#arrow-dir-${cid})`)
      .attr('opacity', 0);

    const dirLabelEl = mainG.append('text')
      .attr('x', deXY[0]).attr('y', deXY[1] - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px').attr('font-weight', 'bold').attr('font-style', 'italic')
      .attr('fill', '#c0392b').text(directionLabel)
      .attr('opacity', 0);

    // Perpendicular axis (2D only; in 3D the perpendicular complement is a plane)
    let perpAxis;
    if (dims === 2) {
      const perpExtent = Math.max(...wordData.map(d => {
        const vx = d.steeredCoord[0] - centroid[0], vy = d.steeredCoord[1] - centroid[1];
        return Math.abs(vx * (-dirVec[1]) + vy * dirVec[0]);
      })) * 1.2 || 1;
      perpAxis = mainG.append('line').attr('class', 'perp-axis')
        .attr('x1', xScale(centroid[0] - (-dirVec[1]) * perpExtent))
        .attr('y1', yScale(centroid[1] - dirVec[0] * perpExtent))
        .attr('x2', xScale(centroid[0] + (-dirVec[1]) * perpExtent))
        .attr('y2', yScale(centroid[1] + dirVec[0] * perpExtent))
        .attr('stroke', '#999').attr('stroke-width', 1).attr('stroke-dasharray', '6,4')
        .attr('opacity', 0);
    } else {
      perpAxis = mainG.append('line').attr('class', 'perp-axis').attr('opacity', 0);
    }

    const projLines = mainG.selectAll('line.proj-line').data(wordData).enter().append('line')
      .attr('class', 'proj-line')
      .attr('x1', (d, i) => wordXY(d, i)[0]).attr('y1', (d, i) => wordXY(d, i)[1])
      .attr('x2', (d, i) => wordXY(d, i)[0]).attr('y2', (d, i) => wordXY(d, i)[1])
      .attr('stroke', '#c0392b').attr('stroke-width', 1).attr('stroke-dasharray', '3,2')
      .attr('opacity', 0);

    const ghosts = mainG.selectAll('circle.ghost').data(wordData).enter().append('circle')
      .attr('class', 'ghost')
      .attr('cx', (d, i) => wordXY(d, i)[0]).attr('cy', (d, i) => wordXY(d, i)[1])
      .attr('r', 3).attr('fill', COLORS.point).attr('opacity', 0);

    const dots = mainG.selectAll('circle.word-dot').data(wordData).enter().append('circle')
      .attr('class', 'word-dot')
      .attr('cx', (d, i) => wordXY(d, i)[0]).attr('cy', (d, i) => wordXY(d, i)[1])
      .attr('r', 3.5).attr('fill', COLORS.point);

    const labelEls = mainG.selectAll('text.word-label').data(wordData).enter().append('text')
      .attr('class', 'word-label')
      .attr('x', (d, i) => wordXY(d, i)[0]).attr('y', (d, i) => wordXY(d, i)[1] - 8)
      .attr('text-anchor', 'middle').attr('font-size', '11px').attr('fill', '#333')
      .text(d => d.word);

    // --- Step controls ---
    let currentStep = 0;
    const km = (tex) => katex.renderToString(tex, { throwOnError: false });
    const stepDescs = [
      `Each arrow connects a word pair \u2014 e.g. ${km('\\overrightarrow{\\text{woman}}')} to ${km('\\overrightarrow{\\text{man}}')}.`,
      `Translate each pair\u2019s difference vector ${km('d_i = \\overrightarrow{w}_i^+ - \\overrightarrow{w}_i^-')} to a common origin. They point in roughly the same direction.`,
      `The top eigenvector of ${km('C = \\tfrac{1}{2}\\sum d_i\\,d_i^T')} gives the direction that best explains these differences.`,
      `Each word\u2019s component along the direction (its \u201cgender component\u201d ${km('\\overrightarrow{w}_{\\mathcal{B}}')}) is shown as a dashed line.`,
      `Project out: ${km('\\overrightarrow{w}_{\\text{steered}} = \\overrightarrow{w} - \\overrightarrow{w}_{\\mathcal{B}}')}`,
    ];

    const descFO = svg.append('foreignObject')
      .attr('x', margin.left).attr('y', height - margin.bottom - 6)
      .attr('width', w).attr('height', margin.bottom + 6);
    const descDiv = descFO.append('xhtml:div')
      .style('font-size', '13px').style('color', '#555').style('line-height', '1.3')
      .style('background', 'rgba(255,255,255,0.85)')
      .style('padding', '2px 4px').style('border-radius', '3px');

    const ctrlDiv = d3.select(el).append('div').attr('class', 'anim-controls')
      .style('margin-top', '4px').style('display', 'flex').style('gap', '8px').style('align-items', 'center');
    const btn = ctrlDiv.append('button')
      .style('background', COLORS.point).style('color', 'white').style('border', 'none')
      .style('border-radius', '4px').style('padding', '5px 14px').style('font-size', '13px')
      .style('cursor', 'pointer').text('Next \u25b6');
    const stepCounter = ctrlDiv.append('span')
      .style('font-size', '12px').style('color', '#999');

    function updateStatus() {
      stepCounter.text(`Step ${currentStep + 1} of ${stepDescs.length}`);
      descDiv.html(stepDescs[currentStep]);
    }
    updateStatus();

    // --- goTo: step transitions ---
    function goTo(s) {
      currentStep = s;
      updateStatus();
      const dur = 800;

      if (dims === 2) {
        // 2D transitions (original logic)
        if (s === 0) {
          btn.text('Next \u25b6').style('background', COLORS.point).style('color', 'white');
          pairArrows.transition().duration(dur).attr('opacity', 1)
            .attr('stroke', COLORS.arrow).attr('stroke-width', 1.5);
          transArrows.transition().duration(dur).attr('opacity', 0)
            .attr('x1', d => xScale(wordData[d.from].coord[0]))
            .attr('y1', d => yScale(wordData[d.from].coord[1]))
            .attr('x2', d => xScale(wordData[d.to].coord[0]))
            .attr('y2', d => yScale(wordData[d.to].coord[1]));
          dirArrow.transition().duration(dur).attr('opacity', 0);
          dirLabelEl.transition().duration(dur).attr('opacity', 0);
          perpAxis.transition().duration(dur).attr('opacity', 0);
          projLines.transition().duration(dur).attr('opacity', 0)
            .attr('x1', d => xScale(d.coord[0])).attr('y1', d => yScale(d.coord[1]))
            .attr('x2', d => xScale(d.coord[0])).attr('y2', d => yScale(d.coord[1]));
          ghosts.transition().duration(dur).attr('opacity', 0);
          dots.transition().duration(dur)
            .attr('cx', d => xScale(d.coord[0])).attr('cy', d => yScale(d.coord[1]));
          labelEls.transition().duration(dur)
            .attr('x', d => xScale(d.coord[0])).attr('y', d => yScale(d.coord[1]) - 8);
        } else if (s === 1) {
          pairArrows.transition().duration(dur).attr('opacity', 0.3);
          transArrows
            .attr('stroke', COLORS.highlight).attr('stroke-width', 2).attr('opacity', 1);
          transArrows.transition().duration(dur * 1.5).ease(d3.easeCubicInOut)
            .attr('x1', xScale(centroid[0])).attr('y1', yScale(centroid[1]))
            .attr('x2', d => xScale(centroid[0] + wordData[d.to].coord[0] - wordData[d.from].coord[0]))
            .attr('y2', d => yScale(centroid[1] + wordData[d.to].coord[1] - wordData[d.from].coord[1]))
            .attr('stroke', COLORS.arrow).attr('stroke-width', 1.5);
        } else if (s === 2) {
          transArrows.transition().duration(dur).attr('opacity', 0.35);
          dirArrow.transition().duration(dur).attr('opacity', 0.9);
          dirLabelEl.transition().duration(dur).attr('opacity', 1);
        } else if (s === 3) {
          transArrows.transition().duration(dur).attr('opacity', 0);
          perpAxis.transition().duration(dur).attr('opacity', 0.5);
          projLines.transition().duration(dur)
            .attr('x2', d => xScale(d.steeredCoord[0]))
            .attr('y2', d => yScale(d.steeredCoord[1]))
            .attr('opacity', 0.6);
        } else if (s === 4) {
          btn.text('Reset').style('background', '#ddd').style('color', '#333');
          ghosts.transition().duration(300).attr('opacity', 0.4);
          pairArrows.transition().duration(dur).attr('opacity', 0);
          dots.transition().duration(1200).ease(d3.easeCubicInOut)
            .attr('cx', d => xScale(d.steeredCoord[0]))
            .attr('cy', d => yScale(d.steeredCoord[1]));
          labelEls.transition().duration(1200).ease(d3.easeCubicInOut)
            .attr('x', d => xScale(d.steeredCoord[0]))
            .attr('y', d => yScale(d.steeredCoord[1]) - 8);
          perpAxis.transition().delay(800).duration(600).attr('opacity', 0);
          projLines.transition().delay(500).duration(800)
            .attr('x1', d => xScale(d.steeredCoord[0]))
            .attr('y1', d => yScale(d.steeredCoord[1]))
            .attr('opacity', 0);
        }
      } else {
        // 3D: update opacities/button; rotation loop handles positions
        if (s === 0) {
          btn.text('Next \u25b6').style('background', COLORS.point).style('color', 'white');
          pairArrows.transition().duration(dur).attr('opacity', 1)
            .attr('stroke', COLORS.arrow).attr('stroke-width', 1.5);
          transArrows.transition().duration(dur).attr('opacity', 0);
          dirArrow.transition().duration(dur).attr('opacity', 0);
          dirLabelEl.transition().duration(dur).attr('opacity', 0);
          projLines.transition().duration(dur).attr('opacity', 0);
          ghosts.transition().duration(dur).attr('opacity', 0);
        } else if (s === 1) {
          pairArrows.transition().duration(dur).attr('opacity', 0.3);
          transArrows.attr('stroke', COLORS.highlight).attr('stroke-width', 2).attr('opacity', 1);
          transArrows.transition().duration(dur * 1.5).ease(d3.easeCubicInOut)
            .attr('stroke', COLORS.arrow).attr('stroke-width', 1.5);
        } else if (s === 2) {
          transArrows.transition().duration(dur).attr('opacity', 0.35);
          dirArrow.transition().duration(dur).attr('opacity', 0.9);
          dirLabelEl.transition().duration(dur).attr('opacity', 1);
        } else if (s === 3) {
          transArrows.transition().duration(dur).attr('opacity', 0);
          projLines.transition().duration(dur).attr('opacity', 0.6);
        } else if (s === 4) {
          btn.text('Reset').style('background', '#ddd').style('color', '#333');
          ghosts.transition().duration(300).attr('opacity', 0.4);
          pairArrows.transition().duration(dur).attr('opacity', 0);
          projLines.transition().delay(500).duration(800).attr('opacity', 0);
        }
      }
    }

    btn.on('click', () => {
      if (currentStep >= 4) goTo(0);
      else goTo(currentStep + 1);
    });

    // --- 3D rotation loop ---
    if (dims === 3) {
      let dragging = false, dragStartX = 0, dragStartY = 0;
      let dragStartAngle = 0, dragStartTilt = 0;
      let autoRotate = true, resumeTimeout = null;
      const svgNode = svg.node();

      svgNode.addEventListener('pointerdown', (e) => {
        if (e.target.tagName === 'circle' || e.target.tagName === 'text') return;
        dragging = true; dragStartX = e.clientX; dragStartY = e.clientY;
        dragStartAngle = rotationAngle; dragStartTilt = tiltAngle;
        autoRotate = false; clearTimeout(resumeTimeout);
        svgNode.style.cursor = 'grabbing';
        svgNode.setPointerCapture(e.pointerId);
      });
      svgNode.addEventListener('pointermove', (e) => {
        if (!dragging) return;
        rotationAngle = dragStartAngle + (e.clientX - dragStartX) * 0.01;
        tiltAngle = Math.max(-Math.PI/2, Math.min(Math.PI/2,
          dragStartTilt + (e.clientY - dragStartY) * 0.01));
      });
      svgNode.addEventListener('pointerup', () => {
        if (!dragging) return;
        dragging = false; svgNode.style.cursor = 'grab';
        resumeTimeout = setTimeout(() => { autoRotate = true; }, 3000);
      });

      // Smooth blend targets for step transitions
      let targetBlend = 0, currentBlend = 0;       // 0=orig, 1=steered
      let targetTransBlend = 0, currentTransBlend = 0; // 0=pair pos, 1=centroid

      function tick() {
        if (autoRotate && !dragging) rotationAngle += 0.005;
        const blendSpeed = 0.04;
        targetBlend = currentStep === 4 ? 1 : 0;
        currentBlend += (targetBlend - currentBlend) * blendSpeed;
        targetTransBlend = currentStep >= 1 ? 1 : 0;
        currentTransBlend += (targetTransBlend - currentTransBlend) * blendSpeed;

        const proj = getProjected();

        // Word dots: blend orig to steered
        dots.each(function(d, i) {
          const ox = xScale(proj.words[i][0]), oy = yScale(proj.words[i][1]);
          const sx = xScale(proj.steered[i][0]), sy = yScale(proj.steered[i][1]);
          d3.select(this).attr('cx', ox + (sx - ox) * currentBlend)
            .attr('cy', oy + (sy - oy) * currentBlend);
        });
        labelEls.each(function(d, i) {
          const ox = xScale(proj.words[i][0]), oy = yScale(proj.words[i][1]);
          const sx = xScale(proj.steered[i][0]), sy = yScale(proj.steered[i][1]);
          d3.select(this).attr('x', ox + (sx - ox) * currentBlend)
            .attr('y', (oy + (sy - oy) * currentBlend) - 8);
        });
        ghosts.each(function(d, i) {
          d3.select(this).attr('cx', xScale(proj.words[i][0]))
            .attr('cy', yScale(proj.words[i][1]));
        });

        // Pair arrows
        pairArrows.each(function(d) {
          d3.select(this)
            .attr('x1', xScale(proj.words[d.from][0])).attr('y1', yScale(proj.words[d.from][1]))
            .attr('x2', xScale(proj.words[d.to][0])).attr('y2', yScale(proj.words[d.to][1]));
        });

        // Translated arrows: blend from pair position to centroid
        transArrows.each(function(d, pi) {
          const fx = xScale(proj.words[d.from][0]), fy = yScale(proj.words[d.from][1]);
          const tx = xScale(proj.words[d.to][0]), ty = yScale(proj.words[d.to][1]);
          const ccx = xScale(proj.centroid[0]), ccy = yScale(proj.centroid[1]);
          const dx = xScale(proj.diffs[pi][0]), dy = yScale(proj.diffs[pi][1]);
          d3.select(this)
            .attr('x1', fx + (ccx - fx) * currentTransBlend)
            .attr('y1', fy + (ccy - fy) * currentTransBlend)
            .attr('x2', tx + (dx - tx) * currentTransBlend)
            .attr('y2', ty + (dy - ty) * currentTransBlend);
        });

        // Direction arrow + label
        dirArrow
          .attr('x1', xScale(proj.centroid[0])).attr('y1', yScale(proj.centroid[1]))
          .attr('x2', xScale(proj.dirEnd[0])).attr('y2', yScale(proj.dirEnd[1]));
        dirLabelEl
          .attr('x', xScale(proj.dirEnd[0])).attr('y', yScale(proj.dirEnd[1]) - 10);

        // Projection lines
        projLines.each(function(d, i) {
          const ox = xScale(proj.words[i][0]), oy = yScale(proj.words[i][1]);
          const sx = xScale(proj.steered[i][0]), sy = yScale(proj.steered[i][1]);
          d3.select(this)
            .attr('x1', ox + (sx - ox) * currentBlend)
            .attr('y1', oy + (sy - oy) * currentBlend)
            .attr('x2', sx).attr('y2', sy);
        });

        rotationAnim = requestAnimationFrame(tick);
      }
      rotationAnim = requestAnimationFrame(tick);
    }
  } // end buildScene

  // --- Eigenvalue selector ---
  if (mdsData && eigenContainer) {
    const eigenEl = typeof eigenContainer === 'string'
      ? document.getElementById(eigenContainer) : eigenContainer;
    if (eigenEl) {
      eigenEl.innerHTML = '';
      const selector = createEigenSelector(eigenEl, mdsData.eigenvalues, dims, (newDims) => {
        if (newDims === 1 || newDims === dims) return; // 1D not supported
        dims = newDims;
        selector.update(dims, mdsData.variance[dims]);
        buildScene();
      });
      // Gray out bar 1 (1D not supported for this animation)
      const bars = d3.select(eigenEl).selectAll('rect.bar');
      bars.filter((d, i) => i === 0)
        .style('opacity', 0.4).style('cursor', 'not-allowed');
      selector.update(dims, mdsData.variance[dims]);
    }
  }

  // Initial build
  buildScene();
}

export { EmbeddingViz, SteeringViz, computeAllMDS, computeAllMDSFromMatrix, render2D, render3D, render1D, renderHero3D, renderSteering2D, renderSubspaceAnimation };
