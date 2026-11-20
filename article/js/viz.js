/**
 * Embedding visualizations using D3 for all rendering (1D, 2D, projected 3D).
 * Three.js is only used for the hero visualization.
 */
import { mds, sortedSVD } from './embeddings.js';

const COLORS = {
  point: '#4a6a8a',
  arrow: 'rgba(74, 106, 138, 0.45)',
  highlight: '#5e8c61',
  crossGroup: 'rgba(94, 140, 97, 0.25)',
  eigenActive: '#4a6a8a',
  eigenInactive: '#e0e0e0',
  direction: '#9e4a4a',
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

// ── One view for 1D, 2D and 3D ─────────────────────────────────────────
// A plot's state is { w1, w2, w3, angle, tilt, k, H }: points are drawn at
// project(k * (w1*P1 + w2*P2 + w3*P3), angle, tilt) on a canvas H pixels tall, where P1, P2, P3
// are the 1D, 2D and 3D layouts as 3-vectors (zero-padded) in one unit. Classical MDS nests:
// the 1D layout is the 2D layout's x, and the 2D layout is the 3D layout's first two
// coordinates. So every view is a setting of these numbers, and switching dimension animates
// them: 1D -> 2D lets y in and turns the labels upright, 2D -> 3D adds depth, tilts, zooms out
// to fit the spinning cloud and ramps up the spin, and the reverse turns back the short way
// and flattens. Every switch ends exactly on the target layout.
const SPIN_TILT = 0.4;     // tilt while spinning (radians)
const SPIN_SPEED = 0.005;  // radians per 1/60 s
const SWITCH_MS = 900;     // duration of a dimension switch (up to 1.5x for a big turn back)
// Easing for every animated move: smoothstep, whose peak speed is 1.5x the average (d3's
// cubic in-out peaks at 3x, which makes long moves skip visibly from frame to frame).
const smooth = u => u * u * (3 - 2 * u);

function viewProject(L, st) {
  const { w1, w2, w3, k } = st;
  return project3Dto2D(L.P3.map((p3, i) => {
    const p1 = L.P1[i], p2 = L.P2[i];
    return [0, 1, 2].map(j => k * (w1 * p1[j] + w2 * p2[j] + w3 * p3[j]));
  }), st.angle, st.tilt);
}

// The state that shows `dims`, for canvas geometry g = { H1, H2, k3 }.
function viewTarget(dims, g) {
  if (dims === 1) return { w1: 1, w2: 0, w3: 0, tilt: 0, k: 1, H: g.H1 };
  if (dims === 2) return { w1: 0, w2: 1, w3: 0, tilt: 0, k: 1, H: g.H2 };
  return { w1: 0, w2: 0, w3: 1, tilt: SPIN_TILT, k: g.k3, H: g.H2 };
}

// Zoom factor (at most 1) that keeps a cloud spinning about the vertical axis at SPIN_TILT
// within half-widths hx, hy of the origin, with 8% room for labels.
function spinFitZoom(P3, hx, hy) {
  let rh = 0, rv = 0;
  for (const [x, y, z] of P3) {
    const r = Math.hypot(x, z);
    rh = Math.max(rh, r);
    rv = Math.max(rv, Math.abs(y) * Math.cos(SPIN_TILT) + r * Math.sin(SPIN_TILT));
  }
  return Math.min(1, hx / (1.08 * rh || 1), hy / (1.08 * rv || 1));
}

// Arrowhead for a line from (x1, y1) to (x2, y2) with stroke width sw: the triangle the SVG
// marker used to draw (6*sw long and wide, tip 1.2*sw past the end), drawn as a path so it
// moves with its line in every browser. It shrinks on lines shorter than twice its length
// (so a short arrow isn't all head, and the head can't flip around as a spinning arrow
// passes through zero length) and is omitted under 1px.
function arrowHeadPath(x1, y1, x2, y2, sw) {
  const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy);
  if (len < 1) return '';
  const ux = dx / len, uy = dy / len;
  const f = Math.min(1, len / (12 * sw));
  const hl = 6 * sw * f, hw = 3 * sw * f;
  const tx = x2 + ux * 1.2 * sw * f, ty = y2 + uy * 1.2 * sw * f;
  const bx = tx - ux * hl, by = ty - uy * hl;
  return `M${tx},${ty}L${bx - uy * hw},${by + ux * hw}L${bx + uy * hw},${by - ux * hw}Z`;
}

// Pan + zoom for 1D/2D plots (3D uses drag-to-orbit instead).
function attachPanZoom(svg) {
  const zoomG = svg.select('g.zoom-container');
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

// In 3D, drag orbits and the wheel scrolls the page: remove pan/zoom and ease out any pan.
function detachPanZoom(svg) {
  svg.on('.zoom', null);
  const zoomG = svg.select('g.zoom-container');
  if (zoomG.attr('transform')) {
    zoomG.transition().duration(SWITCH_MS).ease(smooth).attr('transform', 'translate(0,0) scale(1)');
  }
  if (svg.node()) svg.node().__zoom = d3.zoomIdentity;
}

// Animation state for one plot: dimension switches, the 3D spin, and drag-to-orbit.
// draw(state) redraws the plot for a state.
class DimView {
  constructor(draw) {
    this.draw = draw;
    this.state = null;
    this.spinning = false;    // showing 3D, or on the way there
    this._anim = null;
    this._raf = null;
    this._dragging = false;
    this._autoRotate = true;
    this._resume = null;
    this._visible = true;
    this._gain = 1;           // spin speed, as a fraction of SPIN_SPEED
    this.svgNode = null;
  }

  // Drag-to-orbit on this svg (in 3D): horizontal drag turns, vertical drag tilts. The spin
  // pauses while the plot is off screen (switches still finish), so plots left spinning
  // don't take frames from the one being watched.
  attach(svgNode) {
    if (!svgNode || this.svgNode === svgNode) return;
    this.svgNode = svgNode;
    if (this._io) this._io.disconnect();  // the plot was rebuilt: stop watching the old svg
    // Entries arrive oldest first, several at once when the page is busy: the last is current.
    this._io = new IntersectionObserver((es) => {
      this._visible = es[es.length - 1].isIntersecting;
      if (this._visible && this.spinning) this._run();
    }, { rootMargin: '100px' });
    this._io.observe(svgNode);
    let x0 = 0, y0 = 0, a0 = 0, t0 = 0;
    svgNode.addEventListener('pointerdown', (e) => {
      if (!this.spinning || this._anim) return;
      if (e.target.tagName === 'circle' || e.target.tagName === 'text') return;
      this._dragging = true; this._autoRotate = false; clearTimeout(this._resume);
      x0 = e.clientX; y0 = e.clientY; a0 = this.state.angle; t0 = this.state.tilt;
      svgNode.style.cursor = 'grabbing';
      svgNode.setPointerCapture(e.pointerId);
    });
    svgNode.addEventListener('pointermove', (e) => {
      if (!this._dragging) return;
      this.state.angle = a0 + (e.clientX - x0) * 0.01;
      this.state.tilt = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, t0 + (e.clientY - y0) * 0.01));
    });
    // pointercancel: the browser took the gesture over (e.g. to scroll the page). Without
    // it the plot would think the drag never ended and stop turning.
    const end = () => {
      if (!this._dragging) return;
      this._dragging = false; svgNode.style.cursor = 'grab';
      this._resume = setTimeout(() => { this._autoRotate = true; }, 3000);
    };
    svgNode.addEventListener('pointerup', end);
    svgNode.addEventListener('pointercancel', end);
    this._touch3D(this.spinning);
  }

  // Touch in 3D: a horizontal drag turns the plot; a vertical swipe still scrolls the page.
  _touch3D(on) { if (this.svgNode) this.svgNode.style.touchAction = on ? 'pan-y pinch-zoom' : ''; }

  // Show `dims` now, without animating.
  set(dims, g) {
    this.stop();
    this.state = { angle: 0, ...viewTarget(dims, g) };
    this.spinning = dims === 3; this._autoRotate = true; this._gain = 1; this._touch3D(this.spinning);
    this.draw(this.state);
    if (this.spinning) this._run();
  }

  // Animate to `dims`; onDone runs once the plot is exactly there (not if interrupted).
  go(dims, g, onDone) {
    const to = viewTarget(dims, g);
    let dur = SWITCH_MS, v0 = 0;
    if (dims !== 3) {
      // Turn back to a whole turn, so 1D/2D are seen face-on: the one nearest to where the
      // plot's current turning carries it, starting at that turning speed (so a spinning plot
      // doesn't stop dead first). A bigger turn takes longer, up to 1.5x, so it doesn't whip
      // around. The starting speed is capped at 4 rad/s: a faster drag-flick would overshoot.
      const a = this.state.angle;
      v0 = Math.max(-0.004, Math.min(0.004, this._omega || 0));  // rad/ms
      to.angle = 2 * Math.PI * Math.round((a + v0 * SWITCH_MS / 3) / (2 * Math.PI));
      dur = SWITCH_MS * (1 + 0.5 * Math.min(1, Math.abs(to.angle - a) / Math.PI));
    }
    this.spinning = dims === 3; this._dragging = false; this._autoRotate = true;
    clearTimeout(this._resume); this._touch3D(this.spinning);
    this._anim = { from: { ...this.state }, to, dur, v0, t0: null, onDone: () => {
      if (dims !== 3) this.state.angle = 0;
      if (onDone) onDone();
    } };
    this._run();
  }

  // Redraw the current state (e.g. after label widths change).
  redraw() { if (this.state) this.draw(this.state); }

  // Whether a frame is scheduled (spinning or switching): it will redraw on its own.
  get running() { return this._raf !== null; }

  stop() {
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null; this._anim = null; this._tPrev = undefined; this._aPrev = undefined; this._omega = 0;
  }

  _run() {
    if (this._raf) return;
    const step = (now) => {
      this._raf = null;
      // Spin by elapsed time, not per frame, so uneven frames don't change its speed
      // (capped, so a stalled or background tab doesn't jump).
      const dt = this._tPrev === undefined ? 0 : now - this._tPrev;
      const frames = dt === 0 ? 1 : Math.min(3, dt / (1000 / 60));
      this._tPrev = now;
      const st = this.state, an = this._anim, a0 = this._aPrev;
      let finished = false;
      if (an) {
        // Smoothstep, plus the starting turn speed v0 fading out (cubic Hermite, end speed 0).
        if (an.t0 === null) an.t0 = now;
        const u = Math.min(1, (now - an.t0) / an.dur), e = smooth(u);
        for (const key of Object.keys(an.to)) st[key] = an.from[key] + (an.to[key] - an.from[key]) * e;
        if (an.v0) st.angle += an.v0 * an.dur * u * (1 - u) * (1 - u);
        finished = u >= 1;
      }
      // After a drag the spin eases back in over about a second instead of starting at full speed.
      const turning = this._autoRotate && !this._dragging;
      this._gain = !turning ? 0 : this._gain + (1 - this._gain) * (1 - Math.exp(-dt / 400));
      if (this.spinning && turning) st.angle += SPIN_SPEED * st.w3 * frames * this._gain;
      // How fast the plot is turning (rad/ms), drags included, for the next switch to start from.
      this._omega = dt > 0 && a0 !== undefined ? (st.angle - a0) / dt : 0;
      this._aPrev = st.angle;
      this.draw(st);
      if (finished && this._anim === an) {
        this._anim = null;
        an.onDone();
      }
      if (!this._raf && ((this.spinning && this._visible) || this._anim)) this._raf = requestAnimationFrame(step);
      else if (!this._raf) { this._tPrev = undefined; this._aPrev = undefined; this._omega = 0; }
    };
    this._raf = requestAnimationFrame(step);
  }
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
  const barH = 30, barW = 14, gap = 4, labelH = 16, minBarH = 4;
  const svgW = topEig.length * (barW + gap) - gap;
  const svgH = barH + labelH;

  const row = d3.select(container).append('div')
    .style('display', 'inline-flex')
    .style('align-items', 'flex-end')
    .style('gap', '4px')
    .style('user-select', 'none')
    .style('padding', '3px 6px')
    .style('border-radius', '3px');

  const svg = row.append('svg')
    .attr('width', svgW).attr('height', svgH)
    .style('cursor', 'pointer');

  svg.append('title').text('Click to switch dimensions');

  // Each column is a clickable group containing the bar, a hit area, and a label
  const groups = svg.selectAll('g.eigen-col')
    .data(topEig)
    .enter().append('g')
    .attr('class', 'eigen-col')
    .attr('transform', (d, i) => `translate(${i * (barW + gap)}, 0)`)
    .style('cursor', 'pointer');

  // Invisible hit area covering the full column height
  groups.append('rect')
    .attr('class', 'hit-area')
    .attr('x', 0).attr('y', 0)
    .attr('width', barW).attr('height', svgH)
    .attr('fill', 'transparent');

  // Visible bar (with minimum height so small eigenvalues are still clickable)
  const bars = groups.append('rect')
    .attr('class', 'bar')
    .attr('x', 0)
    .attr('y', d => {
      const h = maxVal > 0 ? Math.max(minBarH, Math.max(0, d) / maxVal * barH) : minBarH;
      return barH - h;
    })
    .attr('width', barW)
    .attr('height', d => maxVal > 0 ? Math.max(minBarH, Math.max(0, d) / maxVal * barH) : minBarH)
    .attr('fill', (d, i) => i < activeDims ? '#aac4de' : '#e8e8e8')
    .attr('rx', 2);

  // Dimension label below bar
  groups.append('text')
    .attr('class', 'label')
    .attr('x', barW / 2)
    .attr('y', barH + 12)
    .attr('text-anchor', 'middle')
    .attr('font-size', '11px')
    .attr('fill', '#aaa')
    .text((d, i) => `${i + 1}d`);

  // Click handler on the whole group
  groups.on('click', function() {
    const newDims = groups.nodes().indexOf(this) + 1;  // by position: two eigenvalues can be equal
    bars.attr('fill', (d, i) => i < newDims ? '#aac4de' : '#e8e8e8');
    onChange(newDims);
  });

  const varianceText = row.append('span')
    .style('font-size', '11px')
    .style('color', '#aaa')
    .style('margin-left', '3px');

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
    queryPoints = new Set(),   // indices of analogy query points (× marker)
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
    if (!disableZoom) attachPanZoom(svg);
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
    .attr('stroke-width', d => d.thick ? 2 : 1.5)
    .attr('stroke', d => d.color === 'red' ? COLORS.highlight : COLORS.arrow)
    .attr('opacity', d => d.thick ? 0.45 : 1)
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
    .attr('r', d => (hiddenPoints.has(d.i) || queryPoints.has(d.i)) ? 0 : highlightSet.has(d.i) ? 5 : 3.5)
    .attr('fill', d => highlightSet.has(d.i) ? COLORS.highlight :
      neighborWords.has(d.word) ? '#999' : COLORS.point);

  // --- Query point markers (× at analogy query positions) ---
  const queryData = pointData.filter(d => queryPoints.has(d.i));
  const qmarks = g.selectAll('text.query-marker').data(queryData, d => d.word);
  qmarks.exit().transition().duration(dur).style('opacity', 0).remove();
  const qmarksEnter = qmarks.enter().append('text').attr('class', 'query-marker')
    .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
    .attr('font-size', '16px').attr('font-weight', 'bold')
    .attr('fill', COLORS.highlight).attr('opacity', 0.8)
    .style('cursor', 'pointer')
    .text('×');
  qmarksEnter.merge(qmarks).transition().duration(dur)
    .attr('x', d => xScale(d.c[0]))
    .attr('y', d => yScale(d.c[1]));

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
    .text(d => queryPoints.has(d.i) ? d.word.replace(/^\u200B/, '') : d.word);
  labelsEnter.merge(labels)
    .attr('text-anchor', labelAnchor)
    .transition().duration(dur)
    .attr('transform', d => labelTransformFor(d))
    .attr('font-size', d => (hiddenPoints.has(d.i) || queryPoints.has(d.i)) ? '13px' : neighborWords.has(d.word) ? '10px' : '11px')
    .attr('font-weight', 'normal')
    .attr('font-style', d => (hiddenPoints.has(d.i) || queryPoints.has(d.i)) ? 'italic' : 'normal')
    .attr('fill', d => (hiddenPoints.has(d.i) || queryPoints.has(d.i)) ? COLORS.highlight : neighborWords.has(d.word) ? '#666' : '#333')
    .attr('opacity', d => (hiddenPoints.has(d.i) || queryPoints.has(d.i)) ? 0.7 : 1)
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

  const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
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
    this.queryPoints = config.queryPoints || new Set();
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

    this._view = new DimView(st => this._draw(st));
    this._switchToken = 0;

    // Create eigen selector (clear any previous one)
    if (this.eigenEl) {
      this.eigenEl.innerHTML = '';
      this.eigenSelector = createEigenSelector(
        this.eigenEl,
        this.mdsData.eigenvalues,
        this.dims,
        (newDims) => this._switchDims(newDims)
      );
    }

    this.render();
  }

  _el() {
    return typeof this.plotEl === 'string' ? document.getElementById(this.plotEl) : this.plotEl;
  }

  // The 1D, 2D and 3D layouts as 3-vectors in one unit, the 2D layout's radius (see DimView).
  _layouts() {
    const raw2 = this.mdsData.coords[2];
    const c = Math.max(...raw2.map(([x, y]) => Math.hypot(x, y))) || 1;
    return {
      P1: this.mdsData.coords[1].map(([x]) => [x / c, 0, 0]),
      P2: raw2.map(([x, y]) => [x / c, y / c, 0]),
      P3: this.mdsData.coords[3].map(([x, y, z]) => [x / c, y / c, z / c]),
    };
  }

  // Canvas geometry shared by all views: one scale (px pixels per unit on both axes),
  // canvas heights for 1D (H1) and 2D/3D (H2), and the 3D zoom-out k3.
  _geometry(el) {
    const L = this._L;
    const plotW = getResponsiveWidth(el);
    const innerW = plotW - 20;  // 10px side margins
    const pad = 0.08;
    let ax = 0, ay = 0;
    for (const [x, y] of L.P2) { ax = Math.max(ax, Math.abs(x)); ay = Math.max(ay, Math.abs(y)); }
    for (const [x] of L.P1) ax = Math.max(ax, Math.abs(x));
    const ex = (ax || 1) * (1 + pad), ey = (ay || 1) * (1 + pad);
    // 2D canvas fits the data's aspect ratio, within 0.3-0.65 of the width
    const H2 = Math.max(Math.round(plotW * 0.3), Math.min(Math.round(plotW * 0.65), Math.round(innerW * ey / ex)));
    const H1 = Math.min(160, Math.round(plotW * 0.25));
    const px = Math.min(innerW / (2 * ex), H2 / (2 * ey));
    const k3 = spinFitZoom(L.P3, innerW / (2 * px), H2 / (2 * px));
    return { plotW, innerW, H1, H2, px, k3 };
  }

  // Build the SVG once; _draw positions everything for a view state.
  _build() {
    const el = this._el();
    const G = this._geom;
    const n = this.words.length;
    const highlightSet = new Set(this.highlights);
    const hidden = this.hiddenPoints, query = this.queryPoints, nb = this.neighborWords;
    d3.select(el).selectAll('svg.plot').remove();
    const svg = d3.select(el).append('svg')
      .attr('class', 'plot').attr('width', G.plotW).attr('height', G.H2)
      .style('cursor', 'grab');
    svg.on('mousedown.cursor', () => svg.style('cursor', 'grabbing'));
    svg.on('mouseup.cursor', () => svg.style('cursor', 'grab'));
    const g = svg.append('g').attr('class', 'zoom-container')
      .append('g').attr('class', 'main').attr('transform', 'translate(10,0)');

    g.selectAll('line.neighbor-link')
      .data(this.neighborLinks.filter(l => l.child < n && l.parent < n)).enter().append('line')
      .attr('class', 'neighbor-link')
      .attr('stroke', '#ccc').attr('stroke-width', 1).attr('stroke-dasharray', '3,2');
    g.selectAll('line.cross-group')
      .data(this.crossGroupLines.filter(([i, j]) => i < n && j < n)).enter().append('line')
      .attr('class', 'cross-group')
      .attr('stroke', COLORS.crossGroup).attr('stroke-width', 1).attr('stroke-dasharray', '4,3');

    // Each arrow is an opaque line and head in a group that carries the transparency, so the
    // line doesn't show darker where it runs under its head. Plain arrows are COLORS.arrow
    // (COLORS.point at 0.45); thick ones are fainter still.
    const arrowData = this.arrows.filter(a => a.from < n && a.to < n);
    const arrowColor = d => d.color === 'red' ? COLORS.highlight : COLORS.point;
    const arrowG = g.selectAll('g.arrow').data(arrowData).enter().append('g')
      .attr('class', 'arrow')
      .attr('opacity', d => (d.color === 'red' ? 1 : 0.45) * (d.thick ? 0.45 : 1));
    arrowG.append('line').attr('class', 'arrow')
      .attr('stroke-width', d => d.thick ? 2 : 1.5)
      .attr('stroke', arrowColor);
    arrowG.append('path').attr('class', 'arrow-head')
      .attr('fill', arrowColor);

    const pointData = this.words.map((word, i) => ({ i, word }));
    g.selectAll('circle.point').data(pointData).enter().append('circle')
      .attr('class', 'point')
      .attr('r', d => (hidden.has(d.i) || query.has(d.i)) ? 0 : highlightSet.has(d.i) ? 5 : 3.5)
      .attr('fill', d => highlightSet.has(d.i) ? COLORS.highlight : nb.has(d.word) ? '#999' : COLORS.point)
      .style('cursor', 'pointer');

    g.selectAll('text.query-marker').data(pointData.filter(d => query.has(d.i))).enter().append('text')
      .attr('class', 'query-marker')
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
      .attr('font-size', '16px').attr('font-weight', 'bold')
      .attr('fill', COLORS.highlight).attr('opacity', 0.8)
      .style('cursor', 'pointer')
      .text('×');

    const special = d => hidden.has(d.i) || query.has(d.i);
    g.selectAll('text.word-label').data(pointData).enter().append('text')
      .attr('class', 'word-label')
      .attr('text-anchor', 'middle')
      .attr('font-size', d => special(d) ? '13px' : nb.has(d.word) ? '10px' : '11px')
      .attr('font-style', d => special(d) ? 'italic' : 'normal')
      .attr('fill', d => special(d) ? COLORS.highlight : nb.has(d.word) ? '#666' : '#333')
      .attr('opacity', d => special(d) ? 0.7 : 1)
      .text(d => query.has(d.i) ? d.word.replace(/^​/, '') : d.word);

    // 1D labels stand vertically, alternating above and below the line along x.
    this._sel = {
      svg, circles: g.selectAll('circle.point'), qmarks: g.selectAll('text.query-marker'),
      labels: g.selectAll('text.word-label'), arrows: g.selectAll('line.arrow'),
      heads: g.selectAll('path.arrow-head'), nlinks: g.selectAll('line.neighbor-link'),
      cross: g.selectAll('line.cross-group'), H: null,
    };
    this._labelSide = new Array(n);
    pointData.slice().sort((a, b) => this._L.P1[a.i][0] - this._L.P1[b.i][0])
      .forEach((d, k) => { this._labelSide[d.i] = k % 2 === 0 ? -1 : 1; });
    this._measureLabels();
    return svg;
  }

  // Label widths, for placing the vertical 1D labels (re-measured once web fonts load).
  _measureLabels() {
    const lens = new Array(this.words.length).fill(0);
    d3.select(this._el()).selectAll('svg.plot text.word-label')
      .each(function(d) { lens[d.i] = this.getComputedTextLength(); });
    this._labelLen = lens;
  }

  // Draw the plot for a view state (any blend of 1D, 2D and 3D).
  _draw(st) {
    const S = this._sel;
    if (!S) return;
    const G = this._geom;
    // Only touch the height when it changes: setting it relayouts the page below.
    if (S.H !== st.H) { S.svg.attr('height', st.H); S.H = st.H; }
    const p = viewProject(this._L, st);
    this._currentCoords2D = p;
    const X = i => G.innerW / 2 + p[i][0] * G.px;
    const Y = i => st.H / 2 - p[i][1] * G.px;

    S.circles.attr('cx', d => X(d.i)).attr('cy', d => Y(d.i));
    S.qmarks.attr('x', d => X(d.i)).attr('y', d => Y(d.i));
    // Labels: upright 8px above the point in 2D/3D; in 1D turned vertical, starting 6px
    // above or below the line. Blends in between turn them smoothly.
    const up = 1 - st.w1, side = this._labelSide, len = this._labelLen;
    S.labels.attr('transform', d =>
      `translate(${X(d.i)},${Y(d.i)}) rotate(${(1 - up) * 90 * side[d.i]}) ` +
      `translate(${(1 - up) * (6 + len[d.i] / 2)},${-8 * up})`);
    S.arrows
      .attr('x1', d => X(d.from)).attr('y1', d => Y(d.from))
      .attr('x2', d => X(d.to)).attr('y2', d => Y(d.to));
    S.heads
      .attr('d', d => arrowHeadPath(X(d.from), Y(d.from), X(d.to), Y(d.to), d.thick ? 2 : 1.5));
    S.nlinks
      .attr('x1', d => X(d.parent)).attr('y1', d => Y(d.parent))
      .attr('x2', d => X(d.child)).attr('y2', d => Y(d.child));
    S.cross
      .attr('x1', d => X(d[0])).attr('y1', d => Y(d[0]))
      .attr('x2', d => X(d[1])).attr('y2', d => Y(d[1]));
  }

  // Switch dimension; every pair of 1D, 2D, 3D animates (see DimView).
  _switchDims(newDims) {
    if (newDims === this.dims) return;
    const token = ++this._switchToken;
    const svg = d3.select(this._el()).select('svg.plot');
    if (newDims === 3) detachPanZoom(svg);
    this.dims = newDims;
    if (this.eigenSelector) this.eigenSelector.update(newDims, this.mdsData.variance[newDims]);
    this._measureLabels();
    this._view.go(newDims, this._geom, () => {
      if (token === this._switchToken && newDims !== 3) attachPanZoom(svg);
    });
  }

  _stopRotation() {
    this._view.stop();
  }

  // Click-to-expand is disabled. The infrastructure (neighborWords, neighborLinks,
  // styling in render2D/render1D/render3D, searchEmb) is still in place if we ever
  // want to re-enable it. To do so: return the handler below instead of null.
  // Known issues when it was active:
  //   - Expanded neighbors were often off-topic (see TODO: adaptive neighbor count)
  //   - MDS quality degraded as words were added (variance spread across more dims)
  //   - Labels got crowded fast
  _makeOnClick() {
    return null;
    /* Original handler for reference:
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
    */
  }

  // Build and draw the plot in its current dimension (no animation).
  render() {
    this._view.stop();
    const el = this._el();
    this._L = this._layouts();
    this._geom = this._geometry(el);
    const svg = this._build();
    this._view.attach(svg.node());
    this._view.set(this.dims, this._geom);
    if (this.dims !== 3) attachPanZoom(svg);
    if (this.eigenSelector) this.eigenSelector.update(this.dims, this.mdsData.variance[this.dims]);
    // Vertical 1D labels are placed by width; re-measure once web fonts have loaded.
    if (document.fonts && !this._fontsHooked) {
      this._fontsHooked = true;
      document.fonts.ready.then(() => { this._measureLabels(); this._view.redraw(); });
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
  const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  // Render at the screen's pixel density so the hero is sharp on high-DPI displays.
  renderer.setPixelRatio(window.devicePixelRatio || 1);
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

  // Camera distance and label size. Wide canvases keep the original framing. On a
  // narrow canvas (phones) the default camera cuts off words at the sides, so find the
  // closest distance at which every word and its label stays in frame at every orbit
  // angle, and grow the labels with the distance so their on-screen size stays at
  // 0.8x the desktop size.
  const DEFAULT_DIST = Math.hypot(2, 1.3, 2);
  let camDist = DEFAULT_DIST, labelScale = 1;
  if (width < 600) {
    const k = 0.8 * (500 / height) / DEFAULT_DIST;   // label scale per unit of distance
    const toVec = c => new THREE.Vector3(c[0] * scale, c[1] * scale, c[2] * scale);
    const dots = wordData.flatMap(d => [toVec(d.origCoord), toVec(d.steeredCoord)]);
    // Each label's text half-width, as a fraction of its sprite's width (the sprite is
    // a 512px canvas with the word drawn centered in bold 44px).
    const measure = document.createElement('canvas').getContext('2d');
    measure.font = 'bold 44px sans-serif';
    const labels = wordData.map(d => ({ pos: toVec(d.steeredCoord), half: measure.measureText(d.word).width / 512 / 2 }));
    const elev = Math.atan2(1.3, Math.hypot(2, 2));
    const probe = new THREE.PerspectiveCamera(camera.fov, camera.aspect, 0.1, 100);
    const v = new THREE.Vector3(), right = new THREE.Vector3();
    const fits = (D) => {
      const ls = Math.max(1, k * D);
      for (let a = 0; a < 72; a++) {
        const th = (a / 72) * 2 * Math.PI;
        probe.position.set(D * Math.cos(elev) * Math.sin(th), D * Math.sin(elev), D * Math.cos(elev) * Math.cos(th));
        probe.lookAt(0, 0, 0);
        probe.updateMatrixWorld();
        right.set(Math.cos(th), 0, -Math.sin(th));
        for (const p of dots) {
          v.copy(p).project(probe);
          if (Math.abs(v.x) > 0.97 || Math.abs(v.y) > 0.97) return false;
        }
        // Both ends of each label's text, at the top of the label (it sits 0.14 * ls above its dot).
        for (const { pos, half } of labels) {
          for (const sgn of [1, -1]) {
            v.copy(pos).addScaledVector(right, sgn * half * 0.7 * ls);
            v.y += (0.14 + 0.06) * ls;
            v.project(probe);
            if (Math.abs(v.x) > 0.97 || Math.abs(v.y) > 0.97) return false;
          }
        }
      }
      return true;
    };
    if (!fits(30)) throw new Error(`renderHero3D: words don't fit a ${width}x${height} canvas`);
    let lo = DEFAULT_DIST, hi = 30;
    if (fits(lo)) hi = lo;
    for (let i = 0; i < 25 && hi - lo > 0.01; i++) {
      const mid = (lo + hi) / 2;
      if (fits(mid)) hi = mid; else lo = mid;
    }
    camDist = hi;
    labelScale = Math.max(1, k * camDist);
  }

  // Group colors
  const groupNames = [...new Set(wordData.map(d => d.group))];
  const groupPalette = ['#4a6a8a', '#5e8c61', '#8a6d94', '#b05c5c', '#6a9e9e', '#8b7355'];
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
    sprite.position.set(sx[0], sx[1] + 0.14 * labelScale, sx[2]);
    sprite.scale.set(0.7 * labelScale, 0.175 * labelScale, 1);
    scene.add(sprite);
  }

  const f = camDist / DEFAULT_DIST;
  camera.position.set(2 * f, 1.3 * f, 2 * f);
  camera.lookAt(0, 0, 0);

  // Render only while the canvas is on screen. OrbitControls (r128) turns and damps by a
  // fixed amount per frame, which spins twice as fast on a 120Hz screen and unevenly when
  // frames are uneven; scale both by the frame's duration (capped at 3 frames, so a stall
  // doesn't jump).
  let animId = null, tPrev = null;
  function animate(t) {
    animId = requestAnimationFrame(animate);
    const k = tPrev === null || t === undefined ? 1 : Math.min(3, (t - tPrev) / (1000 / 60));
    tPrev = t === undefined ? null : t;
    controls.autoRotateSpeed = 0.8 * k;
    controls.dampingFactor = 1 - Math.pow(1 - 0.05, k);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
  const io = new IntersectionObserver((es) => {
    const e = es[es.length - 1];  // entries arrive oldest first: the last is current
    if (e.isIntersecting && animId === null) animate();
    else if (!e.isIntersecting && animId !== null) { cancelAnimationFrame(animId); animId = null; tPrev = null; }
  }, { rootMargin: '100px' });
  io.observe(renderer.domElement);

  return { scene, camera, renderer, controls, el,
    destroy() { io.disconnect(); cancelAnimationFrame(animId); renderer.dispose(); clearTimeout(pauseTimeout); }
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
    this.startSteered = config.startSteered || false;
    this.steered = this.startSteered;

    // Compute MDS at all dimensions from the joint distance matrix
    const nn = 2 * this.n;
    this.mdsData = computeAllMDSFromMatrix(config.distMatrix, nn);

    // Assign coords to wordData at each dimension level
    this._assignCoords();

    this._view = new DimView(st => this._draw(st));
    this._switchToken = 0;
    this._steerT = this.steered ? 1 : 0;  // 0 = original positions, 1 = steered

    // Create eigen selector
    if (this.eigenEl) {
      this.eigenEl.innerHTML = '';
      this.eigenSelector = createEigenSelector(
        this.eigenEl,
        this.mdsData.eigenvalues,
        this.dims,
        (newDims) => this._switchDims(newDims)
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

  // The 1D, 2D and 3D layouts of all 2n points (steered 0..n-1, original n..2n-1) as
  // 3-vectors in one unit, the 2D layout's radius (see DimView).
  _layouts() {
    const raw2 = this.mdsData.coords[2];
    const c = Math.max(...raw2.map(([x, y]) => Math.hypot(x, y))) || 1;
    return {
      P1: this.mdsData.coords[1].map(([x]) => [x / c, 0, 0]),
      P2: raw2.map(([x, y]) => [x / c, y / c, 0]),
      P3: this.mdsData.coords[3].map(([x, y, z]) => [x / c, y / c, z / c]),
    };
  }

  // Canvas geometry shared by all views: one scale fitted to the 2D layout's extent
  // (8% padding, equal aspect), its center, canvas heights for 1D and 2D/3D, and the
  // 3D zoom-out k3.
  _geometry() {
    const L = this._L;
    const plotW = getResponsiveWidth(this.plotEl);
    const innerW = plotW - 20;  // 10px side margins
    const xs = [...L.P2.map(p => p[0]), ...L.P1.map(p => p[0])], ys = L.P2.map(p => p[1]);
    const pad = 0.08;
    const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
    const spanX = ((xMax - xMin) || 0.1) * (1 + 2 * pad), spanY = ((yMax - yMin) || 0.1) * (1 + 2 * pad);
    const H2 = Math.max(Math.round(plotW * 0.3), Math.min(Math.round(plotW * 0.65), Math.round(innerW * spanY / spanX)));
    const H1 = Math.min(200, Math.round(plotW * 0.35));
    const px = Math.min(innerW / spanX, H2 / spanY);
    const xmid = (xMin + xMax) / 2, ymid = (yMin + yMax) / 2;
    const k3 = spinFitZoom(L.P3, innerW / (2 * px) - Math.abs(xmid), H2 / (2 * px) - Math.abs(ymid));
    return { plotW, innerW, H1, H2, px, xmid, ymid, k3 };
  }

  // Build the SVG and the Steer/Reset controls once; _draw positions everything.
  _build() {
    const el = this.plotEl;
    const G = this._geom;
    const wordData = this.wordData, arrows = this.arrows;
    const groupNames = [...new Set(wordData.map(d => d.group))];
    const groupPalette = ['#4a6a8a', '#5e8c61', '#8a6d94', '#b05c5c', '#6a9e9e', '#8b7355'];
    const groupColor = {};
    groupNames.forEach((g, i) => { groupColor[g] = groupPalette[i % groupPalette.length]; });

    d3.select(el).selectAll('svg.plot').remove();
    d3.select(el).selectAll('.steer-controls').remove();
    const svg = d3.select(el).append('svg')
      .attr('class', 'plot').attr('width', G.plotW).attr('height', G.H2)
      .style('cursor', 'grab');
    svg.on('mousedown.cursor', () => svg.style('cursor', 'grabbing'));
    svg.on('mouseup.cursor', () => svg.style('cursor', 'grab'));
    const g = svg.append('g').attr('class', 'zoom-container')
      .append('g').attr('class', 'main').attr('transform', 'translate(10,0)');

    const on = this.steered;
    g.selectAll('line.trail').data(wordData).enter().append('line')
      .attr('class', 'trail')
      .attr('stroke', d => groupColor[d.group]).attr('stroke-width', 1.5)
      .attr('opacity', on ? 0.4 : 0);
    g.selectAll('circle.ghost').data(wordData).enter().append('circle')
      .attr('class', 'ghost')
      .attr('r', 3).attr('fill', d => groupColor[d.group])
      .attr('opacity', on ? 0.3 : 0);
    g.selectAll('line.ghost-arrow').data(arrows).enter().append('line')
      .attr('class', 'ghost-arrow')
      .attr('stroke', '#ccc').attr('stroke-width', 1).attr('stroke-dasharray', '4,3')
      .attr('opacity', on ? 0.3 : 0);
    g.selectAll('line.active-arrow').data(arrows).enter().append('line')
      .attr('class', 'active-arrow')
      .attr('stroke', COLORS.arrow).attr('stroke-width', 1.5);
    g.selectAll('circle.word').data(wordData).enter().append('circle')
      .attr('class', 'word')
      .attr('r', 4).attr('fill', d => groupColor[d.group]);
    g.selectAll('text.word-label').data(wordData).enter().append('text')
      .attr('class', 'word-label')
      .attr('text-anchor', 'middle').attr('font-size', '11px').attr('fill', '#333')
      .text(d => d.word);
    this._sel = {
      svg, dots: g.selectAll('circle.word'), labels: g.selectAll('text.word-label'),
      ghosts: g.selectAll('circle.ghost'), trails: g.selectAll('line.trail'),
      ghostArrows: g.selectAll('line.ghost-arrow'), arrows: g.selectAll('line.active-arrow'), H: null,
    };

    // Steer/Reset controls (hidden when startSteered: the plot is already in its final state)
    const controls = d3.select(el).append('div').attr('class', 'steer-controls')
      .style('margin-top', '8px').style('display', this.startSteered ? 'none' : 'flex')
      .style('gap', '8px').style('align-items', 'center');
    const toggleBtn = controls.append('button')
      .style('border', 'none').style('border-radius', '4px').style('padding', '5px 14px')
      .style('font-size', '13px').style('cursor', 'pointer');
    const statusText = controls.append('span').style('font-size', '12px').style('color', '#999');
    const showButton = () => {
      toggleBtn.text(this.steered ? 'Reset' : '▶ Steer')
        .style('background', this.steered ? '#ddd' : COLORS.point)
        .style('color', this.steered ? '#333' : 'white');
    };
    showButton();
    statusText.text(this.steered ? 'Steered embeddings' : 'Original embeddings');
    toggleBtn.on('click', () => {
      this.steered = !this.steered;
      showButton();
      const dur = this.steered ? 1500 : 800;
      statusText.text(this.steered ? 'Steering...' : 'Original embeddings');
      g.selectAll('circle.ghost, line.ghost-arrow').transition().duration(300).attr('opacity', this.steered ? 0.3 : 0);
      g.selectAll('line.trail').transition().duration(dur).attr('opacity', this.steered ? 0.4 : 0);
      // Move the words by animating the blend between original and steered positions.
      const from = this._steerT, to = this.steered ? 1 : 0;
      if (this._steerTimer) this._steerTimer.stop();
      this._steerTimer = d3.timer((elapsed) => {
        const u = Math.min(1, elapsed / dur);
        this._steerT = from + (to - from) * smooth(u);
        if (!this._view.running) this._view.redraw();
        if (u >= 1) {
          this._steerTimer.stop();
          if (this.steered) statusText.text('Steered embeddings');
        }
      });
    });
    return svg;
  }

  // Draw the plot for a view state (any blend of 1D, 2D and 3D).
  _draw(st) {
    const S = this._sel;
    if (!S) return;
    const G = this._geom, n = this.n, t = this._steerT;
    // Only touch the height when it changes: setting it relayouts the page below.
    if (S.H !== st.H) { S.svg.attr('height', st.H); S.H = st.H; }
    const p = viewProject(this._L, st);
    const orig = i => p[n + i];
    const act = i => [p[n + i][0] + t * (p[i][0] - p[n + i][0]), p[n + i][1] + t * (p[i][1] - p[n + i][1])];
    // The 2D layout's center sits at the canvas center; in 1D the line is centered vertically.
    const X = q => G.innerW / 2 + (q[0] - G.xmid) * G.px;
    const Y = q => st.H / 2 - (q[1] - G.ymid * (1 - st.w1)) * G.px;
    S.dots.attr('cx', (d, i) => X(act(i))).attr('cy', (d, i) => Y(act(i)));
    S.labels.attr('x', (d, i) => X(act(i))).attr('y', (d, i) => Y(act(i)) - 8);
    S.ghosts.attr('cx', (d, i) => X(orig(i))).attr('cy', (d, i) => Y(orig(i)));
    S.trails
      .attr('x1', (d, i) => X(orig(i))).attr('y1', (d, i) => Y(orig(i)))
      .attr('x2', (d, i) => X(act(i))).attr('y2', (d, i) => Y(act(i)));
    S.ghostArrows
      .attr('x1', d => X(orig(d.from))).attr('y1', d => Y(orig(d.from)))
      .attr('x2', d => X(orig(d.to))).attr('y2', d => Y(orig(d.to)));
    S.arrows
      .attr('x1', d => X(act(d.from))).attr('y1', d => Y(act(d.from)))
      .attr('x2', d => X(act(d.to))).attr('y2', d => Y(act(d.to)));
  }

  // Switch dimension; every pair of 1D, 2D, 3D animates (see DimView).
  _switchDims(newDims) {
    if (newDims === this.dims) return;
    const token = ++this._switchToken;
    const svg = d3.select(this.plotEl).select('svg.plot');
    if (newDims === 3) detachPanZoom(svg);
    this.dims = newDims;
    if (this.eigenSelector) this.eigenSelector.update(newDims, this.mdsData.variance[newDims]);
    this._view.go(newDims, this._geom, () => {
      if (token === this._switchToken && newDims !== 3) attachPanZoom(svg);
    });
  }

  _stopRotation() {
    this._view.stop();
  }

  // Build and draw the plot in its current dimension (no animation).
  render() {
    this._view.stop();
    this._L = this._layouts();
    this._geom = this._geometry();
    const svg = this._build();
    this._view.attach(svg.node());
    this._view.set(this.dims, this._geom);
    if (this.dims !== 3) attachPanZoom(svg);
    if (this.eigenSelector) this.eigenSelector.update(this.dims, this.mdsData.variance[this.dims]);
  }
}

/**
 * 2D animated steering visualization (legacy stateless version).
 * Shows words moving from original to steered positions with fading trails.
 */
function renderSteering2D(container, wordData, options = {}) {
  const el = typeof container === 'string' ? document.getElementById(container) : container;
  const defaultW = getResponsiveWidth(el);
  const { arrows = [] } = options;
  const margin = { top: 0, right: 10, bottom: 0, left: 10 };
  const innerWRaw = defaultW - margin.left - margin.right;

  d3.select(el).selectAll('svg.plot').remove();
  d3.select(el).selectAll('.steer-controls').remove();

  const allX = wordData.flatMap(d => [d.origCoord[0], d.steeredCoord[0]]);
  const allY = wordData.flatMap(d => [d.origCoord[1], d.steeredCoord[1]]);
  const pad = 0.1;
  const xMin = Math.min(...allX), xMax = Math.max(...allX);
  const yMin = Math.min(...allY), yMax = Math.max(...allY);
  const xSpan = (xMax - xMin) || 0.2, ySpan = (yMax - yMin) || 0.2;
  const xPad = xSpan * pad, yPad = ySpan * pad;
  // Adaptive height based on data aspect ratio
  const dataW = xSpan + 2 * xPad, dataH = ySpan + 2 * yPad;
  const dataRatio = dataH / dataW;
  const rawH = Math.round(innerWRaw * dataRatio) + margin.top + margin.bottom;
  const minH = Math.round(defaultW * 0.3);
  const maxH = Math.round(defaultW * 0.65);
  const width = defaultW;
  const height = Math.max(minH, Math.min(maxH, rawH));
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;
  // Equal scaling: expand the smaller axis to match the aspect ratio
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
  const groupPalette = ['#4a6a8a', '#5e8c61', '#8a6d94', '#b05c5c', '#6a9e9e', '#8b7355'];
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

// ── Subspace animation ─────────────────────────────────────────────────
const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul3 = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const lerp2 = (a, b, t) => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];

// Everything the subspace animation draws, for word positions Q (3-vectors) and pairs
// { from, to }: the pair differences moved to the centroid, the direction (top eigenvector
// of sum_i d_i d_i^T, pointing along sum_i d_i), each word with its component along the
// direction removed, and what is perpendicular to the direction: a line along u (in the
// xy-plane) and the plane spanned by u and v. Throws where the direction's orientation or
// the perpendicular line is undefined.
function subspaceScene(Q, pairs) {
  const c = [0, 1, 2].map(j => d3.mean(Q, q => q[j]));
  const diffs = pairs.map(p => sub3(Q[p.to], Q[p.from]));
  const C = [0, 1, 2].map(a => [0, 1, 2].map(b => d3.sum(diffs, d => d[a] * d[b])));
  const { u: U } = sortedSVD(C);
  let dir = [U[0][0], U[1][0], U[2][0]];
  const along = dot3(dir, diffs.reduce(add3, [0, 0, 0]));
  if (along === 0) throw new Error('subspace animation: the pair differences sum to a vector perpendicular to the direction, so it has no orientation');
  if (along < 0) dir = mul3(dir, -1);
  const comp = Q.map(q => dot3(sub3(q, c), dir));
  const steered = Q.map((q, i) => sub3(q, mul3(dir, comp[i])));
  const h = Math.hypot(dir[0], dir[1]);
  if (h < 1e-9) throw new Error('subspace animation: the direction is vertical, so the perpendicular line is undefined');
  const u = [-dir[1] / h, dir[0] / h, 0];
  const v = [dir[1] * u[2] - dir[2] * u[1], dir[2] * u[0] - dir[0] * u[2], dir[0] * u[1] - dir[1] * u[0]];
  // The line and plane reach 20% past the furthest steered word.
  const eu = 1.2 * d3.max(steered, s => Math.abs(dot3(sub3(s, c), u)));
  const ev = 1.2 * d3.max(steered, s => Math.abs(dot3(sub3(s, c), v)));
  return {
    Q, c, steered,
    dirStart: add3(c, mul3(dir, d3.min(comp))), dirEnd: add3(c, mul3(dir, d3.max(comp))),
    diffEnds: diffs.map(d => add3(c, d)),
    axis: [sub3(c, mul3(u, eu)), add3(c, mul3(u, eu))],
    plane: [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(([a, b]) => add3(c, add3(mul3(u, a * eu), mul3(v, b * ev)))),
  };
}

/**
 * Step-by-step walkthrough of finding a direction from word pairs and projecting it out:
 *   0  arrows from each pair's first word to its second;
 *   1  the differences d_i moved to start at the centroid;
 *   2  the direction (top eigenvector of sum_i d_i d_i^T);
 *   3  each word's component along it (dashed), and what is perpendicular to it
 *      (a line in 2D, a plane in 3D);
 *   4  each word moved by minus that component, onto the line or plane.
 * It runs on the 2D or 3D MDS layout itself, not in 100 dimensions (the article says so).
 *
 * 2D and 3D are one scene (see DimView): the layout is w2*P2 + w3*P3, i.e. (x, y, w3*z),
 * and subspaceScene recomputes the direction, projections, line and plane from it every
 * frame, so switching dimension at any step, even mid-step, moves everything continuously.
 * The perpendicular line lies in the plane: it fades out as the plane fades in. Every step's
 * quantities (opacities, how far the arrows and words have moved) are eased over time, so a
 * click mid-step continues from where things are. 1D is disabled: projecting out the only
 * axis would put every word on one point.
 *
 * wordData: [{ word }]; options.pairs: [{ from, to }] indices into wordData;
 * options.mdsData: computeAllMDS output for these words; options.eigenContainer: element
 * (or id) for the dimension selector.
 */
class SubspaceAnimation {
  constructor(el, wordData, { pairs = [], directionLabel = 'direction', mdsData, eigenContainer = null }) {
    if (!mdsData) throw new Error('subspace animation: mdsData is required');
    this.el = el;
    this.words = wordData.map(d => d.word);
    this.pairs = pairs;
    this.directionLabel = directionLabel;
    this.mdsData = mdsData;
    this.dims = 2;
    this.step = 0;
    // Both layouts in the unit of the 2D layout's radius.
    const raw2 = mdsData.coords[2];
    const r = d3.max(raw2, ([x, y]) => Math.hypot(x, y)) || 1;
    this._P2 = raw2.map(([x, y]) => [x / r, y / r, 0]);
    this._P3 = mdsData.coords[3].map(([x, y, z]) => [x / r, y / r, z / r]);
    // Eased step quantities, all 0..1 (see _goTo).
    this._tw = {};
    for (const k of ['pairA', 'transA', 'transT', 'dirA', 'perpA', 'projT', 'projA', 'ghostA', 'steerT']) {
      this._tw[k] = { from: 0, to: 0, t0: 0, delay: 0, dur: 1 };
    }
    this._tw.pairA.to = this._tw.pairA.from = 1;
    this._raf = null;
    this._view = new DimView(st => this._draw(st));

    const eigenEl = typeof eigenContainer === 'string' ? document.getElementById(eigenContainer) : eigenContainer;
    if (eigenEl) {
      eigenEl.innerHTML = '';
      this._selector = createEigenSelector(eigenEl, mdsData.eigenvalues, this.dims, d => this._switchDims(d));
      d3.select(eigenEl).selectAll('g.eigen-col').filter((d, i) => i === 0)
        .style('opacity', 0.4).style('cursor', 'not-allowed');
      this._selector.update(this.dims, mdsData.variance[this.dims]);
    }

    this._geom = this._geometry();
    const svg = this._build();
    this._view.attach(svg.node());
    this._view.set(this.dims, this._geom);
  }

  _sceneAt(w2, w3) {
    return subspaceScene(this._P2.map((p2, i) => add3(mul3(p2, w2), mul3(this._P3[i], w3))), this.pairs);
  }

  // One scale for 2D and 3D, fitted to the words and arrows the 2D steps draw (12% padding,
  // equal aspect; the perpendicular line may run off the edge), the canvas height, and the
  // 3D zoom-out k3 that keeps the spinning scene, plane included, inside.
  _geometry() {
    const plotW = getResponsiveWidth(this.el), innerW = plotW - 20;  // 10px side margins
    const s2 = this._sceneAt(1, 0);
    const pts = [...s2.Q, ...s2.steered, s2.dirStart, s2.dirEnd, ...s2.diffEnds];
    const [xMin, xMax] = d3.extent(pts, p => p[0]), [yMin, yMax] = d3.extent(pts, p => p[1]);
    const pad = 0.12;
    const spanX = ((xMax - xMin) || 0.2) * (1 + 2 * pad), spanY = ((yMax - yMin) || 0.2) * (1 + 2 * pad);
    const H = Math.max(Math.round(plotW * 0.35), Math.min(Math.round(plotW * 0.65), Math.round(innerW * spanY / spanX)));
    const px = Math.min(innerW / spanX, H / spanY);
    const xmid = (xMin + xMax) / 2, ymid = (yMin + yMax) / 2;
    const s3 = this._sceneAt(0, 1);
    const k3 = spinFitZoom([...s3.Q, ...s3.steered, s3.dirStart, s3.dirEnd, ...s3.diffEnds, ...s3.plane],
      innerW / (2 * px) - Math.abs(xmid), H / (2 * px) - Math.abs(ymid));
    return { plotW, innerW, H, H1: H, H2: H, px, xmid, ymid, k3 };
  }

  _build() {
    const el = this.el, G = this._geom, pairs = this.pairs, words = this.words;
    d3.select(el).selectAll('svg.plot, .anim-desc, .anim-controls').remove();
    const svg = d3.select(el).append('svg').attr('class', 'plot')
      .attr('width', G.plotW).attr('height', G.H);
    const g = svg.append('g').attr('transform', 'translate(10,0)');
    // Arrow sets are drawn opaque inside a group that carries the transparency, so a line
    // doesn't show darker where it runs under its own head.
    const plane = g.append('polygon').attr('class', 'perp-plane')
      .attr('fill', '#999').attr('fill-opacity', 0.08)
      .attr('stroke', '#999').attr('stroke-width', 0.5).attr('stroke-dasharray', '4,3');
    const axis = g.append('line').attr('class', 'perp-axis')
      .attr('stroke', '#999').attr('stroke-width', 1).attr('stroke-dasharray', '6,4');
    const pairG = g.append('g').attr('class', 'pair-arrows');
    pairG.selectAll('line').data(pairs).enter().append('line')
      .attr('stroke', COLORS.point).attr('stroke-width', 1.5);
    pairG.selectAll('path').data(pairs).enter().append('path').attr('fill', COLORS.point);
    const transG = g.append('g').attr('class', 'trans-arrows');
    transG.selectAll('line').data(pairs).enter().append('line');
    transG.selectAll('path').data(pairs).enter().append('path');
    const dirG = g.append('g').attr('class', 'dir-arrow');
    const dirLine = dirG.append('line')
      .attr('stroke', COLORS.direction).attr('stroke-width', 2.5).attr('stroke-dasharray', '8,4');
    const dirHead = dirG.append('path').attr('fill', COLORS.direction);
    const projG = g.append('g').attr('class', 'proj-lines');
    projG.selectAll('line').data(words).enter().append('line')
      .attr('stroke', COLORS.direction).attr('stroke-width', 1).attr('stroke-dasharray', '3,2');
    const ghostG = g.append('g').attr('class', 'ghosts');
    ghostG.selectAll('circle').data(words).enter().append('circle')
      .attr('class', 'ghost').attr('r', 3).attr('fill', COLORS.point);
    g.selectAll('circle.word-dot').data(words).enter().append('circle')
      .attr('class', 'word-dot').attr('r', 3.5).attr('fill', COLORS.point);
    g.selectAll('text.word-label').data(words).enter().append('text')
      .attr('class', 'word-label')
      .attr('text-anchor', 'middle').attr('font-size', '11px').attr('fill', '#333')
      .text(d => d);
    const dirLabel = g.append('text').attr('class', 'dir-label')
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px').attr('font-weight', 'bold').attr('font-style', 'italic')
      .attr('fill', COLORS.direction).text(this.directionLabel);

    // Step text: all steps share one grid cell, so the box is as tall as the longest and the
    // page below doesn't shift between steps; the current one fades in.
    const km = (tex) => katex.renderToString(tex, { throwOnError: false });
    const stepDescs = [
      `Each arrow connects a word pair — e.g. ${km('\\overrightarrow{\\text{woman}}')} to ${km('\\overrightarrow{\\text{man}}')}.`,
      `Translate each pair’s difference vector ${km('d_i = \\overrightarrow{w}_i^+ - \\overrightarrow{w}_i^-')} to a common origin. They point in roughly the same direction.`,
      `The top eigenvector of ${km('C = \\tfrac{1}{2}\\sum d_i\\,d_i^T')} gives the direction that best explains these differences.`,
      `Each word’s component along the direction (its “gender component” ${km('\\overrightarrow{w}_{\\mathcal{B}}')}) is shown as a dashed line.`,
      `Project out: ${km('\\overrightarrow{w}_{\\text{steered}} = \\overrightarrow{w} - \\overrightarrow{w}_{\\mathcal{B}}')}`,
    ];
    const desc = d3.select(el).append('div').attr('class', 'anim-desc')
      .style('display', 'grid').style('font-size', '13px').style('color', '#555')
      .style('line-height', '1.3').style('padding', '2px 14px 0');
    const descs = desc.selectAll('div').data(stepDescs).enter().append('div')
      .style('grid-area', '1 / 1').style('transition', 'opacity 0.3s').html(d => d);

    const ctrl = d3.select(el).append('div').attr('class', 'anim-controls')
      .style('margin-top', '6px').style('display', 'flex').style('gap', '8px').style('align-items', 'center');
    const btn = ctrl.append('button')
      .style('border', 'none').style('border-radius', '4px').style('padding', '5px 14px')
      .style('font-size', '13px').style('cursor', 'pointer');
    const counter = ctrl.append('span').style('font-size', '12px').style('color', '#999');
    btn.on('click', () => this._goTo(this.step >= 4 ? 0 : this.step + 1));

    this._sel = {
      svg, plane, axis, pairG, transG, dirG, dirLine, dirHead, dirLabel, projG, ghostG,
      pairLines: pairG.selectAll('line'), pairHeads: pairG.selectAll('path'),
      transLines: transG.selectAll('line'), transHeads: transG.selectAll('path'),
      projLines: projG.selectAll('line'), ghosts: ghostG.selectAll('circle'),
      dots: g.selectAll('circle.word-dot'), labels: g.selectAll('text.word-label'),
      descs, btn, counter,
    };
    this._showStep();
    return svg;
  }

  _showStep() {
    const S = this._sel, s = this.step;
    S.descs.style('opacity', (d, i) => i === s ? 1 : 0).attr('aria-hidden', (d, i) => i === s ? null : 'true');
    S.counter.text(`Step ${s + 1} of 5`);
    const last = s === 4;
    S.btn.text(last ? 'Reset' : 'Next ▶')
      .style('background', last ? '#ddd' : COLORS.point).style('color', last ? '#333' : 'white');
  }

  // Value of an eased step quantity now.
  _val(k, now) {
    const t = this._tw[k];
    const u = Math.min(1, Math.max(0, (now - t.t0 - t.delay) / t.dur));
    return t.from + (t.to - t.from) * smooth(u);
  }

  // Ease quantity k from where it is now to `to`.
  _tween(k, to, dur = 800, delay = 0) {
    const now = performance.now();
    this._tw[k] = { from: this._val(k, now), to, t0: now, delay, dur };
  }

  _goTo(s) {
    this.step = s;
    this._showStep();
    const T = (k, to, dur, delay) => this._tween(k, to, dur, delay);
    if (s === 0) {
      for (const k of ['transA', 'transT', 'dirA', 'perpA', 'projT', 'projA', 'ghostA', 'steerT']) T(k, 0);
      T('pairA', 1);
    } else if (s === 1) {
      T('pairA', 0.3); T('transA', 1, 200); T('transT', 1, 1200);
    } else if (s === 2) {
      T('transA', 0.35); T('dirA', 1);
    } else if (s === 3) {
      T('transA', 0); T('perpA', 1); T('projT', 1); T('projA', 1);
    } else {
      T('ghostA', 1, 300); T('pairA', 0); T('steerT', 1, 1200);
    }
    this._kick();
  }

  // Redraw every frame until the step quantities settle (the 3D spin redraws on its own).
  _kick() {
    if (this._raf) return;
    const loop = () => {
      this._raf = null;
      const now = performance.now();
      const busy = Object.values(this._tw).some(t => now < t.t0 + t.delay + t.dur);
      if (!this._view.running || !busy) this._view.redraw();
      if (busy) this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }

  _draw(st) {
    const S = this._sel;
    if (!S) return;
    const G = this._geom, now = performance.now(), a = k => this._val(k, now);
    const sc = this._sceneAt(st.w2, st.w3);
    const P = p => {
      const [x, y] = project3Dto2D([mul3(p, st.k)], st.angle, st.tilt)[0];
      return [G.innerW / 2 + (x - G.xmid) * G.px, G.H / 2 - (y - G.ymid) * G.px];
    };
    const word = sc.Q.map(P), steered = sc.steered.map(P);
    const steerT = a('steerT'), transT = a('transT'), projT = a('projT');
    const at = word.map((w, i) => lerp2(w, steered[i], steerT));

    S.dots.attr('cx', (d, i) => at[i][0]).attr('cy', (d, i) => at[i][1]);
    S.labels.attr('x', (d, i) => at[i][0]).attr('y', (d, i) => at[i][1] - 8);
    S.ghosts.attr('cx', (d, i) => word[i][0]).attr('cy', (d, i) => word[i][1]);
    S.ghostG.attr('opacity', 0.4 * a('ghostA'));

    S.pairLines
      .attr('x1', d => word[d.from][0]).attr('y1', d => word[d.from][1])
      .attr('x2', d => word[d.to][0]).attr('y2', d => word[d.to][1]);
    S.pairHeads.attr('d', d => arrowHeadPath(...word[d.from], ...word[d.to], 1.5));
    S.pairG.attr('opacity', 0.45 * a('pairA'));

    // Differences slide from their pair to the centroid, turning from green to the arrow color.
    const c = P(sc.c), ends = sc.diffEnds.map(P);
    const t0 = this.pairs.map(d => lerp2(word[d.from], c, transT));
    const t1 = this.pairs.map((d, i) => lerp2(word[d.to], ends[i], transT));
    const transColor = d3.interpolateRgb(COLORS.highlight, COLORS.point)(transT), transW = 2 - 0.5 * transT;
    S.transLines.attr('stroke', transColor).attr('stroke-width', transW)
      .attr('x1', (d, i) => t0[i][0]).attr('y1', (d, i) => t0[i][1])
      .attr('x2', (d, i) => t1[i][0]).attr('y2', (d, i) => t1[i][1]);
    S.transHeads.attr('fill', transColor).attr('d', (d, i) => arrowHeadPath(...t0[i], ...t1[i], transW));
    S.transG.attr('opacity', a('transA') * (1 - 0.55 * transT));

    const ds = P(sc.dirStart), de = P(sc.dirEnd);
    S.dirLine.attr('x1', ds[0]).attr('y1', ds[1]).attr('x2', de[0]).attr('y2', de[1]);
    S.dirHead.attr('d', arrowHeadPath(...ds, ...de, 2.5));
    S.dirG.attr('opacity', 0.9 * a('dirA'));
    S.dirLabel.attr('x', de[0]).attr('y', de[1] - 10).attr('opacity', a('dirA'));

    // Each word's component along the direction grows from the word to its steered position.
    S.projLines
      .attr('x1', (d, i) => word[i][0]).attr('y1', (d, i) => word[i][1])
      .attr('x2', (d, i) => lerp2(word[i], steered[i], projT)[0])
      .attr('y2', (d, i) => lerp2(word[i], steered[i], projT)[1]);
    S.projG.attr('opacity', 0.6 * a('projA'));

    const perpA = a('perpA'), ax = sc.axis.map(P);
    S.axis.attr('x1', ax[0][0]).attr('y1', ax[0][1]).attr('x2', ax[1][0]).attr('y2', ax[1][1])
      .attr('opacity', 0.5 * perpA * (1 - st.w3));
    S.plane.attr('points', sc.plane.map(P).map(q => q.join(',')).join(' '))
      .attr('opacity', perpA * st.w3);
  }

  _switchDims(newDims) {
    if (newDims === 1 || newDims === this.dims) {  // 1D is disabled
      if (this._selector) this._selector.update(this.dims, this.mdsData.variance[this.dims]);
      return;
    }
    this.dims = newDims;
    if (this._selector) this._selector.update(newDims, this.mdsData.variance[newDims]);
    this._sel.svg.style('cursor', newDims === 3 ? 'grab' : null);
    this._view.go(newDims, this._geom);
  }
}

function renderSubspaceAnimation(container, wordData, options = {}) {
  const el = typeof container === 'string' ? document.getElementById(container) : container;
  if (!el) return null;
  return new SubspaceAnimation(el, wordData, options);
}


export { EmbeddingViz, SteeringViz, computeAllMDS, computeAllMDSFromMatrix, render2D, render3D, render1D, renderHero3D, renderSteering2D, renderSubspaceAnimation };
