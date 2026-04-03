/**
 * UI smoke tests for the Steering Word Embeddings article.
 *
 * Usage:
 *   1. Start server: python3 ~/projects/blog/main/serve.py article/ --port 8768
 *   2. Run tests:    node article/test.mjs [URL]
 *
 * Requires puppeteer: npm install puppeteer (or use /tmp/node_modules)
 */

import puppeteer from '/tmp/node_modules/puppeteer/lib/esm/puppeteer/puppeteer.js';

const URL = process.argv[2] || 'http://localhost:8768/index.html';
const WAIT_MS = 60000;  // time for 50K vectors to load + compute

let browser, page;
const results = [];

function test(name, passed, detail = '') {
  results.push({ name, passed, detail });
  console.log(`  ${passed ? '✓' : '✗'} ${name}${detail ? ': ' + detail : ''}`);
}

async function setup() {
  browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox'] });
  page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });

  const jsErrors = [];
  page.on('pageerror', err => {
    jsErrors.push(err.message);
  });

  await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await new Promise(r => setTimeout(r, WAIT_MS));

  return jsErrors;
}

async function testNoJSErrors(jsErrors) {
  test('No JS errors', jsErrors.length === 0,
    jsErrors.length ? jsErrors.join('; ').substring(0, 200) : '');
}

async function testNoDistill() {
  const hasDistill = await page.evaluate(() =>
    !!document.querySelector('dt-article, dt-banner, dt-cite, dt-appendix'));
  test('No Distill elements remain', !hasDistill);
}

async function testHeroVisualization() {
  const has = await page.evaluate(() => !!document.querySelector('#hero-viz canvas'));
  test('Hero 3D canvas exists', has);

  const border = await page.evaluate(() =>
    document.querySelector('#hero-viz canvas')?.style.border || '');
  test('Hero canvas has border', border.includes('solid'));

  const cursor = await page.evaluate(() =>
    document.querySelector('#hero-viz canvas')?.style.cursor || '');
  test('Hero canvas has grab cursor', cursor === 'grab');
}

async function testHeroLegend() {
  const legend = await page.evaluate(() => {
    const el = document.querySelector('.margin-note') || document.querySelector('#hero-viz + p');
    return el?.innerHTML || '';
  });
  test('Hero legend has steel blue color', legend.includes('#4a6a8a'));
  test('Hero legend has green color', legend.includes('#5e8c61'));
}

async function testPlotCount() {
  const count = await page.evaluate(() => document.querySelectorAll('svg.plot').length);
  test('At least 7 SVG plots rendered', count >= 7, `got ${count}`);
}

async function testAlignment() {
  const m = await page.evaluate(() => {
    const p = document.querySelector('.entry-content > p') || document.querySelector('article p');
    const svg = document.querySelector('svg.plot');
    return {
      text: Math.round(p?.getBoundingClientRect().left),
      svg: Math.round(svg?.getBoundingClientRect().left),
    };
  });
  // SVG is indented 48px from text (margin-left on .plot-container for eigenvalue bars)
  test('SVG aligns with text', Math.abs(m.text + 48 - m.svg) < 5,
    `text=${m.text} svg=${m.svg}`);
}

async function testMathRendering() {
  const count = await page.evaluate(() => document.querySelectorAll('.katex').length);
  test('KaTeX equations rendered', count >= 4, `got ${count}`);
}

async function testAnalogyInput() {
  const hasButton = await page.evaluate(() => !!document.getElementById('analogy-go'));
  test('Analogy button exists', hasButton);

  // Click it and check result
  await page.click('#analogy-go');
  await new Promise(r => setTimeout(r, 3000));
  const result = await page.evaluate(() =>
    document.getElementById('analogy-result')?.textContent || '');
  test('Analogy produces result', result.includes('queen') || result.includes('is to'),
    result.substring(0, 80));
}

async function testSteeringAnimation() {
  const has = await page.evaluate(() => ({
    plot: !!document.querySelector('#plot-gendered-steer svg.plot'),
    button: !!document.querySelector('.steer-controls button'),
  }));
  test('Steering plot exists', has.plot);
  test('Steer button exists', has.button);
}

async function testClickToExpand() {
  // Use numbers plot (simpler, fewer words) for reliable testing
  const plotSel = '#plot-numbers-digits';

  const before = await page.evaluate((sel) => {
    const labels = document.querySelectorAll(`${sel} svg.plot text.word-label`);
    return { count: labels.length, words: [...labels].map(l => l.textContent) };
  }, plotSel);
  test('Click-to-expand: plot has words before click', before.count > 0,
    `${before.count} words`);

  // Click a word
  const word = await page.$(`${plotSel} svg.plot text.word-label`);
  if (word) {
    await word.click();
    await new Promise(r => setTimeout(r, 2000));
  }

  const after = await page.evaluate((sel) => {
    const labels = document.querySelectorAll(`${sel} svg.plot text.word-label`);
    return { count: labels.length, words: [...labels].map(l => l.textContent) };
  }, plotSel);
  test('Click-to-expand: word count increased', after.count > before.count,
    `${before.count} → ${after.count}`);

  // Check that neighbor links (dashed lines) were added
  const neighborLinks = await page.evaluate((sel) =>
    document.querySelectorAll(`${sel} svg.plot line.neighbor-link`).length, plotSel);
  test('Click-to-expand: neighbor links drawn', neighborLinks > 0,
    `${neighborLinks} links`);

  // Click a second word (a neighbor) to verify recursive expansion
  const neighborIdx = await page.evaluate((sel) => {
    const labels = [...document.querySelectorAll(`${sel} svg.plot text.word-label`)];
    // Find first gray-colored label (a neighbor, not an original word)
    return labels.findIndex(l => l.getAttribute('fill')?.includes('102'));
  }, plotSel);
  const words2 = await page.$$(`${plotSel} svg.plot text.word-label`);
  if (neighborIdx >= 0 && neighborIdx < words2.length) {
    await words2[neighborIdx].click();
    await new Promise(r => setTimeout(r, 2000));
  }
  const afterSecond = await page.evaluate((sel) =>
    document.querySelectorAll(`${sel} svg.plot text.word-label`).length, plotSel);
  test('Click-to-expand: recursive expansion works', afterSecond > after.count,
    `${after.count} → ${afterSecond}`);

  // Verify original words are still present after expansion
  const currentWords = await page.evaluate((sel) =>
    [...document.querySelectorAll(`${sel} svg.plot text.word-label`)].map(l => l.textContent), plotSel);
  const origStillPresent = before.words.every(w => currentWords.includes(w));
  test('Click-to-expand: original words preserved', origStillPresent);
}

async function testNeighborStylingAcrossDimensions() {
  // Expand on numbers-digits (already has neighbors from earlier test)
  // Check 2D neighbor styling
  const style2d = await page.evaluate(() => {
    const points = [...document.querySelectorAll('#plot-numbers-digits svg.plot circle.point')];
    const fills = points.map(p => p.getAttribute('fill'));
    const uniqueFills = [...new Set(fills)];
    return { uniqueFills, hasGray: fills.some(f => f === 'rgb(153, 153, 153)' || f === '#999') };
  });
  test('Neighbor styling 2D: neighbors have different color', style2d.hasGray,
    `fills: ${style2d.uniqueFills.join(', ')}`);

  // Check 2D neighbor links
  const links2d = await page.evaluate(() =>
    document.querySelectorAll('#plot-numbers-digits svg.plot line.neighbor-link').length);
  test('Neighbor styling 2D: dashed links present', links2d > 0, `${links2d} links`);

  // Switch to 1D
  await page.evaluate(() => {
    const bars = document.querySelectorAll('#eigen-numbers-digits svg g.eigen-col');
    if (bars.length >= 1) bars[0].dispatchEvent(new Event('click'));
  });
  await new Promise(r => setTimeout(r, 2000));

  const style1d = await page.evaluate(() => {
    const circles = [...document.querySelectorAll('#plot-numbers-digits svg.plot circle')];
    const fills = circles.map(c => c.getAttribute('fill'));
    const hasGray = fills.some(f => f === '#999' || f === 'rgb(153, 153, 153)');
    const links = document.querySelectorAll('#plot-numbers-digits svg.plot line.neighbor-link').length;
    return { hasGray, linkCount: links };
  });
  test('Neighbor styling 1D: neighbors have different color', style1d.hasGray);
  test('Neighbor styling 1D: dashed links present', style1d.linkCount > 0,
    `${style1d.linkCount} links`);

  // Switch back to 2D for subsequent tests
  await page.evaluate(() => {
    const bars = document.querySelectorAll('#eigen-numbers-digits svg g.eigen-col');
    if (bars.length >= 2) bars[1].dispatchEvent(new Event('click'));
  });
  await new Promise(r => setTimeout(r, 2000));
}

async function testClickToExpand3D() {
  // Switch numbers-digits to 3D (now rendered as projected SVG, not canvas)
  await page.evaluate(() => {
    const bars = document.querySelectorAll('#eigen-numbers-digits svg g.eigen-col');
    if (bars.length >= 3) bars[2].dispatchEvent(new Event('click'));
  });
  await new Promise(r => setTimeout(r, 3000));

  // 3D is now D3 SVG with projected rotation — verify SVG still exists
  const has3dSvg = await page.evaluate(() =>
    !!document.querySelector('#plot-numbers-digits svg.plot'));
  test('Click-to-expand 3D: SVG exists (projected 3D)', has3dSvg);

  // Click a word to expand in projected 3D view
  const beforeCount = await page.evaluate(() =>
    document.querySelectorAll('#plot-numbers-digits svg.plot text.word-label').length);
  const word = await page.$('#plot-numbers-digits svg.plot text.word-label');
  if (word) await word.click();
  await new Promise(r => setTimeout(r, 2000));
  const afterCount = await page.evaluate(() =>
    document.querySelectorAll('#plot-numbers-digits svg.plot text.word-label').length);
  test('Click-to-expand 3D: neighbors added', afterCount > beforeCount,
    `${beforeCount} → ${afterCount}`);

  // Switch back to 2D for subsequent tests
  await page.evaluate(() => {
    const bars = document.querySelectorAll('#eigen-numbers-digits svg g.eigen-col');
    if (bars.length >= 2) bars[1].dispatchEvent(new Event('click'));
  });
  await new Promise(r => setTimeout(r, 2000));
}

async function testClickToExpand1D() {
  // Switch numbers-words to 1D
  await page.evaluate(() => {
    const bars = document.querySelectorAll('#eigen-numbers-words svg g.eigen-col');
    if (bars.length >= 1) bars[0].dispatchEvent(new Event('click'));
  });
  await new Promise(r => setTimeout(r, 2000));

  const before = await page.evaluate(() =>
    document.querySelectorAll('#plot-numbers-words svg.plot text.word-label').length);

  // Click a word in 1D
  const word = await page.$('#plot-numbers-words svg.plot text.word-label');
  if (word) {
    await word.click();
    await new Promise(r => setTimeout(r, 2000));
  }

  const after = await page.evaluate(() =>
    document.querySelectorAll('#plot-numbers-words svg.plot text.word-label').length);
  test('Click-to-expand 1D: word count increased', after > before,
    `${before} → ${after}`);

  // Switch back to 2D
  await page.evaluate(() => {
    const bars = document.querySelectorAll('#eigen-numbers-words svg g.eigen-col');
    if (bars.length >= 2) bars[1].dispatchEvent(new Event('click'));
  });
  await new Promise(r => setTimeout(r, 2000));
}

async function testDimensionTransitionAnimation() {
  // Start in 2D, switch to 1D, verify points animate (SVG stays, positions change)
  const plotSel = '#plot-numbers-words';

  // Get initial point positions in 2D
  const before2D = await page.evaluate((sel) => {
    const points = [...document.querySelectorAll(`${sel} svg.plot circle.point`)];
    return points.slice(0, 3).map(p => ({
      cx: parseFloat(p.getAttribute('cx')),
      cy: parseFloat(p.getAttribute('cy')),
    }));
  }, plotSel);

  // Switch to 1D
  await page.evaluate(() => {
    const bars = document.querySelectorAll('#eigen-numbers-words svg g.eigen-col');
    if (bars.length >= 1) bars[0].dispatchEvent(new Event('click'));
  });

  // Check mid-animation: SVG should still exist (not destroyed and rebuilt)
  await new Promise(r => setTimeout(r, 100));
  const midAnim = await page.evaluate((sel) => {
    const svg = document.querySelector(`${sel} svg.plot`);
    const points = [...(svg?.querySelectorAll('circle.point') || [])];
    return {
      svgExists: !!svg,
      pointCount: points.length,
    };
  }, plotSel);
  test('Dimension transition: SVG preserved during 1D↔2D animation', midAnim.svgExists);
  test('Dimension transition: points exist during animation', midAnim.pointCount > 0,
    `${midAnim.pointCount} points`);

  // Wait for animation to complete
  await new Promise(r => setTimeout(r, 1000));

  // After animation: y-coords should be near 0 (1D = strip plot rendered as 2D with y≈0)
  const after1D = await page.evaluate((sel) => {
    const points = [...document.querySelectorAll(`${sel} svg.plot circle.point`)];
    const ys = points.map(p => parseFloat(p.getAttribute('cy')));
    const allSameY = ys.every(y => Math.abs(y - ys[0]) < 1);
    return { allSameY, sampleY: ys.slice(0, 3) };
  }, plotSel);
  test('Dimension transition: 1D points on same y-line', after1D.allSameY,
    `y-values: ${after1D.sampleY.map(y => y.toFixed(1)).join(', ')}`);

  // Switch back to 2D
  await page.evaluate(() => {
    const bars = document.querySelectorAll('#eigen-numbers-words svg g.eigen-col');
    if (bars.length >= 2) bars[1].dispatchEvent(new Event('click'));
  });
  await new Promise(r => setTimeout(r, 1000));

  // Points should have spread out in y
  const back2D = await page.evaluate((sel) => {
    const points = [...document.querySelectorAll(`${sel} svg.plot circle.point`)];
    const ys = points.map(p => parseFloat(p.getAttribute('cy')));
    const yRange = Math.max(...ys) - Math.min(...ys);
    return { yRange };
  }, plotSel);
  test('Dimension transition: 2D points spread in y', back2D.yRange > 10,
    `y-range: ${back2D.yRange.toFixed(1)}`);

  // Test 3D transition: should fade out/in
  await page.evaluate(() => {
    const bars = document.querySelectorAll('#eigen-numbers-words svg g.eigen-col');
    if (bars.length >= 3) bars[2].dispatchEvent(new Event('click'));
  });
  await new Promise(r => setTimeout(r, 500));
  // 3D is now projected SVG with rotation, not canvas
  const has3D = await page.evaluate((sel) =>
    !!document.querySelector(`${sel} svg.plot`), plotSel);
  test('Dimension transition: 3D SVG exists (projected)', has3D);

  // Switch back to 2D for other tests
  await page.evaluate(() => {
    const bars = document.querySelectorAll('#eigen-numbers-words svg g.eigen-col');
    if (bars.length >= 2) bars[1].dispatchEvent(new Event('click'));
  });
  await new Promise(r => setTimeout(r, 1000));
}

async function testPanZoom() {
  // Check all SVG plots have zoom infrastructure
  const allPlots = await page.evaluate(() => {
    const svgs = [...document.querySelectorAll('svg.plot')];
    return svgs
      .filter(svg => {
        // Skip non-zoomable plots (e.g., subspace animation has its own controls)
        const container = svg.closest('.plot-container');
        return container && !container.id.includes('subspace-anim');
      })
      .map(svg => ({
        hasZoomContainer: !!svg.querySelector('g.zoom-container'),
        hasCursor: svg.style.cursor === 'grab',
        id: svg.closest('.plot-container')?.id || 'unknown',
      }));
  });

  const allHaveZoom = allPlots.every(p => p.hasZoomContainer);
  const allHaveCursor = allPlots.every(p => p.hasCursor);
  test('Pan+zoom: all plots have zoom container', allHaveZoom,
    allPlots.filter(p => !p.hasZoomContainer).map(p => p.id).join(', ') || 'all OK');
  test('Pan+zoom: all plots have grab cursor', allHaveCursor,
    allPlots.filter(p => !p.hasCursor).map(p => p.id).join(', ') || 'all OK');

  // Test that zoom is wired up by programmatically dispatching a zoom transform
  const zoomWorks = await page.evaluate(() => {
    const svg = document.querySelector('#plot-superlatives svg.plot');
    const zoomG = svg?.querySelector('g.zoom-container');
    if (!svg || !zoomG) return { wired: false };

    // Dispatch a wheel event to trigger d3.zoom
    const rect = svg.getBoundingClientRect();
    const evt = new WheelEvent('wheel', {
      deltaY: -100, clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2,
      bubbles: true, cancelable: true
    });
    svg.dispatchEvent(evt);

    // Check if transform was applied
    const transform = zoomG.getAttribute('transform');
    return { wired: true, hasTransform: !!transform, transform };
  });
  test('Pan+zoom: wheel event triggers zoom', zoomWorks.hasTransform,
    zoomWorks.transform || 'no transform');

  // Click-to-expand is disabled, so we just verify zoom didn't break the plot
  const wordsAfterZoom = await page.evaluate(() =>
    document.querySelectorAll('#plot-superlatives svg.plot text.word-label').length);
  test('Pan+zoom: plot intact after zooming', wordsAfterZoom > 0,
    `${wordsAfterZoom} words`);
}

async function testPanZoomDoesNotBreakOtherPlots() {
  // Verify that zoom on one plot doesn't affect another
  const analogySvg = await page.evaluate(() =>
    !!document.querySelector('#plot-analogy-king svg.plot g.zoom-container'));
  test('Pan+zoom: analogy plot also has zoom', analogySvg);

  const steeringSvg = await page.evaluate(() =>
    !!document.querySelector('#plot-gendered-steer svg.plot g.zoom-container'));
  test('Pan+zoom: steering plot also has zoom', steeringSvg);
}

async function testSuperlativesComplete() {
  const words = await page.evaluate(() =>
    [...document.querySelectorAll('#plot-superlatives svg.plot text.word-label')]
      .map(t => t.textContent || t.innerHTML));
  const hasSoftest = words.includes('softest');
  const hasMeanest = words.includes('meanest');
  // Note: after click-to-expand, count may be > 30
  test('Superlatives: softest present', hasSoftest);
  test('Superlatives: meanest present', hasMeanest);
}

async function testFemineFirst() {
  const firstWord = await page.evaluate(() => {
    const labels = document.querySelectorAll('#plot-gendered-steer svg.plot text.word-label');
    return labels[0]?.textContent || labels[0]?.innerHTML || '';
  });
  test('Gendered pairs feminine first', firstWord === 'woman',
    `first word: "${firstWord}"`);
}

async function testAcknowledgments() {
  const ackText = await page.evaluate(() => {
    // Find Acknowledgments section by heading text
    const headings = [...document.querySelectorAll('h2, h3')];
    const ackH = headings.find(h => /acknowledg/i.test(h.textContent));
    if (!ackH) return '';
    let text = '';
    let el = ackH.nextElementSibling;
    while (el && !/^H[1-3]$/.test(el.tagName)) { text += el.textContent; el = el.nextElementSibling; }
    return text;
  });
  test('Mueller acknowledged', ackText.includes('Mueller'));
  const citations = await page.evaluate(() => {
    const headings = [...document.querySelectorAll('h2, h3')];
    const refH = headings.find(h => /references/i.test(h.textContent));
    if (!refH) return 0;
    const ol = refH.nextElementSibling;
    return ol?.querySelectorAll('li').length || 0;
  });
  test('Citations present', citations >= 2, `got ${citations}`);
}

async function testDefaultVocab() {
  const active = await page.evaluate(() =>
    document.querySelector('.vocab-selector button.active')?.dataset.size || '');
  test('Default vocab is medium', active === 'medium', `got "${active}"`);
}

async function testExplorer() {
  // Default content should have rendered a plot on load
  const defaultPlot = await page.evaluate(() =>
    !!document.querySelector('#plot-explorer svg.plot'));
  test('Explorer: default plot rendered on load', defaultPlot);

  const defaultWords = await page.evaluate(() =>
    document.querySelectorAll('#plot-explorer svg.plot text.word-label').length);
  test('Explorer: default plot has words', defaultWords > 0, `${defaultWords} words`);

  // Type custom input (live update, debounced 500ms)
  await page.evaluate(() => {
    const el = document.getElementById('explorer-input');
    el.value = 'king queen prince princess\nman woman boy girl';
    el.dispatchEvent(new Event('input'));
  });
  await new Promise(r => setTimeout(r, 3000));

  const customWords = await page.evaluate(() => {
    const labels = [...document.querySelectorAll('#plot-explorer svg.plot text.word-label')];
    return labels.map(l => l.textContent);
  });
  test('Explorer: custom plot has expected words', customWords.includes('king') && customWords.includes('girl'),
    customWords.join(', '));

  // Check missing word handling
  await page.evaluate(() => {
    const el = document.getElementById('explorer-input');
    el.value = 'xyznotaword123';
    el.dispatchEvent(new Event('input'));
  });
  await new Promise(r => setTimeout(r, 1000));
  const errorMsg = await page.evaluate(() =>
    document.getElementById('explorer-status')?.textContent || '');
  test('Explorer: shows error for missing words', errorMsg.includes('Need at least') || errorMsg.includes('not found'),
    errorMsg.substring(0, 80));
}

// Run all tests
console.log(`\nTesting: ${URL}\n`);
const jsErrors = await setup();

await testNoJSErrors(jsErrors);
await testNoDistill();
await testHeroVisualization();
await testHeroLegend();
await testPlotCount();
await testAlignment();
await testMathRendering();
await testAnalogyInput();
await testSteeringAnimation();
// Click-to-expand is disabled — tests kept for reference if re-enabled
// await testClickToExpand();
// await testNeighborStylingAcrossDimensions();
// await testClickToExpand3D();
// await testClickToExpand1D();
await testDimensionTransitionAnimation();
await testPanZoom();
await testPanZoomDoesNotBreakOtherPlots();
await testSuperlativesComplete();
await testFemineFirst();
await testAcknowledgments();
await testDefaultVocab();
await testExplorer();

await browser.close();

// Summary
const passed = results.filter(r => r.passed).length;
const failed = results.filter(r => !r.passed).length;
console.log(`\n${passed}/${passed + failed} tests passed`);
if (failed > 0) {
  console.log('\nFailed:');
  results.filter(r => !r.passed).forEach(r =>
    console.log(`  ✗ ${r.name}${r.detail ? ': ' + r.detail : ''}`));
  process.exit(1);
}
