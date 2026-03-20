/**
 * UI smoke tests for the Steering Word Embeddings article.
 *
 * Usage:
 *   1. Start server: python3 -m http.server 8768 --directory article
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
    if (!err.message.includes('template.v1')) jsErrors.push(err.message);
  });

  await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await new Promise(r => setTimeout(r, WAIT_MS));

  return jsErrors;
}

async function testNoJSErrors(jsErrors) {
  test('No JS errors (excluding Distill template)', jsErrors.length === 0,
    jsErrors.length ? jsErrors.join('; ').substring(0, 200) : '');
}

async function testBannerHidden() {
  const visible = await page.evaluate(() => {
    const b = document.querySelector('dt-banner');
    return b ? window.getComputedStyle(b).display !== 'none' : false;
  });
  test('Distill draft banner hidden', !visible);
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
    const p = document.querySelector('#hero-viz + p');
    return p?.innerHTML || '';
  });
  test('Hero legend has steel blue color', legend.includes('#5778a4'));
  test('Hero legend has orange color', legend.includes('#e49444'));
}

async function testPlotCount() {
  const count = await page.evaluate(() => document.querySelectorAll('svg.plot').length);
  test('At least 7 SVG plots rendered', count >= 7, `got ${count}`);
}

async function testAlignment() {
  const m = await page.evaluate(() => {
    const p = document.querySelector('dt-article > p');
    const svg = document.querySelector('svg.plot');
    const caption = document.querySelector('.variance-caption');
    return {
      text: Math.round(p?.getBoundingClientRect().left),
      svg: Math.round(svg?.getBoundingClientRect().left),
      caption: Math.round(caption?.getBoundingClientRect().left),
    };
  });
  test('SVG aligns with text', Math.abs(m.text - m.svg) < 5,
    `text=${m.text} svg=${m.svg}`);
  test('Caption aligns with text', Math.abs(m.text - m.caption) < 5,
    `text=${m.text} caption=${m.caption}`);
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

  // Click a second word to verify recursive expansion
  const words2 = await page.$$(`${plotSel} svg.plot text.word-label`);
  if (words2.length > after.count - 3) {
    // Click one of the newly added neighbors
    await words2[words2.length - 1].click();
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
    const bars = document.querySelectorAll('#eigen-numbers-digits svg rect.bar');
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
    const bars = document.querySelectorAll('#eigen-numbers-digits svg rect.bar');
    if (bars.length >= 2) bars[1].dispatchEvent(new Event('click'));
  });
  await new Promise(r => setTimeout(r, 2000));
}

async function testClickToExpand3D() {
  // Switch numbers-digits to 3D
  await page.evaluate(() => {
    const bars = document.querySelectorAll('#eigen-numbers-digits svg rect.bar');
    if (bars.length >= 3) bars[2].dispatchEvent(new Event('click'));
  });
  await new Promise(r => setTimeout(r, 3000));

  const hasCanvas = await page.evaluate(() =>
    !!document.querySelector('#plot-numbers-digits canvas'));
  test('Click-to-expand 3D: canvas exists after switch', hasCanvas);

  // Note: raycasting click is hard to test in headless puppeteer,
  // but we verify the infrastructure is wired
  const hasRaycastSetup = await page.evaluate(() => {
    // The canvas should have pointerup listener (for raycasting)
    const canvas = document.querySelector('#plot-numbers-digits canvas');
    // Can't directly check listeners, but verify canvas is interactive
    return canvas?.style.cursor === 'grab';
  });
  test('Click-to-expand 3D: canvas is interactive', hasRaycastSetup);

  // Switch back to 2D for subsequent tests
  await page.evaluate(() => {
    const bars = document.querySelectorAll('#eigen-numbers-digits svg rect.bar');
    if (bars.length >= 2) bars[1].dispatchEvent(new Event('click'));
  });
  await new Promise(r => setTimeout(r, 2000));
}

async function testClickToExpand1D() {
  // Switch numbers-words to 1D
  await page.evaluate(() => {
    const bars = document.querySelectorAll('#eigen-numbers-words svg rect.bar');
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
    const bars = document.querySelectorAll('#eigen-numbers-words svg rect.bar');
    if (bars.length >= 2) bars[1].dispatchEvent(new Event('click'));
  });
  await new Promise(r => setTimeout(r, 2000));
}

async function testPanZoom() {
  // Check all SVG plots have zoom infrastructure
  const allPlots = await page.evaluate(() => {
    const svgs = [...document.querySelectorAll('svg.plot')];
    return svgs.map(svg => ({
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

  // Test that zoom filter doesn't block word clicks
  // (zoom should only drag from background, not from words/circles)
  const zoomFilter = await page.evaluate(() => {
    // After zooming, clicking a word should still expand
    const wordsBefore = document.querySelectorAll('#plot-superlatives svg.plot text.word-label').length;
    return { wordsBefore };
  });

  const wordAfterZoom = await page.$('#plot-superlatives svg.plot text.word-label');
  if (wordAfterZoom) {
    await wordAfterZoom.click();
    await new Promise(r => setTimeout(r, 2000));
  }
  const wordsAfterClick = await page.evaluate(() =>
    document.querySelectorAll('#plot-superlatives svg.plot text.word-label').length);
  test('Pan+zoom: click-to-expand works after zooming', wordsAfterClick > zoomFilter.wordsBefore,
    `${zoomFilter.wordsBefore} → ${wordsAfterClick}`);
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
  const appendix = await page.evaluate(() =>
    document.querySelector('dt-appendix')?.textContent || '');
  test('Mueller acknowledged', appendix.includes('Mueller'));
  const citations = await page.evaluate(() =>
    document.querySelectorAll('dt-cite').length);
  test('Citations present', citations >= 2, `got ${citations}`);
}

async function testDefaultVocab() {
  const active = await page.evaluate(() =>
    document.querySelector('.vocab-selector button.active')?.dataset.size || '');
  test('Default vocab is medium', active === 'medium', `got "${active}"`);
}

// Run all tests
console.log(`\nTesting: ${URL}\n`);
const jsErrors = await setup();

await testNoJSErrors(jsErrors);
await testBannerHidden();
await testHeroVisualization();
await testHeroLegend();
await testPlotCount();
await testAlignment();
await testMathRendering();
await testAnalogyInput();
await testSteeringAnimation();
await testClickToExpand();
await testNeighborStylingAcrossDimensions();
await testClickToExpand3D();
await testClickToExpand1D();
await testPanZoom();
await testPanZoomDoesNotBreakOtherPlots();
await testSuperlativesComplete();
await testFemineFirst();
await testAcknowledgments();
await testDefaultVocab();

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
