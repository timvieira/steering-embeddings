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
  const before = await page.evaluate(() =>
    document.querySelectorAll('#plot-superlatives svg.plot text.word-label').length);

  const word = await page.$('#plot-superlatives svg.plot text.word-label');
  if (word) {
    await word.click();
    await new Promise(r => setTimeout(r, 2000));
  }

  const after = await page.evaluate(() =>
    document.querySelectorAll('#plot-superlatives svg.plot text.word-label').length);
  test('Click-to-expand adds neighbors', after > before,
    `${before} → ${after}`);
}

async function testPanZoom() {
  const hasZoom = await page.evaluate(() => {
    const svg = document.querySelector('svg.plot');
    return svg?.style.cursor === 'grab' && !!svg.querySelector('g.zoom-container');
  });
  test('2D plots have pan+zoom', hasZoom);
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
await testPanZoom();
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
