// Browser QA for the standalone report. Uses the bundled Node runtime's libraries.
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const fs = require('node:fs');
const {chromium} = require(path.join(path.dirname(process.execPath), '..', 'node_modules', 'playwright'));

(async () => {
  const folder = path.resolve(process.argv[2] || 'question3/global_policy/adaptive_mpc/outputs/development_v2/report');
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true,
  });
  const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(pathToFileURL(path.join(folder, 'comparison.html')).href);
  if (await page.locator('#truth').isChecked()) throw Error('Truth must be hidden by default');
  const initialSeed = await page.locator('#seed').inputValue();
  const seeds = await page.locator('#seed option').evaluateAll(a => a.map(o => o.value));
  const variants = await page.locator('#variant option').evaluateAll(a => a.map(o => o.value));
  let maxCandidates = 0;
  for (const seed of seeds) {
    await page.selectOption('#seed', seed);
    for (const variant of variants) {
      await page.selectOption('#variant', variant);
      const max = Number(await page.locator('#step').getAttribute('max'));
      for (const step of [0, 22, Math.floor(max / 2), max]) {
        await page.locator('#step').fill(String(Math.min(max, step)));
        await page.locator('#step').dispatchEvent('input');
        maxCandidates = Math.max(maxCandidates, await page.locator('#candidateTable tr').count());
      }
    }
  }
  await page.selectOption('#seed', initialSeed);
  await page.selectOption('#variant', variants.includes('adaptive') ? 'adaptive' : variants.at(-1));
  const scanRows = await page.locator('#scanTable tr').count();
  await page.screenshot({path: path.join(folder, 'comparison.png'), fullPage: true});
  const max = Number(await page.locator('#step').getAttribute('max'));
  await page.locator('#step').fill(String(Math.min(max, 35)));
  await page.locator('#step').dispatchEvent('input');
  await page.locator('#map').screenshot({path: path.join(folder, 'early_map.png')});
  await page.selectOption('#coverageChannel', '1');
  const coverageCircles = await page.locator('#map g[clip-path] circle').count();
  await page.locator('#map').screenshot({path: path.join(folder, 'channel_coverage.png')});
  await page.locator('#step').fill('0');
  await page.locator('#step').dispatchEvent('input');
  if (await page.locator('#map g[clip-path] circle').count()) throw Error('Future coverage leaked into initial frame');
  await page.locator('#truth').check();
  await page.locator('#candidates').uncheck();
  const body = await page.locator('body').innerText();
  if (errors.length || body.includes('undefined') || body.includes('NaN')) throw Error(JSON.stringify({errors}));
  const result = {errors, cases: seeds.length, variants: variants.length,
    maxCandidates, finalStopChannelRows: scanRows, truthDefaultHidden: true, initialSeed,
    channelCoverageCirclesAt35: coverageCircles, initialCoverageEmpty: true};
  fs.writeFileSync(path.join(folder, 'browser_qa.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
  await browser.close();
})().catch(error => {console.error(error); process.exitCode = 1;});
