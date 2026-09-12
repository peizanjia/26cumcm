// Local file only; browser automation never connects to the official simulator.
const { chromium } = require('C:/Users/YOGA/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const { pathToFileURL } = require('node:url');
const path = require('node:path');
(async () => {
  const folder = path.resolve(process.argv[2] || 'question3/global_policy/dynamic/outputs/demo');
  const browser = await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
  const page = await browser.newPage({viewport:{width:1320,height:1120}});
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(pathToFileURL(path.join(folder,'replay.html')).href);
  await page.locator('#coverage-channel').selectOption('2');
  await page.evaluate(()=>{const s=document.getElementById('step');s.value=s.max;s.dispatchEvent(new Event('input'));});
  await page.locator('#truth').check();
  await page.screenshot({path:path.join(folder,'replay_final.png'),fullPage:true});
  const final=await page.locator('#closure').innerText();
  const frame=await page.evaluate(()=>DATA.frames.findIndex(f=>f.decision?.candidates?.length));
  if(frame<0)throw Error('No action comparison available');
  await page.evaluate(i=>{const s=document.getElementById('step');s.value=i;s.dispatchEvent(new Event('input'));},frame);
  await page.screenshot({path:path.join(folder,'replay_decision.png'),fullPage:true});
  const rows=await page.locator('#candidate-costs tr').count();
  if(errors.length||rows<2||await page.locator('#channels tr').count()!==20)throw Error(JSON.stringify({errors,rows}));
  console.log(JSON.stringify({errors,candidateRows:rows,finalClosure:final}));
  await browser.close();
})().catch(e=>{console.error(e);process.exitCode=1;});
