const {chromium}=require('C:/Users/YOGA/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const {pathToFileURL}=require('node:url');const path=require('node:path');
(async()=>{
 const folder=path.resolve(process.argv[2]||'question3/global_policy/joint_rollout/outputs/report');
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(pathToFileURL(path.join(folder,'comparison.html')).href);
 const seeds=await page.locator('#seed option').evaluateAll(a=>a.map(o=>o.value));
 const names=await page.locator('#variant option').evaluateAll(a=>a.map(o=>o.value));
 for(const seed of seeds){await page.locator('#seed').selectOption(seed);for(const name of names){await page.locator('#variant').selectOption(name);await page.locator('#step').fill('1');await page.locator('#step').dispatchEvent('input');}}
 await page.locator('#seed').selectOption(seeds[0]);await page.locator('#variant').selectOption('combined');await page.locator('#terminals').check();
 await page.screenshot({path:path.join(folder,'comparison.png'),fullPage:true});
 const text=await page.locator('body').innerText();if(errors.length||text.includes('undefined')||text.includes('NaN'))throw Error(JSON.stringify({errors}));
 const replay=await page.locator('#replay').getAttribute('href');await page.goto(pathToFileURL(path.join(folder,replay)).href);
 const max=await page.locator('#step').getAttribute('max');await page.locator('#step').fill(String(Math.floor(Number(max)/2)));await page.locator('#step').dispatchEvent('input');await page.locator('#coverage-channel').selectOption('1');await page.locator('#truth').check();
 await page.screenshot({path:path.join(folder,'replay.png'),fullPage:true});
 if(errors.length)throw Error(JSON.stringify(errors));console.log(JSON.stringify({errors,caseCount:seeds.length,variants:names.length,replayFrames:Number(max)+1}));await browser.close();
})().catch(e=>{console.error(e);process.exitCode=1;});
