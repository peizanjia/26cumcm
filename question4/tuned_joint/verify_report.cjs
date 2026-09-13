/** Repeatable isolated Edge QA for the compact, recorded-history report.
 * Usage: node verify_report.cjs --report <report.html or URL> --output <QA dir>
 * No existing browser tabs, servers, simulator or scenario seeds are touched.
 */
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const {pathToFileURL} = require("node:url");
const assert = require("node:assert/strict");
const playwrightPath = process.env.Q4_PLAYWRIGHT_PACKAGE || "C:/Users/WangYing/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright";
let chromium;
try { ({chromium} = require("playwright")); }
catch { ({chromium} = require(playwrightPath)); }

function argumentsFrom(argv) {
  const args = {};
  for (let i=0;i<argv.length;i++) {
    if (!argv[i].startsWith("--") || i+1>=argv.length) throw new Error("Expected --report PATH and --output PATH");
    args[argv[i].slice(2)] = argv[++i];
  }
  if (!args.report) throw new Error("--report is required; no default live page is opened");
  const remote = /^https?:\/\//i.test(args.report);
  const output = path.resolve(args.output || (remote ? "report_browser_qa" : path.join(path.dirname(args.report), "figures", "browser_qa")));
  return {url: remote ? args.report : pathToFileURL(path.resolve(args.report)).href, output};
}

function approx(actual, expected, message, tolerance=1e-6) {
  assert.ok(Math.abs(actual-expected)<=tolerance, `${message}: ${actual} != ${expected}`);
}

async function state(page) {
  return page.evaluate(() => {
    const counts={};
    for (const side of ["left", "right"]) {
      const r=run(side), done=r.commands.filter(c=>c.time_s<=time+1e-7);
      counts[side]={strategy:document.getElementById(side).value,
        commands:r.commands.length, ended:done.length,
        clears:new Set(done.filter(c=>c.kind==="clear"&&c.response.clear_result==="success").map(c=>c.channel)).size,
        sources:r.summary.source_count,
        sourceInfo:document.getElementById(side+"-info").textContent,
        plottedMeasurements:document.querySelectorAll(`#${side}-map circle[stroke="#9cabba"], #${side}-map circle[fill="#367bb0"]`).length,
        plottedClears:[...document.querySelectorAll(`#${side}-map text`)].filter(n=>/^[★×]/.test(n.textContent)).length,
        plottedTruth:[...document.querySelectorAll(`#${side}-map text`)].filter(n=>/^c\d+$/.test(n.textContent)).length,
        plottedRoutes:document.querySelectorAll(`#${side}-map path[stroke="#6d91b1"]`).length};
    }
    return {time, max:maximum(), currentCase:activeCase().seed, counts,
      viewport:window.innerWidth, scrollWidth:document.documentElement.scrollWidth,
      bodyScrollWidth:document.body.scrollWidth};
  });
}

async function verifyVisibleState(page) {
  const s=await state(page);
  assert.ok(s.scrollWidth<=s.viewport+1, `Horizontal document overflow ${s.scrollWidth} > ${s.viewport}`);
  assert.ok(s.bodyScrollWidth<=s.viewport+1, `Horizontal body overflow ${s.bodyScrollWidth} > ${s.viewport}`);
  for (const side of ["left","right"]) {
    const r=s.counts[side];
    assert.ok(r.sourceInfo.includes(`已完成 ${r.ended}/${r.commands} 个动作`), `${side}: displayed completed count disagrees with actual time`);
    assert.ok(r.sourceInfo.includes(`成功清除 ${r.clears}/${r.sources}`), `${side}: displayed clear count disagrees with real completed actions`);
  }
  return s;
}

async function setTimeFraction(page, fraction) {
  await page.locator("#time").evaluate((input,f)=>{
    input.value=String(Number(input.max)*f);
    input.dispatchEvent(new Event("input",{bubbles:true}));
  },fraction);
  return verifyVisibleState(page);
}

async function testViewport(browser, viewport, name, output) {
  const context=await browser.newContext({viewport,deviceScaleFactor:1});
  const page=await context.newPage();
  const errors=[],failedRequests=[];
  page.on("pageerror",error=>errors.push(error.message));
  page.on("console",message=>{if(message.type()==="error")errors.push(message.text());});
  page.on("requestfailed",request=>failedRequests.push({url:request.url(),error:request.failure()?.errorText}));
  const record={name,viewport,cases:[],strategyPairs:0,timePositions:0,errors,failedRequests,screenshots:[]};
  try {
    await page.goto(options.url,{waitUntil:"load"});
    await page.waitForSelector("#left-map circle[fill='#2474ae']");
    const metadata=await page.evaluate(()=>({cases:DATA.cases.map(c=>c.seed),strategies:DATA.strategies,
      reference:DATA.reference,current:DATA.current,compatibility:DATA.compatibility,
      figureCount:DATA.figures.length,commandCount:DATA.cases.reduce((n,c)=>n+Object.values(c.runs).reduce((m,r)=>m+r.commands.length,0),0)}));
    record.metadata=metadata;
    for (let caseIndex=0;caseIndex<metadata.cases.length;caseIndex++) {
      await page.selectOption("#case",String(caseIndex));
      const caseResult={seed:metadata.cases[caseIndex],pairs:[]};
      for (const left of metadata.strategies) for (const right of metadata.strategies) {
        await page.selectOption("#left",left);
        await page.selectOption("#right",right);
        for (const fraction of [0,.25,.75,1]) {
          const s=await setTimeFraction(page,fraction);
          assert.equal(s.currentCase,metadata.cases[caseIndex]);
          record.timePositions++;
        }
        caseResult.pairs.push([left,right]);
        record.strategyPairs++;
      }
      await page.selectOption("#left",metadata.reference);
      await page.selectOption("#right",metadata.current);
      await page.click("#end");
      let s=await verifyVisibleState(page);
      approx(s.time,s.max,"End button uses the shared actual total");
      for (const side of ["left","right"]) {
        assert.equal(s.counts[side].ended,s.counts[side].commands);
        assert.ok(s.counts[side].plottedMeasurements>0,"Recorded measurements must be visible at the end");
        assert.ok(s.counts[side].plottedClears>0,"Recorded clear results must be visible at the end");
        assert.ok(s.counts[side].plottedTruth>0,"Offline source labels must be visible when enabled");
      }
      for (const [control,field] of [["truth","plottedTruth"],["measures","plottedMeasurements"],["clears","plottedClears"],["route","plottedRoutes"]]) {
        await page.locator("#"+control).uncheck();
        s=await verifyVisibleState(page);
        for (const side of ["left","right"])assert.equal(s.counts[side][field],0,`${control} should disappear on ${side}`);
        await page.locator("#"+control).check();
        s=await verifyVisibleState(page);
        for (const side of ["left","right"])assert.ok(s.counts[side][field]>0,`${control} should reappear on ${side}`);
      }
      await page.click("#start");
      s=await verifyVisibleState(page);
      approx(s.time,0,"Start button resets actual time");
      assert.equal(s.counts.left.ended,0);assert.equal(s.counts.right.ended,0);
      const firstTime=await page.evaluate(()=>Math.min(run("left").commands[0].time_s,run("right").commands[0].time_s));
      await page.click("#next");s=await verifyVisibleState(page);approx(s.time,firstTime,"Next step goes to the next real completed action");
      await page.click("#previous");s=await verifyVisibleState(page);approx(s.time,0,"Previous step returns to the prior real event");
      await page.selectOption("#speed","100");
      await page.click("#play");
      await page.waitForTimeout(180);
      s=await state(page);assert.ok(s.time>0,"Playback must advance actual virtual time");
      await page.click("#play");
      const paused=await state(page);
      await page.waitForTimeout(140);
      s=await verifyVisibleState(page);approx(s.time,paused.time,"Pause keeps time fixed");
      record.cases.push(caseResult);
    }
    // Full page layout and local relative assets are checked without starting a server.
    await page.locator("details summary").click();
    await page.locator("#gallery").scrollIntoViewIfNeeded();
    await page.waitForFunction(()=>[...document.querySelectorAll("#gallery img")].every(i=>i.complete&&i.naturalWidth>0));
    assert.equal(await page.locator("#gallery img").count(),metadata.figureCount);
    await verifyVisibleState(page);
    await page.locator("details summary").click();
    await page.selectOption("#case","0");
    await setTimeFraction(page,.67);
    await page.evaluate(()=>window.scrollTo(0,0));
    const shot=path.join(output,`${name}.png`);
    await page.screenshot({path:shot,fullPage:true});record.screenshots.push(shot);
    const replayShot=path.join(output,`${name}_replay.png`);
    await page.locator("section.panel").first().screenshot({path:replayShot});record.screenshots.push(replayShot);
    assert.deepEqual(errors,[],"Browser JavaScript/console errors");
    assert.deepEqual(failedRequests,[],"Failed page or relative-image requests");
    record.passed=true;
  } catch(error) {
    record.passed=false;record.failure=error.stack||String(error);
    try {const shot=path.join(output,`${name}_failure.png`);await page.screenshot({path:shot,fullPage:true});record.screenshots.push(shot);}catch{}
    throw Object.assign(error,{qaRecord:record});
  } finally {
    fs.writeFileSync(path.join(output,`${name}.json`),JSON.stringify(record,null,2)+"\n");
    await context.close();
  }
  return record;
}

const options=argumentsFrom(process.argv.slice(2));
(async()=>{
  fs.mkdirSync(options.output,{recursive:true});
  const browser=await chromium.launch({channel:"msedge",headless:true});
  const result={report:options.url,scope:"isolated browser interaction QA only; no simulation",browser:"Edge",viewports:[]};
  try {
    result.viewports.push(await testViewport(browser,{width:1440,height:1120},"desktop",options.output));
    result.viewports.push(await testViewport(browser,{width:390,height:844},"mobile",options.output));
    result.passed=true;
  } catch(error) {
    if(error.qaRecord)result.viewports.push(error.qaRecord);
    result.passed=false;result.failure=error.stack||String(error);process.exitCode=1;
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(options.output,"qa.json"),JSON.stringify(result,null,2)+"\n");
  }
  process.stdout.write(JSON.stringify({passed:result.passed,output:options.output,
    viewports:result.viewports.map(v=>({name:v.name,passed:v.passed,strategyPairs:v.strategyPairs,timePositions:v.timePositions,errors:v.errors})),failure:result.failure},null,2)+"\n");
})();
