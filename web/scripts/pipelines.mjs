// Pipeline creation in Chrome (#26): web/dist/pipelines.html loads the
// Tokyo extract on the main thread `--runs` times in a fresh profile
// (cold: the browser's shader cache empty) and as often again in that
// profile reopened (warm), reporting pipelines made and the GPU queue's
// wait from map creation to idle (`web/demo/pipelines.js`). With
// `--android`, in Chrome on this worktree's emulator (`--physical`: on its
// assigned phone, as measure.mjs): cold after `chrome://gpuclean` empties
// its GPU disk cache, warm in a tab after.
//
//   pnpm pipelines [--runs=3] [--android [--physical]]
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";
import { openAndroid, serveDemo } from "./serve.mjs";

const args = process.argv.slice(2);
const runs = Number(args.find((a) => a.startsWith("--runs="))?.slice(7) ?? 3);
const androidArg = args.includes("--android");
const physical = args.includes("--physical");
const server = await serveDemo();
const ARGS = ["--enable-features=Vulkan,WebGPU", "--use-angle=metal"];

/** One load in a profile, the browser started for it and closed after. */
async function load(profile) {
  const context = await chromium.launchPersistentContext(profile, { args: ARGS, viewport: { width: 1024, height: 768 } });
  try {
    const page = context.pages()[0] ?? await context.newPage();
    await page.goto(`${server.url}pipelines.html`);
    return await page.evaluate(() => window.pipelines);
  } finally {
    await context.close();
  }
}

/** One load in a tab of Chrome on the device, after emptying its GPU cache if `clean`. */
async function loadAndroid(clean) {
  if (clean) {
    const { page, close } = await openAndroid(server, { physical });
    await page.goto("chrome://gpuclean").catch(() => {});
    await close();
  }
  const { page, close } = await openAndroid(server, { physical });
  try {
    await page.goto(`${server.url}pipelines.html`);
    return await page.evaluate(() => window.pipelines);
  } finally {
    await close();
  }
}

const rows = [];
try {
  for (let r = 0; r < runs; r++) {
    if (androidArg) {
      const cold = await loadAndroid(true);
      const warm = await loadAndroid(false);
      rows.push({ cold, warm });
      console.error(`pipelines: run ${r + 1}: ${JSON.stringify({ cold, warm })}`);
      continue;
    }
    const profile = await mkdtemp(join(tmpdir(), "orbis-pipelines-"));
    try {
      const cold = await load(profile);
      const warm = await load(profile);
      rows.push({ cold, warm });
      console.error(`pipelines: run ${r + 1}: ${JSON.stringify({ cold, warm })}`);
    } finally {
      await rm(profile, { recursive: true, force: true });
    }
  }
} finally {
  server.close();
}
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const f = (x) => x.toFixed(0);
const { ua, adapter, shaders, render, compute } = rows[0].cold;
console.log(`\n${ua}; ${adapter}; medians of ${runs} ${androidArg ? "runs" : "fresh profiles"}\n`);
console.log(`${shaders} shader modules, ${render} render and ${compute} compute pipelines\n`);
console.log("| | map to idle ms | GPU queue wait during load: total / largest / first submission ms | pipeline calls on the page ms |");
console.log("|---|---|---|---|");
for (const key of ["cold", "warm"]) {
  const m = (k) => median(rows.map((r) => r[key][k]));
  console.log(`| ${key} | ${f(m("idleMs"))} | ${f(m("waitTotalMs"))} / ${f(m("waitMaxMs"))} / ${f(m("firstWaitMs"))} | ${m("createMs").toFixed(1)} |`);
}
