// Each pipeline a native run wrote down (`ORB_DUMP_PIPELINES=dir`) compiled
// cold in a browser, one at a time: its WGSL made unique to the run as
// `?nonce=` makes it (demo/cold.js), so no cache holds it, then the module
// made and the pipeline compiled with `create*PipelineAsync`, timed from
// the module to the promise. A trivial pipeline compiles first, so the
// first one timed doesn't pay for starting the browser's compiler.
//
//   pnpm -C web compiles <dir>/pipelines.tsv [--browser=chromium|firefox]
//     [--runs=n] [--only=label,…]
//
// Prints a line per pipeline (the dump's line, label and entry points,
// the fastest of the runs and every run, in ms) and the total of the
// fastest. Firefox is the installed one, headed (its WebGPU); Chromium is
// Playwright's, headless.
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, join } from "node:path";

import { launchChromium, launchFirefox } from "./serve.mjs";

const args = Object.fromEntries(
  process.argv
    .slice(2)
    .filter((a) => a.startsWith("--"))
    .map((a) => {
      const [k, v] = a.slice(2).split("=");
      return [k, v ?? true];
    }),
);
const tsv = process.argv.slice(2).find((a) => !a.startsWith("--"));
if (!tsv) {
  console.error("usage: compiles <pipelines.tsv> [--browser=…] [--runs=n]");
  process.exit(2);
}
const browserName = args.browser ?? "chromium";
const runs = Number(args.runs ?? 3);
const only = args.only?.split(",");
const dir = dirname(tsv);

/** A dumped module's WGSL, without the nonce the dumping run gave it. */
const wgsl = (file) =>
  readFileSync(join(dir, file), "utf8").replace(
    / var orb_nonce: u32 = \d+u;/g,
    "",
  );
const constants = (text) =>
  Object.fromEntries(
    text
      .split(",")
      .filter(Boolean)
      .map((kv) => {
        const [k, v] = kv.split("=");
        return [k, Number(v)];
      }),
  );
const formats = {
  Rgba8Unorm: "rgba8unorm",
  Bgra8Unorm: "bgra8unorm",
  Rgba16Float: "rgba16float",
  R32Float: "r32float",
  R8Unorm: "r8unorm",
  Rg8Unorm: "rg8unorm",
  R16Float: "r16float",
  Rg16Float: "rg16float",
  R32Uint: "r32uint",
  Rg32Uint: "rg32uint",
  Rgba32Float: "rgba32float",
  Depth32Float: "depth32float",
};
const blends = {
  None: undefined,
  "Some(PremultipliedAlpha)": {
    color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
  },
  "Some(Additive)": {
    color: { srcFactor: "one", dstFactor: "one" },
    alpha: { srcFactor: "one", dstFactor: "one" },
  },
};
const topologies = {
  TriangleList: "triangle-list",
  TriangleStrip: "triangle-strip",
  LineList: "line-list",
};

/** The dump's pipelines as WebGPU descriptors, modules by file. */
const pipelines = readFileSync(tsv, "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line, index) => {
    const f = line.split("\t");
    const stage = ([file, entryPoint, k]) => ({
      file,
      entryPoint,
      constants: constants(k),
    });
    if (f[0] === "compute")
      return { index, kind: "compute", label: f[1], cs: stage(f.slice(2, 5)) };
    const [format, write] = f[9] === "-" ? [] : f[9].split(":");
    return {
      index,
      kind: "render",
      label: f[1],
      vs: stage(f.slice(2, 5)),
      fs: stage(f.slice(5, 8)),
      targets: f[8]
        .split("|")
        .filter(Boolean)
        .map((t) => {
          const at = t.indexOf(":");
          return {
            format: formats[t.slice(0, at)],
            blend: blends[t.slice(at + 1)],
          };
        }),
      depth: format && {
        format: formats[format],
        depthWriteEnabled: write === "true",
        depthCompare: "greater-equal",
      },
      topology: topologies[f[10]],
      cull: f[11] === "true",
      samples: Number(f[12]),
    };
  })
  .filter((p) => !only || only.includes(p.label));
const files = [
  ...new Set(
    pipelines.flatMap((p) =>
      p.kind === "compute" ? [p.cs.file] : [p.vs.file, p.fs.file],
    ),
  ),
];
const sources = Object.fromEntries(files.map((f) => [f, wgsl(f)]));

// A secure origin for WebGPU: an empty page on localhost.
const server = createServer((_, res) => {
  res.setHeader("content-type", "text/html");
  res.end("<!doctype html><title>compiles</title>");
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const url = `http://127.0.0.1:${server.address().port}/`;

const browser =
  browserName === "firefox"
    ? await launchFirefox({
        prefs: { "privacy.reduceTimerPrecision": false },
      })
    : await launchChromium({ unsafe: false });
const page = await browser.newPage();
await page.goto(url);

/** In the page: compiles `p` cold with `nonce`, resolving to its ms. */
const compile = async ({ p, sources, nonce }) => {
  const gpu = (globalThis.orbCompile ??= await (async () => {
    const adapter = await navigator.gpu.requestAdapter();
    const features = ["clip-distances"].filter((f) => adapter.features.has(f));
    const device = await adapter.requestDevice({
      requiredFeatures: features,
      requiredLimits: {
        maxStorageBuffersPerShaderStage:
          adapter.limits.maxStorageBuffersPerShaderStage,
        maxStorageTexturesPerShaderStage:
          adapter.limits.maxStorageTexturesPerShaderStage,
        maxBindGroups: adapter.limits.maxBindGroups,
      },
    });
    const info = adapter.info ?? {};
    return { device, name: `${info.vendor} ${info.architecture}` };
  })());
  const { device } = gpu;
  const nonced = (code) =>
    code.replace(
      /(@(?:vertex|fragment|compute)\b[^{]*\{)/g,
      `$1 var orb_nonce: u32 = ${nonce >>> 0}u;`,
    );
  const modules = new Map();
  const module = (file) => {
    if (!modules.has(file))
      modules.set(
        file,
        device.createShaderModule({ code: nonced(sources[file]) }),
      );
    return modules.get(file);
  };
  device.pushErrorScope("validation");
  const start = performance.now();
  let made;
  try {
    if (p.kind === "compute") {
      made = await device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module: module(p.cs.file),
          entryPoint: p.cs.entryPoint,
          constants: p.cs.constants,
        },
      });
    } else {
      made = await device.createRenderPipelineAsync({
        layout: "auto",
        vertex: {
          module: module(p.vs.file),
          entryPoint: p.vs.entryPoint,
          constants: p.vs.constants,
        },
        fragment: {
          module: module(p.fs.file),
          entryPoint: p.fs.entryPoint,
          constants: p.fs.constants,
          targets: p.targets,
        },
        primitive: {
          topology: p.topology,
          cullMode: p.cull ? "back" : "none",
        },
        depthStencil: p.depth,
        multisample: { count: p.samples },
      });
    }
  } catch (e) {
    await device.popErrorScope();
    return { error: String(e.message ?? e).split("\n")[0] };
  }
  const ms = performance.now() - start;
  const error = await device.popErrorScope();
  if (error) return { error: error.message.split("\n")[0] };
  return { ms, name: gpu.name, made: !!made };
};

const warmup = {
  kind: "compute",
  label: "warm-up",
  cs: { file: "warm", entryPoint: "warm", constants: {} },
};
const warm = await page.evaluate(compile, {
  p: warmup,
  sources: { warm: "@compute @workgroup_size(1) fn warm() {}" },
  nonce: Math.floor(Math.random() * 2 ** 31),
});
console.log(`# ${browserName}: ${warm.name ?? warm.error}`);
let total = 0;
for (const p of pipelines) {
  const times = [];
  let error;
  for (let r = 0; r < runs; r++) {
    const result = await page.evaluate(compile, {
      p,
      sources,
      nonce: Math.floor(Math.random() * 2 ** 31),
    });
    if (result.error) {
      error = result.error;
      break;
    }
    times.push(result.ms);
  }
  const entries =
    p.kind === "compute"
      ? p.cs.entryPoint
      : `${p.vs.entryPoint} ${JSON.stringify(p.vs.constants)} ${p.fs.entryPoint}`;
  if (error) {
    console.log(`#${p.index}\t${p.label}\t${entries}\tfailed: ${error}`);
    continue;
  }
  const best = Math.min(...times);
  total += best;
  console.log(
    `#${p.index}\t${p.label}\t${entries}\tmin ${best.toFixed(0)}\t(${times.map((t) => t.toFixed(0)).join(" ")})`,
  );
}
console.log(`total of the fastest: ${total.toFixed(0)} ms`);
await browser.close();
server.close();
