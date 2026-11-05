// Pipeline creation in the browser (#26), on the main thread so the page
// sees the device: the Tokyo extract from `createMap` to `idle`, counting
// the shader modules and pipelines made and timing each queue submission
// to its `onSubmittedWorkDone`. Browsers compile pipelines in the GPU
// process when they are made or first used, so the queue's wait during
// load holds the compilation; `web/scripts/pipelines.mjs` compares a
// fresh profile (cold) with the same profile reopened (the browser's
// shader cache warm).
import { createMap } from "orbis";
import { demoStyle } from "./styles.js";

const status = document.getElementById("status");
/** Posts a result to `?report=<url>`, for browsers no driver controls (web/scripts/open.mjs). */
const report = (result) => {
  const to = new URLSearchParams(location.search).get("report");
  if (to) fetch(to, { method: "POST", mode: "no-cors", body: JSON.stringify(result) });
};
const counts = { shaders: 0, render: 0, compute: 0, createMs: 0 };
const waits = [];
for (const [method, key] of [["createShaderModule", "shaders"], ["createRenderPipeline", "render"],
  ["createComputePipeline", "compute"], ["createRenderPipelineAsync", "render"], ["createComputePipelineAsync", "compute"]]) {
  const original = GPUDevice.prototype[method];
  GPUDevice.prototype[method] = function (...args) {
    counts[key]++;
    const started = performance.now();
    const result = original.apply(this, args);
    counts.createMs += performance.now() - started;
    return result;
  };
}
const submit = GPUQueue.prototype.submit;
GPUQueue.prototype.submit = function (buffers) {
  const started = performance.now();
  submit.call(this, buffers);
  const done = this.onSubmittedWorkDone().then(() => performance.now() - started);
  waits.push(done);
};

window.pipelines = (async () => {
  try {
    const started = performance.now();
    const map = await createMap({ container: document.getElementById("map"), style: demoStyle(new URLSearchParams(location.search), "tokyo"),
      thread: "main", renderer: "webgpu", camera: { center: [139.74, 35.685], zoom: 15 } });
    await map.once("idle");
    const idleMs = performance.now() - started;
    const done = await Promise.all(waits);
    const result = {
      ...counts, idleMs, submissions: done.length,
      waitTotalMs: done.reduce((a, b) => a + b, 0), waitMaxMs: Math.max(...done),
      firstWaitMs: done[0],
      adapter: await navigator.gpu.requestAdapter().then((a) => [a?.info?.vendor, a?.info?.architecture].filter(Boolean).join(" ")),
      ua: navigator.userAgent,
    };
    status.textContent = JSON.stringify(result, null, 1);
    status.dataset.state = "done";
    report(result);
    return result;
  } catch (e) {
    status.textContent = String(e);
    status.dataset.state = "error";
    report({ error: String(e) });
    throw e;
  }
})();
