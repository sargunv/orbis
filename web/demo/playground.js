import { call, whenReady, latestFrame } from "./main.js";

const editor = document.querySelector("#style");
const error = document.querySelector("#error");
const status = document.querySelector("#edit-status");
const native = document.querySelector("#native");
const canvas = document.querySelector("#map");
let original;
let pending = [];
let frame = 0;
let revision = 0;
let chain = Promise.resolve();
let hovered;
let mirroredStyle;
let mirrorQueue = Promise.resolve();

window.addEventListener("orbis-frame", ({detail: ack}) => {
  frame++;
  pending = pending.filter((edit) => {
    if (ack.generation < edit.generation) return true;
    status.textContent = `Applied in ${frame - edit.frame} frame${frame - edit.frame === 1 ? "" : "s"} · generation ${edit.generation} · edit ${edit.ms.toFixed(3)} ms`;
    return false;
  });
});

function mirror(style = mirroredStyle) {
  mirroredStyle = style;
  if (!import.meta.env.DEV) { native.textContent = "Pages runs in the browser. Use the local playground to link a native viewer."; return Promise.resolve(); }
  const targets = new Map();
  for (const layer of style.layers) {
    if (!layer.source || !["vector","geojson"].includes(style.sources[layer.source]?.type)) continue;
    const target = {source:layer.source,sourceLayer:layer["source-layer"]};
    targets.set(JSON.stringify(target),target);
  }
  const commands = [["setStyle",style], ...[...targets.values()].map(target => ["removeFeatureState",target])];
  if (hovered) commands.push(["setFeatureState",hovered,{hover:true}]);
  const body = JSON.stringify(commands);
  mirrorQueue = mirrorQueue.catch(() => {}).then(async () => {
    const response = await fetch("/orbis/__playground/style", {
      method:"PUT",headers:{"Content-Type":"application/json"},body,
    });
    const result = await response.json();
    native.textContent = result.linked ? "Style and hover sent to the native viewer file." : "Native link is off. Set ORB_PLAYGROUND_STYLE when starting Vite.";
  });
  return mirrorQueue;
}

function apply() {
  const text = editor.value;
  const current = ++revision;
  chain = chain.catch(() => {}).then(async () => {
    try {
      const style = JSON.parse(text);
      const before = frame;
      const started = performance.now();
      const generation = await call("edit", JSON.stringify([["setStyle",style]]));
      if (current !== revision) return;
      pending.push({generation, frame:before, ms:performance.now()-started});
      error.textContent = "";
      editor.removeAttribute("aria-invalid");
      await mirror(JSON.parse(await call("getStyle")));
    } catch (e) {
      if (current !== revision) return;
      error.textContent = String(e);
      editor.setAttribute("aria-invalid","true");
    }
  });
}
let timer;
editor.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(apply,200); });
document.querySelector("#reset").addEventListener("click", () => { editor.value = original; apply(); });

let pointer;
let querying = false;
async function hover() {
  if (querying) return;
  querying = true;
  try {
    while (pointer !== undefined) {
      const point = pointer; pointer = undefined;
      await chain.catch(() => {});
      const hit = point && await call("hitTest",...point);
      const target = hit?.id != null && mirroredStyle?.sources[hit.source] ? {source:hit.source,sourceLayer:hit.sourceLayer,id:hit.id} : null;
      if (JSON.stringify(target) === JSON.stringify(hovered)) continue;
      const commands = [];
      if (hovered && mirroredStyle?.sources[hovered.source]) commands.push(["removeFeatureState",hovered,"hover"]);
      if (target) commands.push(["setFeatureState",target,{hover:true}]);
      if (commands.length) await call("edit",JSON.stringify(commands));
      hovered = target;
      if (mirroredStyle) await mirror();
      document.querySelector("#hover").textContent = hit ? `${hit.layer} / ${hit.id ?? "no id"}` : "Hover to inspect a feature";
    }
  } finally { querying = false; }
}
canvas.addEventListener("pointermove", (e) => { if (!e.buttons) { pointer=[e.offsetX,e.offsetY]; void hover(); } });
canvas.addEventListener("pointerleave", () => { pointer=null; void hover(); });

await whenReady;
if (!latestFrame?.idle) await new Promise((resolve) => {
  const listener = ({detail}) => { if (detail.idle) { window.removeEventListener("orbis-frame",listener); resolve(); } };
  window.addEventListener("orbis-frame",listener);
});
const {layers,sources,...root} = JSON.parse(await call("getStyle"));
original = JSON.stringify({...root,layers,sources},null,2);
editor.value = original;
status.textContent = "Ready. Edits apply after a 200 ms pause.";
await mirror(JSON.parse(original));
// The same API is useful from the console and the headless verification harness.
window.playground = {call, get frame() { return frame; }, get latestFrame() { return latestFrame; }};
