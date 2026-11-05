// The placement worker: places the map's labels (decision 0009),
// answering each frame with its commit, transferred, or with `{error}`
// when it couldn't place it; a trapped one says it is lost and closes.
//
// Spawner → worker: {module, port}. Port messages are Rust's (orb-web's
// `placement` module); this file only carries them.
import init, { PlacementWorker } from "../pkg/core.js";

self.onmessage = async ({ data: { module, port } }: MessageEvent<{ module: WebAssembly.Module; port: MessagePort }>) => {
  await init({ module_or_path: module });
  const placer = new PlacementWorker();
  port.onmessage = ({ data }) => {
    if (data?.trap) placer.trap();
    try {
      const reply = placer.receive(data);
      if (reply) port.postMessage(reply, [reply.committed.buffer]);
    } catch (e) {
      // A frame asked for is always answered, so the map asks again.
      if (data?.place) port.postMessage({ error: String(e instanceof Error ? e.message : e) });
      else console.error(e);
    }
  };
  // A trap (a Rust panic) leaves the wasm unusable: the worker says it is
  // lost and closes, and the page replaces it.
  const trapped = (e: unknown) => {
    if (!(e instanceof WebAssembly.RuntimeError)) return false;
    port.postMessage({ lost: e.message });
    self.close();
    return true;
  };
  self.addEventListener("unhandledrejection", (e) => { if (trapped(e.reason)) e.preventDefault(); });
  self.addEventListener("error", (e) => { if (trapped(e.error)) e.preventDefault(); });
};
