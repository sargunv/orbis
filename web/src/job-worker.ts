// A job worker: fetches and lays out tiles, rasterizes glyphs and answers
// feature queries for the map. Requests and replies are Rust's messages
// (orb-web's `jobs` module): each reply's message, packed tile or DEM
// pixels, and image tile's ImageBitmap are transferred.
//
// Spawner → worker: {module, port}. Port messages are Rust's (orb-web's
// `jobs` module); this file only carries them.
import init, { JobWorker, reloadRegions, setRequestHook } from "../pkg/core.js";
import { onRegionsChanged } from "./regions.ts";
import { askPage } from "./requests.ts";
import { loadShaper } from "./shaper.ts";

self.onmessage = async ({ data: { module, port } }: MessageEvent<{ module: WebAssembly.Module; port: MessagePort }>) => {
  await init({ module_or_path: module });
  const jobs = new JobWorker();
  onRegionsChanged(reloadRegions);
  // Thai, Lao, Khmer and Burmese word breaks load when labels need them.
  jobs.setWordDataUrl(new URL("../pkg/seasia.bin", import.meta.url).href);
  // Styles with `font-faces` draw their files with shaper.wasm.
  jobs.setShaperLoader(loadShaper);
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
  // The page's request transform, asked for each fetch while it has one.
  const page = askPage(port);
  port.onmessage = async ({ data }) => {
    if (page.answer(data)) return;
    const { requestHook } = data as { requestHook?: boolean };
    if (requestHook !== undefined) {
      setRequestHook(requestHook ? page.hook : undefined);
      return;
    }
    try {
      if ((data as { trap?: boolean }).trap) jobs.trap();
      const reply = await jobs.receive(data);
      if (reply) port.postMessage(reply, [reply.message.buffer, reply.buffer, reply.image].filter(Boolean));
    } catch (e) {
      // Requests with an id reply with their failure; this is a malformed
      // message or a style update that didn't apply.
      if (!trapped(e)) console.error(e);
    }
  };
};
