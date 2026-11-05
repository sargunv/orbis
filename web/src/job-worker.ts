// A job worker: fetches and lays out tiles, rasterizes glyphs and answers
// feature queries for the map. Requests and replies are Rust's messages
// (orb-web's `jobs` module): each reply's message, packed tile or DEM
// pixels, and image tile's ImageBitmap are transferred.
//
// Spawner → worker: {module, port}. Port messages are Rust's (orb-web's
// `jobs` module); this file only carries them.
import init, { JobWorker, reloadRegions } from "../pkg/core.js";
import { onRegionsChanged } from "./regions.ts";
import { loadShaper } from "./shaper.ts";

self.onmessage = async ({ data: { module, port } }: MessageEvent<{ module: WebAssembly.Module; port: MessagePort }>) => {
  await init({ module_or_path: module });
  const jobs = new JobWorker();
  onRegionsChanged(reloadRegions);
  // Thai, Lao, Khmer and Burmese word breaks load when labels need them.
  jobs.setWordDataUrl(new URL("../pkg/seasia.bin", import.meta.url).href);
  // Styles with `font-faces` draw their files with shaper.wasm.
  jobs.setShaperLoader(loadShaper);
  port.onmessage = async ({ data }) => {
    try {
      const reply = await jobs.receive(data);
      if (reply) port.postMessage(reply, [reply.message.buffer, reply.buffer, reply.image].filter(Boolean));
    } catch (e) {
      // Requests with an id reply with their failure; this is a malformed
      // message or a style update that didn't apply.
      console.error(e);
    }
  };
};
