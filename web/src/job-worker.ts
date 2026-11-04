// A job worker: fetches and lays out tiles, rasterizes glyphs and answers
// feature queries for the render worker, replying on its port with each
// packed tile's buffer, image tile's ImageBitmap (with a terrain DEM
// tile's elevations) or glyph bitmaps, transferred.
//
// Page → worker: {module, port}. Port messages are Rust's (orb-web's
// `jobs` module); this file only carries them.
import init, { JobWorker } from "../pkg/core.js";
import { loadShaper } from "./shaper.ts";

self.onmessage = async ({ data: { module, port } }: MessageEvent<{ module: WebAssembly.Module; port: MessagePort }>) => {
  await init({ module_or_path: module });
  const jobs = new JobWorker();
  // Thai, Lao, Khmer and Burmese word breaks load when labels need them.
  jobs.setWordDataUrl(new URL("../pkg/seasia.bin", import.meta.url).href);
  // Styles with `font-faces` draw their files with shaper.wasm.
  jobs.setShaperLoader(loadShaper);
  port.onmessage = async ({ data }) => {
    try {
      const reply = await jobs.receive(data);
      if (reply) port.postMessage(reply, [reply.buffer, reply.image, reply.dem, reply.glyphs].filter(Boolean));
    } catch (e) {
      console.error(e);
    }
  };
};
