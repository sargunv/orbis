// The map's helper workers, one script for both kinds so that a first
// visit downloads it once: job workers fetch and lay out tiles, rasterize
// glyphs and answer feature queries; the placement worker places the
// map's labels. Requests and replies are Rust's messages (orb-web's `jobs`
// and `placement` modules); this file only carries them.
//
// Spawner → worker: {backend, module, port, placement}: the map's compiled
// build, `placement` true for the placement worker. A job worker transfers each reply's message, packed
// tile or DEM pixels and image tile's ImageBitmap; the placement worker
// answers each frame with its commit, transferred, or with `{error}` when
// it couldn't place it. A trapped worker says it is lost and closes.
import type {
  JobWorker as Jobs,
  PlacementWorker as Placer,
} from "../pkg/core.js";
import { bindings, type Backend, type Bindings } from "./builds.ts";
import { onRegionsChanged } from "./regions.ts";
import { askPage } from "./requests.ts";
import { loadShaper } from "./shaper.ts";

interface Start {
  backend: Backend;
  module: WebAssembly.Module;
  port: MessagePort;
  placement?: boolean;
}

self.onmessage = async ({
  data: { backend, module, port, placement },
}: MessageEvent<Start>) => {
  const glue = await bindings(backend);
  await glue.default({ module_or_path: module });
  // A trap (a Rust panic) leaves the wasm unusable: the worker says it is
  // lost and closes, and the map replaces it.
  const trapped = (e: unknown) => {
    if (!(e instanceof WebAssembly.RuntimeError)) return false;
    port.postMessage({ lost: e.message });
    self.close();
    return true;
  };
  self.addEventListener("unhandledrejection", (e) => {
    if (trapped(e.reason)) e.preventDefault();
  });
  self.addEventListener("error", (e) => {
    if (trapped(e.error)) e.preventDefault();
  });
  port.onmessage = placement
    ? placer(glue, port, trapped)
    : jobs(glue, port, trapped);
};

function placer(
  glue: Bindings,
  port: MessagePort,
  trapped: (e: unknown) => boolean,
): (e: MessageEvent) => void {
  const placer: Placer = new glue.PlacementWorker();
  return ({ data }) => {
    try {
      if (data?.trap) placer.trap();
      const reply = placer.receive(data);
      if (reply) port.postMessage(reply, [reply.committed.buffer]);
    } catch (e) {
      // A trap in a frame's placement loses the worker like any other: its
      // wasm is unusable, so answering with an error would have the map
      // ask the same dead worker again. Any other failure answers a frame
      // asked for, so the map asks again.
      if (trapped(e)) return;
      if (data?.place)
        port.postMessage({ error: String(e instanceof Error ? e.message : e) });
      else console.error(e);
    }
  };
}

function jobs(
  glue: Bindings,
  port: MessagePort,
  trapped: (e: unknown) => boolean,
): (e: MessageEvent) => Promise<void> {
  const jobs: Jobs = new glue.JobWorker();
  onRegionsChanged(glue.reloadRegions);
  // Styles with `font-faces` draw their files with shaper.wasm.
  jobs.setShaperLoader(loadShaper);
  // The page's request transform, asked for each fetch while it has one.
  const page = askPage(port);
  return async ({ data }) => {
    if (page.answer(data)) return;
    const { requestHook } = data as { requestHook?: boolean };
    if (requestHook !== undefined) {
      glue.setRequestHook(requestHook ? page.hook : undefined);
      return;
    }
    try {
      if ((data as { trap?: boolean }).trap) jobs.trap();
      const reply = await jobs.receive(data);
      if (reply)
        port.postMessage(
          reply,
          [reply.message.buffer, reply.buffer, reply.image].filter(Boolean),
        );
    } catch (e) {
      // Requests with an id reply with their failure; this is a malformed
      // message or a style update that didn't apply.
      if (!trapped(e)) console.error(e);
    }
  };
}
