// The placement worker: places the map's labels (decision 0009),
// answering each frame with its commit, transferred, or with `{error}`
// when it couldn't place it.
//
// Spawner → worker: {module, port}. Port messages are Rust's (orb-web's
// `placement` module); this file only carries them.
import init, { PlacementWorker } from "../pkg/core.js";

self.onmessage = async ({ data: { module, port } }: MessageEvent<{ module: WebAssembly.Module; port: MessagePort }>) => {
  await init({ module_or_path: module });
  const placer = new PlacementWorker();
  port.onmessage = ({ data }) => {
    try {
      const reply = placer.receive(data);
      if (reply) port.postMessage(reply, [reply.committed.buffer]);
    } catch (e) {
      // A frame asked for is always answered, so the map asks again.
      if (data?.place) port.postMessage({ error: String(e instanceof Error ? e.message : e) });
      else console.error(e);
    }
  };
};
