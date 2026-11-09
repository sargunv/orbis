// The job workers and the placement worker, spawned by the map.

/** The default: `clamp(hardwareConcurrency − 2, 1, 4)`. */
export function defaultJobWorkers(): number {
  return Math.min(Math.max((navigator.hardwareConcurrency ?? 4) - 2, 1), 4);
}

/** A worker, and the port the map talks to it over. */
export interface Spawned {
  worker: Worker;
  port: MessagePort;
}

/** A job worker. It loads its script while `module` compiles, and gets
 * it once it has. */
export function spawnJob(module: WebAssembly.Module | Promise<WebAssembly.Module>): Spawned {
  return spawn(new Worker(new URL("./job-worker.ts", import.meta.url), { type: "module" }), module);
}

/** The placement worker, as a job worker. */
export function spawnPlacement(module: WebAssembly.Module | Promise<WebAssembly.Module>): Spawned {
  return spawn(new Worker(new URL("./placement-worker.ts", import.meta.url), { type: "module" }), module);
}

function spawn(worker: Worker, module: WebAssembly.Module | Promise<WebAssembly.Module>): Spawned {
  const channel = new MessageChannel();
  void Promise.resolve(module).then((module) => worker.postMessage({ module, port: channel.port2 }, [channel.port2]), () => {});
  return { worker, port: channel.port1 };
}
