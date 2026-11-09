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

/** A job worker. */
export function spawnJob(module: WebAssembly.Module): Spawned {
  const channel = new MessageChannel();
  const worker = new Worker(new URL("./job-worker.ts", import.meta.url), { type: "module" });
  worker.postMessage({ module, port: channel.port2 }, [channel.port2]);
  return { worker, port: channel.port1 };
}

/** The placement worker. */
export function spawnPlacement(module: WebAssembly.Module): Spawned {
  const channel = new MessageChannel();
  const worker = new Worker(new URL("./placement-worker.ts", import.meta.url), { type: "module" });
  worker.postMessage({ module, port: channel.port2 }, [channel.port2]);
  return { worker, port: channel.port1 };
}
