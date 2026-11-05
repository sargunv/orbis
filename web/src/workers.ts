// The job workers and the placement worker (plan §7.1), spawned by the map.

/** Plan §7.1's default: `clamp(hardwareConcurrency − 2, 1, 4)`. */
export function defaultJobWorkers(): number {
  return Math.min(Math.max((navigator.hardwareConcurrency ?? 4) - 2, 1), 4);
}

/** Spawns `count` job workers and the placement worker; the map talks to
 * them over `ports` and `placement`. */
export function spawnWorkers(module: WebAssembly.Module, count: number) {
  const workers: Worker[] = [];
  const ports: MessagePort[] = [];
  for (let i = 0; i < count; i++) {
    const channel = new MessageChannel();
    const job = new Worker(new URL("./job-worker.ts", import.meta.url), { type: "module" });
    job.postMessage({ module, port: channel.port2 }, [channel.port2]);
    ports.push(channel.port1);
    workers.push(job);
  }
  const placement = spawnPlacement(module);
  workers.push(placement.worker);
  return { workers, ports, placement: placement.port };
}

/** A placement worker, and the port the map talks to it over. */
export function spawnPlacement(module: WebAssembly.Module) {
  const channel = new MessageChannel();
  const worker = new Worker(new URL("./placement-worker.ts", import.meta.url), { type: "module" });
  worker.postMessage({ module, port: channel.port2 }, [channel.port2]);
  return { worker, port: channel.port1 };
}
