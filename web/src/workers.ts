// The job workers and the placement worker (plan §7.1), spawned by the
// thread that draws: the page in main-thread mode, the render worker in
// worker mode, so the page creates no workers it never talks to.

/** Plan §7.1's default: `clamp(hardwareConcurrency − 2, 1, 4)`. */
export function defaultJobWorkers(): number {
  return Math.min(Math.max((navigator.hardwareConcurrency ?? 4) - 2, 1), 4);
}

/** Spawns `count` job workers and the placement worker; the map talks to
 * them over `ports` and `placement`. Job workers fetch through the HTTP
 * cache when `httpCache` (its byte limit) isn't 0. */
export function spawnWorkers(module: WebAssembly.Module, count: number, httpCache: number) {
  const workers: Worker[] = [];
  const ports: MessagePort[] = [];
  for (let i = 0; i < count; i++) {
    const channel = new MessageChannel();
    const job = new Worker(new URL("./job-worker.ts", import.meta.url), { type: "module" });
    job.postMessage({ module, port: channel.port2, httpCache }, [channel.port2]);
    ports.push(channel.port1);
    workers.push(job);
  }
  const placement = new MessageChannel();
  const placer = new Worker(new URL("./placement-worker.ts", import.meta.url), { type: "module" });
  placer.postMessage({ module, port: placement.port2 }, [placement.port2]);
  workers.push(placer);
  return { workers, ports, placement: placement.port1 };
}
