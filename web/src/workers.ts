// The job workers and the placement worker, spawned by the map.

import { heldModules, type Build } from "./builds.ts";

/** The default: `clamp(hardwareConcurrency − 2, 1, 4)`. */
export function defaultJobWorkers(): number {
  return Math.min(Math.max((navigator.hardwareConcurrency ?? 4) - 2, 1), 4);
}

/** A worker, and the port the map talks to it over. */
export interface Spawned {
  worker: Worker;
  port: MessagePort;
}

/** A map's job workers and placement worker. */
export interface Workers {
  jobs: Spawned[];
  placement: Spawned;
}

/** `jobs` job workers and the placement worker, loading their scripts
 * while `build` compiles. */
export function spawnWorkers(build: Promise<Build>, jobs: number): Workers {
  return {
    jobs: Array.from({ length: jobs }, () => spawnJob(build)),
    placement: spawnPlacement(build),
  };
}

/** Stops workers a map didn't start with. */
export function terminateWorkers({ jobs, placement }: Workers): void {
  for (const { worker } of [...jobs, placement]) worker.terminate();
}

/** A job worker. It loads its script while `build` compiles, and gets
 * the compiled module once it has. */
export function spawnJob(build: Build | Promise<Build>): Spawned {
  return spawn(build, false);
}

/** The placement worker, as a job worker. */
export function spawnPlacement(build: Build | Promise<Build>): Spawned {
  return spawn(build, true);
}

function spawn(build: Build | Promise<Build>, placement: boolean): Spawned {
  const worker = new Worker(new URL("./helper-worker.ts", import.meta.url), {
    type: "module",
  });
  const channel = new MessageChannel();
  void Promise.resolve(build).then(
    ({ backend, module }) =>
      worker.postMessage(
        {
          backend,
          module,
          modules: heldModules(backend),
          port: channel.port2,
          placement,
        },
        [channel.port2],
      ),
    () => {},
  );
  return { worker, port: channel.port1 };
}
