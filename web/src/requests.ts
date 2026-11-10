// How every map on the page fetches (GL JS `transformRequest`): a page-wide
// transform each request goes through, on the page and in the map and job
// workers, which ask the page for it (a job worker of a map worker asks
// that worker, which asks the page).

/** What a transform makes of a request: the URL to fetch and headers to send. */
export interface RequestParameters {
  url: string;
  headers?: Record<string, string>;
}

/** Answers a request's parameters (or a promise of them), `null` to refuse
 * it, or `undefined` to leave it as it is. */
export type RequestTransform = (
  url: string,
) =>
  | RequestParameters
  | null
  | undefined
  | Promise<RequestParameters | null | undefined>;

type SetHook = (hook?: (url: string) => unknown) => void;

let transform: RequestTransform | undefined;
const hooks = new Set<SetHook>();
/** Where a worker hears from the page: its port, or the worker itself. */
type Port = { postMessage(message: unknown): void };

const ports = new Set<Port>();

/** Transforms every request the page's maps make from here on, theirs and
 * their workers'; no transform, none. */
export function setRequestTransform(fn?: RequestTransform): void {
  transform = fn;
  for (const set of hooks) set(fn);
  for (const port of ports) port.postMessage({ requestHook: fn !== undefined });
}

/** What the transform makes of a request for `url`; null when it refuses. */
export async function transformed(
  url: string,
): Promise<RequestParameters | null> {
  const result = await transform?.(url);
  return result === undefined ? { url } : result;
}

/** A wasm module's `setRequestHook`, kept up to date. */
export function connectModule(set: SetHook): void {
  hooks.add(set);
  set(transform);
}

/** A worker's port, told whether to ask the page; returns its release. */
export function connectWorker(port: Port): () => void {
  ports.add(port);
  if (transform) port.postMessage({ requestHook: true });
  return () => ports.delete(port);
}

/** Answers a worker's `{request, url}`; false for other messages. */
export function answerWorker(port: Port, data: unknown): boolean {
  const { request, url } = data as { request?: number; url?: string };
  if (request === undefined || url === undefined) return false;
  void Promise.resolve(transform?.(url)).then(
    (result) => port.postMessage({ request, result }),
    () => port.postMessage({ request, result: null }),
  );
  return true;
}

/** A worker's hook: the page's transform, asked over `port`. */
export function askPage(port: Port): {
  hook: (url: string) => Promise<unknown>;
  answer: (data: unknown) => boolean;
} {
  let next = 0;
  const waiting = new Map<number, (result: unknown) => void>();
  return {
    hook: (url) =>
      new Promise((resolve) => {
        const request = next++;
        waiting.set(request, resolve);
        port.postMessage({ request, url });
      }),
    answer: (data) => {
      const { request, result } = data as {
        request?: number;
        result?: unknown;
      };
      const resolve = request === undefined ? undefined : waiting.get(request);
      if (!resolve) return false;
      waiting.delete(request!);
      resolve(result);
      return true;
    },
  };
}
