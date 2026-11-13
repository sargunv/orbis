// The wasm map, stopping at its first trap with one clear error rather
// than a borrow held forever (hosts.ts).

/**
 * `core`, stopping at its first trap. A Rust panic traps the module
 * (`-Cpanic=immediate-abort`) partway through a call, leaving the object
 * borrowed and its memory half-written: called again, every method would
 * throw "recursive use of an object". The trapping call throws an error
 * that says what happened instead, `trapped` hears it once, every call
 * after throws it too without entering the module, and an answer a call
 * still waits on (the next frame's pixels, a snapshot's end) rejects with
 * it. `stop` stops it for a trap outside a call.
 */
export function guard<Core extends object>(
  core: Core,
  trapped: (e: Error) => void,
): { core: Core; stop: (e: unknown) => Error } {
  let stopped: Error | undefined;
  let halt: (e: Error) => void = () => {};
  const halted = new Promise<never>((_, reject) => (halt = reject));
  halted.catch(() => {});
  const stop = (e: unknown): Error => {
    if (stopped) return stopped;
    console.error(e);
    stopped = trapError(e);
    halt(stopped);
    trapped(stopped);
    return stopped;
  };
  // A method's guarded self, made once: frames call many.
  const methods = new Map<PropertyKey, (...args: unknown[]) => unknown>();
  const guarded = new Proxy(core, {
    get(target, key) {
      const value: unknown = Reflect.get(target, key, target);
      if (typeof value !== "function") return value;
      let method = methods.get(key);
      if (!method) {
        method = (...args) => {
          if (stopped) throw stopped;
          try {
            const result: unknown = value.apply(target, args);
            // The frames that would settle it stop with the map.
            return result instanceof Promise
              ? Promise.race([result, halted])
              : result;
          } catch (e) {
            throw e instanceof WebAssembly.RuntimeError ? stop(e) : e;
          }
        };
        methods.set(key, method);
      }
      return method;
    },
  });
  return { core: guarded, stop };
}

/** What a map whose wasm trapped throws, the trap its cause. */
export function trapError(e: unknown): Error {
  const why = e instanceof Error ? e.message : String(e);
  return new Error(
    e instanceof ModuleNotLoadedError
      ? `the map stopped: ${why}`
      : `the map stopped: its wasm trapped (${why}), a bug in Orbis`,
    { cause: e },
  );
}

/** Whether `e` is a trap of the map's wasm: a bug, which a map on
 * another thread would hit too. */
export function isTrap(e: unknown): boolean {
  return (
    e instanceof WebAssembly.RuntimeError ||
    (e instanceof Error && e.cause instanceof WebAssembly.RuntimeError)
  );
}

/** A call into a feature module this thread hasn't loaded (builds.ts):
 * a bug in the split's paths or in the map's gate. The Rust frames under
 * it can't unwind, so the instance is as unusable as after a trap, and is
 * one: a helper worker reports itself lost and closes, and a map stops. */
export class ModuleNotLoadedError extends WebAssembly.RuntimeError {
  /** `slot` is the placeholder's, which `cargo xtask web` names in
   * `target/web/<build>-whole.placeholders`. */
  constructor(name: string, slot?: string) {
    const which = slot === undefined ? "" : ` (placeholder ${slot})`;
    super(
      `it called into core-${name}.wasm${which} before loading it, a bug in Orbis`,
    );
    this.name = "ModuleNotLoadedError";
  }
}
