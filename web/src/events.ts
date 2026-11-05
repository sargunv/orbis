// A typed event emitter with GL JS-style `on`/`off`/`once` and an async
// iterator per event type (plan §14.3: `for await (const e of map.events("idle"))`).

export type Listener<T> = (event: T) => void;

export class Emitter<Events> {
  #listeners = new Map<keyof Events, Set<Listener<never>>>();

  on<K extends keyof Events>(type: K, listener: Listener<Events[K]>): () => void {
    let set = this.#listeners.get(type);
    if (!set) this.#listeners.set(type, (set = new Set()));
    set.add(listener as Listener<never>);
    return () => this.off(type, listener);
  }

  off<K extends keyof Events>(type: K, listener: Listener<Events[K]>): void {
    this.#listeners.get(type)?.delete(listener as Listener<never>);
  }

  once<K extends keyof Events>(type: K): Promise<Events[K]> {
    return new Promise((resolve) => {
      const off = this.on(type, (event) => { off(); resolve(event); });
    });
  }

  /** Every event of `type` from now on; stops listening when the loop ends. */
  async *events<K extends keyof Events>(type: K): AsyncGenerator<Events[K]> {
    const queue: Events[K][] = [];
    let wake: (() => void) | null = null;
    const off = this.on(type, (event) => { queue.push(event); wake?.(); });
    try {
      for (;;) {
        while (queue.length) yield queue.shift() as Events[K];
        await new Promise<void>((resolve) => { wake = resolve; });
        wake = null;
      }
    } finally {
      off();
    }
  }

  emit<K extends keyof Events>(type: K, event: Events[K]): void {
    const set = this.#listeners.get(type);
    if (!set) return;
    for (const listener of [...set]) (listener as Listener<Events[K]>)(event);
  }

  has(type: keyof Events): boolean {
    return (this.#listeners.get(type)?.size ?? 0) > 0;
  }
}
