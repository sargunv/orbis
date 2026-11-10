// mln-rs's browser build behind the `Map` that demo/load.js drives:
// `views.mjs`'s `mlnrs()` copies this file beside mln-rs's own Web Workers
// (map-worker.js, worker.js) and its wasm-bindgen package (pkg/), as
// index.js. It does what mln-rs's page (examples/web/web/main.js) does:
// compiles the module, starts its tile workers' Web Workers, each connected
// to the map's by a MessageChannel, and hands the map's Web Worker a canvas
// filling the container. The map loads in that worker, so `idle` is its
// one `loaded` message: the map fully loaded. Its worker takes no camera
// after the first, so there is no `jumpTo` (and no `tour`).

const here = (path) => new URL(path, import.meta.url);

export class Map {
  #errors = [];
  #loaded;

  constructor({
    container,
    style,
    center: [lon, lat],
    zoom,
    bearing = 0,
    pitch = 0,
  }) {
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "display: block; width: 100%; height: 100%";
    container.append(canvas);
    const ratio = window.devicePixelRatio || 1;
    const [width, height] = [canvas.clientWidth, canvas.clientHeight];
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    this.#loaded = new Promise((resolve, reject) => {
      const fail = (error) => {
        for (const listener of this.#errors) listener({ error });
        reject(error);
      };
      const watch = (worker) => {
        worker.onerror = (e) => fail(new Error(e.message));
        return worker;
      };
      WebAssembly.compileStreaming(fetch(here("./pkg/mln_web_bg.wasm")))
        .then((module) => {
          // main.js's default: up to 4, leaving a core for the page.
          const count = Math.max(
            1,
            Math.min(4, (navigator.hardwareConcurrency || 2) - 1),
          );
          const ports = Array.from({ length: count }, () => {
            const channel = new MessageChannel();
            watch(
              new Worker(here("./worker.js"), { type: "module" }),
            ).postMessage({ module, port: channel.port2 }, [channel.port2]);
            return channel.port1;
          });
          const map = watch(
            new Worker(here("./map-worker.js"), { type: "module" }),
          );
          map.onmessage = ({ data }) => {
            if (data.loaded) resolve(data.loaded);
          };
          const offscreen = canvas.transferControlToOffscreen();
          map.postMessage(
            {
              module,
              canvas: offscreen,
              ports,
              width,
              height,
              ratio,
              style: new URL(style, location.href).href,
              camera: [lat, lon, zoom, bearing, pitch],
            },
            [offscreen, ...ports],
          );
        })
        .catch(fail);
    });
  }

  on(type, listener) {
    if (type === "error") this.#errors.push(listener);
  }

  /** Resolves once the map is fully loaded: `{ loadedMs, replies }`. */
  once(type) {
    if (type !== "idle") throw new Error(`mln-rs: no ${type} event`);
    return this.#loaded;
  }

  jumpTo() {
    throw new Error("mln-rs's map worker takes no camera after the first");
  }
}
