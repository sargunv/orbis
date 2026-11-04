// View annotations as DOM elements anchored to the map (plan §7.2): markers
// and popups. On the main thread they move in the frame the canvas draws;
// with a render worker, on each frame acknowledgement with that frame's
// camera, so they stay with what is on screen (±1 frame, measured in M3).

import type { LngLat, Point } from "./types.ts";

export type Anchor = "center" | "top" | "bottom" | "left" | "right" |
  "top-left" | "top-right" | "bottom-left" | "bottom-right";

const SHIFT: Record<Anchor, string> = {
  center: "-50%,-50%", top: "-50%,0", bottom: "-50%,-100%", left: "0,-50%", right: "-100%,-50%",
  "top-left": "0,0", "top-right": "-100%,0", "bottom-left": "0,-100%", "bottom-right": "-100%,-100%",
};

export interface AnnotationOptions {
  /** The element to show; an empty div by default. */
  element?: HTMLElement;
  lngLat: LngLat;
  /** Which part of the element sits on the point; "center" by default. */
  anchor?: Anchor;
  /** Pixels to shift it by. */
  offset?: Point;
}

/** One DOM annotation. */
export class Annotation {
  readonly element: HTMLElement;
  #lngLat: LngLat;
  #anchor: Anchor;
  #offset: Point;
  #layer: Annotations;

  constructor(layer: Annotations, options: AnnotationOptions) {
    this.#layer = layer;
    this.element = options.element ?? document.createElement("div");
    this.#lngLat = options.lngLat;
    this.#anchor = options.anchor ?? "center";
    this.#offset = options.offset ?? [0, 0];
    Object.assign(this.element.style, { position: "absolute", left: "0", top: "0", willChange: "transform" });
  }

  get lngLat(): LngLat { return this.#lngLat; }

  setLngLat(lngLat: LngLat): this {
    this.#lngLat = lngLat;
    this.#layer.changed();
    return this;
  }

  remove(): void { this.#layer.remove(this); }

  /** @internal */
  place(x: number, y: number): void {
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      this.element.style.visibility = "hidden";
      return;
    }
    const [dx, dy] = this.#offset;
    this.element.style.visibility = "";
    this.element.style.transform = `translate(${x + dx}px,${y + dy}px) translate(${SHIFT[this.#anchor]})`;
  }
}

/** A map's DOM annotations. */
export class Annotations {
  #overlay: HTMLElement;
  #items = new Set<Annotation>();
  #changed: () => void;

  /** @internal */
  constructor(overlay: HTMLElement, changed: () => void) {
    this.#overlay = overlay;
    this.#changed = changed;
  }

  add(options: AnnotationOptions): Annotation {
    const annotation = new Annotation(this, options);
    this.#items.add(annotation);
    this.#overlay.append(annotation.element);
    this.#changed();
    return annotation;
  }

  /** @internal */
  remove(annotation: Annotation): void {
    this.#items.delete(annotation);
    annotation.element.remove();
  }

  /** @internal */
  changed(): void { this.#changed(); }

  get size(): number { return this.#items.size; }

  /** @internal Positions every annotation with `project`: [lon, lat, …] → [x, y, …]. */
  place(project: (points: Float64Array) => Float64Array): void {
    if (!this.#items.size) return;
    const items = [...this.#items];
    const points = new Float64Array(items.length * 2);
    items.forEach((a, i) => { points[2 * i] = a.lngLat[0]; points[2 * i + 1] = a.lngLat[1]; });
    const pixels = project(points);
    items.forEach((a, i) => a.place(pixels[2 * i], pixels[2 * i + 1]));
  }
}
