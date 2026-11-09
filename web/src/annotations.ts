// View annotations as DOM elements anchored to the map: markers
// and popups. They move with the frames the map reports, on the terrain.

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
  /** Where it was last placed, to write the DOM only when that changes. */
  #at: Point | null | undefined;

  constructor(layer: Annotations, options: AnnotationOptions) {
    this.#layer = layer;
    this.element = options.element ?? document.createElement("div");
    this.#lngLat = options.lngLat;
    this.#anchor = options.anchor ?? "center";
    this.#offset = options.offset ?? [0, 0];
    Object.assign(this.element.style, { position: "absolute", left: "0", top: "0", willChange: "transform" });
  }

  get lngLat(): LngLat { return this.#lngLat; }

  /** The pixel it was last placed at, null where the map doesn't show its
   * point, or undefined before the map has drawn. */
  get point(): Point | null | undefined { return this.#at && [this.#at[0], this.#at[1]]; }

  setLngLat(lngLat: LngLat): this {
    this.#lngLat = lngLat;
    this.#layer.changed();
    return this;
  }

  remove(): void { this.#layer.remove(this); }

  /** @internal */
  place(x: number, y: number): void {
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      if (this.#at !== null) this.element.style.visibility = "hidden";
      this.#at = null;
      return;
    }
    if (this.#at === null) this.element.style.visibility = "";
    if (this.#at && this.#at[0] === x && this.#at[1] === y) return;
    this.#at = [x, y];
    const [dx, dy] = this.#offset;
    this.element.style.transform = `translate(${x + dx}px,${y + dy}px) translate(${SHIFT[this.#anchor]})`;
  }
}

/** A map's DOM annotations. */
export class Annotations {
  #overlay: HTMLElement;
  #items = new Set<Annotation>();
  #changed: (points: Float64Array) => void;

  /** @internal `changed` hears the points, [lon, lat, …], when they change. */
  constructor(overlay: HTMLElement, changed: (points: Float64Array) => void) {
    this.#overlay = overlay;
    this.#changed = changed;
  }

  add(options: AnnotationOptions): Annotation {
    const annotation = new Annotation(this, options);
    this.#items.add(annotation);
    this.#overlay.append(annotation.element);
    this.changed();
    return annotation;
  }

  /** @internal */
  remove(annotation: Annotation): void {
    this.#items.delete(annotation);
    annotation.element.remove();
    this.changed();
  }

  /** @internal */
  changed(): void {
    const points = new Float64Array(this.#items.size * 2);
    let i = 0;
    for (const a of this.#items) {
      points[i++] = a.lngLat[0];
      points[i++] = a.lngLat[1];
    }
    this.#changed(points);
  }

  get size(): number { return this.#items.size; }

  /** @internal Positions every annotation at its pixel, [x, y, …] in the
   * order of the points last given; pixels for other points are dropped. */
  place(pixels: Float64Array): void {
    if (pixels.length !== this.#items.size * 2) return;
    let i = 0;
    for (const a of this.#items) {
      a.place(pixels[i], pixels[i + 1]);
      i += 2;
    }
  }
}
