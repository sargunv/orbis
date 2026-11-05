// The page's video player (decision 0039): the map asks for each video
// source's playback with `video` events, and each frame the element
// presents goes to the map, which copies the element's current frame into
// its texture (`copyExternalImageToTexture`).

/** What the map asks of the player. */
export type VideoEventObject =
  | { type: "video"; source: string; request: "open"; urls: string[] }
  | { type: "video"; source: string; request: "play" | "pause" | "close" }
  | { type: "video"; source: string; request: "seek"; seconds: number };

/** The core map's video calls. */
export interface VideoCore {
  videoFrame(source: string, video: HTMLVideoElement): void;
  videoFailed(source: string, message: string): void;
}

export class Videos {
  readonly #videos = new Map<string, HTMLVideoElement>();
  readonly #core: () => VideoCore | undefined;
  readonly #changed: () => void;

  /** `core()` is the map now (it changes when a lost GPU recovers);
   * `changed` asks for a frame. */
  constructor(core: () => VideoCore | undefined, changed: () => void) {
    this.#core = core;
    this.#changed = changed;
  }

  request(event: VideoEventObject): void {
    const video = this.#videos.get(event.source);
    switch (event.request) {
      case "open": this.#open(event.source, event.urls); break;
      case "play": void video?.play().catch(() => {}); break;
      case "pause": video?.pause(); break;
      case "seek": if (video) video.currentTime = event.seconds; break;
      case "close": this.#close(event.source); break;
    }
  }

  destroy(): void {
    for (const source of [...this.#videos.keys()]) this.#close(source);
  }

  #open(source: string, urls: string[]): void {
    this.#close(source);
    const video = document.createElement("video");
    video.crossOrigin = "anonymous";
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = "auto";
    // The browser plays the first source it can (GL JS `getVideo`).
    for (const url of urls) {
      const element = document.createElement("source");
      element.src = url;
      video.appendChild(element);
    }
    this.#videos.set(source, video);
    const current = () => this.#videos.get(source) === video;
    const frame = () => {
      if (!current()) return;
      try {
        this.#core()?.videoFrame(source, video);
        this.#changed();
      } catch {
        // The style dropped the source; its close is on the way.
      }
    };
    // Each presented frame, and the first one and a paused seek's.
    const presented = () => {
      frame();
      if (current()) video.requestVideoFrameCallback(presented);
    };
    video.requestVideoFrameCallback(presented);
    video.addEventListener("loadeddata", frame);
    video.addEventListener("seeked", frame);
    // The last source failing means none plays.
    video.lastElementChild?.addEventListener("error", () => {
      if (current()) this.#core()?.videoFailed(source, `none of ${urls.join(", ")} plays here`);
    });
    void video.play().catch(() => {});
  }

  #close(source: string): void {
    const video = this.#videos.get(source);
    if (!video) return;
    this.#videos.delete(source);
    video.pause();
    video.replaceChildren();
    video.load();
  }
}
