# 0045: Video sources: events out, frames in

v8 `video` sources put a playing video on four corners. Image sources
(decision 0034) already draw any picture on four corners on every surface.
What video adds is a decoder, and the core has none. This records how
hosts supply the frames.

## Design

A `video` source parses as an image source whose content is a list of URLs
(`orb_v8::ImageContent::Video`). Its one tile is filled from frames instead
of a fetch. There is no host-service trait. The map and its host talk
through the `Map` API, as they do for everything else:

- **Events out.** On a host that plays video (`Map::set_plays_video`), the
  map emits `MapEvent::Video { source, request }`. `Open { urls }` comes
  when a source appears: play the first URL the host can, muted, looping,
  from the start, as GL JS does. `Play`, `Pause` and `Seek` come from
  `Map::control_video` (GL JS's `VideoSource` methods), and `Close` when the
  source goes.
- **Frames in.** The host calls `Map::video_frame(source, frame)` with each
  new frame, or `Map::video_failed` when nothing plays. A frame is
  straight-alpha RGBA rows, or on the web an image held in JS. The newest
  frame fills the tile when it is requested. After that, each frame is
  written over the tile's texture layer in place. The map keeps the newest
  frame for a tile requested again.
- **Hosts without video.** On such a host each video source fails with a
  warning and draws nothing. The tile resolves empty, so the map still
  idles.

Frames wake the map the way any `Map` call does. A playing video costs a
frame per video frame, not one per display refresh, and a paused one
costs nothing.

## Per platform

- **Web.** `web/src/videos.ts` gives each source an `HTMLVideoElement` with
  a `<source>` per URL, so the browser picks what it plays. On each
  presented frame (`requestVideoFrameCallback`), plus `loadeddata` and
  `seeked`, it passes the element to `videoFrame`. The HAL registers the
  element as a host image, and the next frame copies its current picture
  with `copyExternalImageToTexture` (WebGPU), or through a 2D canvas on
  WebGL2. The map draws on the page's thread (decision 0040), where the
  elements are.
- **Apple.** `MapView` plays each source with `AVPlayer` and an
  `AVPlayerItemVideoOutput`. While a video plays, the display link pulls
  new pixel buffers and pushes them as RGBA. AVFoundation reads MP4 and
  QuickTime, not WebM.
- **Android.** `MapView` plays each source with `MediaPlayer` (MediaCodec
  underneath) into a YUV `ImageReader` on a video thread. Frames are
  converted to RGBA there and pushed on the main thread. Android reads MP4
  and WebM.
- **Desktop and in-process Rust hosts** (`orb-demo`, native fixtures) don't
  play video (#77). A decoder dependency wasn't worth its weight for that.
  An embedder can play video itself through the same events and calls.

## Results

GL JS's `video/default` (WebM, VP8) passes in Chrome. Natively it skips as
`host:video-decoder`. A runtime test drives the protocol with RGBA frames:
the first frame fills the tile, the next replaces it in place, and requests
come out as events.
