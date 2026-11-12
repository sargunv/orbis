//! The browser build of Orbis: `core.wasm` and its wasm-bindgen surface.
//!
//! One module, run by the web package's JS in three roles:
//! - **the map** ([`create_map`]): the map draws into a canvas, with its
//!   camera, gestures and queries: an `OffscreenCanvas` in a map worker
//!   where the browser allows it, else a canvas on the page's thread.
//! - **job workers** ([`jobs::JobWorker`]) fetch and lay out tiles.
//! - **a placement worker** ([`PlacementWorker`]) places labels.
//!
//! The map draws through WebGPU; the `host` module gives every role the
//! browser's fetch, storage, image decoding and canvas glyphs. Which job
//! worker each request goes to (`dispatch`) is plain Rust, built natively
//! for its tests.

#[cfg(any(target_arch = "wasm32", test))]
mod dispatch;

#[cfg(target_arch = "wasm32")]
mod canvas;
#[cfg(all(target_arch = "wasm32", feature = "gl"))]
mod gl_context;
#[cfg(target_arch = "wasm32")]
mod host;
#[cfg(target_arch = "wasm32")]
mod jobs;
#[cfg(target_arch = "wasm32")]
mod js;
#[cfg(target_arch = "wasm32")]
mod map;
#[cfg(target_arch = "wasm32")]
mod placement;
#[cfg(target_arch = "wasm32")]
pub mod regions;

#[cfg(target_arch = "wasm32")]
pub use jobs::JobWorker;
#[cfg(target_arch = "wasm32")]
pub use map::{FrameGpu, FrameMemory, FrameStats, FrameTimes, OrbisMap, create_map};
#[cfg(target_arch = "wasm32")]
pub use placement::PlacementWorker;

/// Panics say why on the console in a build with `panic-messages`
/// (`cargo xtask web --panics`), in every instance: the map's and its
/// workers'.
#[cfg(all(target_arch = "wasm32", feature = "panic-messages"))]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
fn panic_messages() {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&format!("orbis panicked: {info}").into());
    }));
}
