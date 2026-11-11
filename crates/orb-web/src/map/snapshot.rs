//! Still output in the browser: the camera's view drawn offscreen once
//! everything in it has loaded and nothing fades (orb-api's
//! `Map::snapshot`). The page drives it a frame at a time, so tile replies
//! keep arriving between frames: [`OrbisMap::begin_snapshot`], then
//! [`OrbisMap::snapshot_frame`] until it is settled, then
//! [`OrbisMap::end_snapshot`] for the pixels.

use orb_camera::Viewport;
use orb_hal::{Gpu, TextureDesc, TextureFormat, TextureId, TextureUsages, TextureViewDimension};
use orb_runtime::RenderMode;
use wasm_bindgen::prelude::*;

use crate::OrbisMap;

/// A snapshot being drawn.
#[derive(Debug)]
pub(crate) struct Snapshot {
    texture: TextureId,
    viewport: Viewport,
    scale: f32,
    /// Its frames so far.
    settle: orb_runtime::Settle,
}

#[wasm_bindgen]
impl OrbisMap {
    /// Starts a snapshot of the camera's view, `width × height` CSS pixels
    /// at `scale` device pixels each. The page draws no frames of its own
    /// until it ends.
    #[wasm_bindgen(js_name = beginSnapshot)]
    pub fn begin_snapshot(&mut self, width: u32, height: u32, scale: f64) {
        self.cancel_snapshot();
        let px = |v: u32| ((f64::from(v) * scale).round() as u32).max(1);
        let format = self.canvas.format();
        let texture = self.canvas.gpu().create_texture(&TextureDesc {
            label: "snapshot",
            width: px(width),
            height: px(height),
            array_layers: 1,
            dimension: TextureViewDimension::D2,
            format: format.color,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            sample_count: 1,
            mip_levels: 1,
            view_formats: &[],
        });
        self.map.set_mode(RenderMode::Still);
        self.snapshot = Some(Snapshot {
            texture,
            viewport: Viewport {
                width: f64::from(width),
                height: f64::from(height),
            },
            scale: scale as f32,
            settle: orb_runtime::Settle::default(),
        });
    }

    /// Draws a frame of the snapshot; whether it is settled
    /// ([`orb_runtime::Settle`]), or an error at once for a style that
    /// failed to load. Wait a little for data between calls.
    #[wasm_bindgen(js_name = snapshotFrame)]
    pub fn snapshot_frame(&mut self) -> Result<bool, JsError> {
        let snapshot = self
            .snapshot
            .as_mut()
            .ok_or_else(|| JsError::new("no snapshot started"))?;
        if self.canvas.lost() {
            return Err(JsError::new("the GPU was lost"));
        }
        let format = self.canvas.format();
        let (map, gpu) = (&mut self.map, self.canvas.gpu());
        let renderer = self
            .renderer
            .get_or_insert_with(|| map.create_renderer(gpu, format));
        let report = map.render(
            gpu,
            renderer,
            snapshot.texture,
            snapshot.viewport,
            snapshot.scale,
            orb_host::Clock::now(&crate::host::WebClock),
        );
        // A style that failed to load never settles: say why at once.
        snapshot
            .settle
            .frame(map, Some(&report))
            .map_err(|message| JsError::new(&message))
    }

    /// Ends the snapshot: resolves to its pixels, straight (not
    /// premultiplied) RGBA8 rows top to bottom, as `ImageData` holds them.
    #[wasm_bindgen(js_name = endSnapshot)]
    pub fn end_snapshot(&mut self) -> Result<js_sys::Promise, JsError> {
        let snapshot = self
            .snapshot
            .take()
            .ok_or_else(|| JsError::new("no snapshot started"))?;
        let bgra = bgra(self.canvas.format().color);
        let gpu = self.canvas.gpu();
        let pixels = gpu.read_texture(snapshot.texture);
        // The copy is submitted: the texture can go.
        gpu.destroy_texture(snapshot.texture);
        self.map.set_mode(RenderMode::Continuous);
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            Ok(rgba(pixels.await, bgra, true).into())
        }))
    }

    /// Ends a snapshot without its pixels.
    #[wasm_bindgen(js_name = cancelSnapshot)]
    pub fn cancel_snapshot(&mut self) {
        if let Some(snapshot) = self.snapshot.take() {
            self.canvas.gpu().destroy_texture(snapshot.texture);
            self.map.set_mode(RenderMode::Continuous);
        }
    }
}

/// Whether a target of `format` holds its bytes blue first.
pub(crate) fn bgra(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb
    )
}

/// A target's premultiplied pixels as RGBA: straight, as `ImageData` holds
/// them ([`orb_runtime::unpremultiply`]), or premultiplied as they are.
pub(crate) fn rgba(mut pixels: Vec<u8>, bgra: bool, straight: bool) -> js_sys::Uint8Array {
    if bgra {
        for px in pixels.as_chunks_mut::<4>().0 {
            px.swap(0, 2);
        }
    }
    if straight {
        orb_runtime::unpremultiply(&mut pixels);
    }
    js_sys::Uint8Array::from(pixels.as_slice())
}
