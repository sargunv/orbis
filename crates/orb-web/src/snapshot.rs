//! Still output in the browser: the camera's view drawn offscreen once
//! everything in it has loaded and nothing fades (orb-api's
//! `Map::snapshot`). The page drives it a frame at a time, so tile replies
//! keep arriving between frames: [`OrbisMap::begin_snapshot`], then
//! [`OrbisMap::snapshot_frame`] until it is settled, then
//! [`OrbisMap::end_snapshot`] for the pixels.

use orb_camera::Viewport;
use orb_hal::{TextureDesc, TextureFormat, TextureId, TextureUsages, TextureViewDimension};
use orb_runtime::RenderMode;
use wasm_bindgen::prelude::*;

use crate::OrbisMap;

/// A snapshot being drawn.
#[derive(Debug)]
pub(crate) struct Snapshot {
    texture: TextureId,
    viewport: Viewport,
    scale: f32,
    /// Whether the last frame was idle: the next idle one is settled.
    idle: bool,
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
        });
        self.map.set_mode(RenderMode::Still);
        self.snapshot = Some(Snapshot {
            texture,
            viewport: Viewport {
                width: f64::from(width),
                height: f64::from(height),
            },
            scale: scale as f32,
            idle: false,
        });
    }

    /// Draws a frame of the snapshot; whether it is settled: two idle
    /// frames in a row, the second drawing what the first's replies
    /// brought. Wait a little for data between calls.
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
            orb_host::Clock::now(&orb_host_web::WebClock),
        );
        let settled = report.idle && snapshot.idle;
        snapshot.idle = report.idle;
        Ok(settled)
    }

    /// Ends the snapshot: resolves to its pixels, straight (not
    /// premultiplied) RGBA8 rows top to bottom, as `ImageData` holds them.
    #[wasm_bindgen(js_name = endSnapshot)]
    pub fn end_snapshot(&mut self) -> Result<js_sys::Promise, JsError> {
        let snapshot = self
            .snapshot
            .take()
            .ok_or_else(|| JsError::new("no snapshot started"))?;
        let bgra = matches!(
            self.canvas.format().color,
            TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb
        );
        let gpu = self.canvas.gpu();
        let pixels = gpu.read_texture(snapshot.texture);
        // The copy is submitted: the texture can go.
        gpu.destroy_texture(snapshot.texture);
        self.map.set_mode(RenderMode::Continuous);
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            let mut pixels = pixels.await;
            for px in pixels.as_chunks_mut::<4>().0 {
                if bgra {
                    px.swap(0, 2);
                }
                let a = u32::from(px[3]);
                if a != 0 && a != 255 {
                    for c in &mut px[..3] {
                        *c = ((u32::from(*c) * 255 + a / 2) / a).min(255) as u8;
                    }
                }
            }
            Ok(js_sys::Uint8Array::from(pixels.as_slice()).into())
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
