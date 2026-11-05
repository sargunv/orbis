//! WebGL2: contexts on canvases, and the host's bytes and images.
//!
//! Host bytes are copied into wasm memory when written (WebGL2 has no
//! `writeBuffer` from an `ArrayBuffer` the executor could reach through
//! glow), and `ImageBitmap`s are read through a 2D canvas, since WebGL
//! ignores premultiplication for them.

use orb_hal::{HostBytesId, HostImageId};
use wasm_bindgen::JsCast;

use crate::{GlGpu, HostBytes};

/// The attributes Orbis's canvases get: premultiplied alpha, no depth,
/// stencil or antialiasing (frames are blitted in), and `failIfMajorPerformanceCaveat`
/// off, so software WebGL still draws a map.
fn attributes() -> web_sys::WebGlContextAttributes {
    let attributes = web_sys::WebGlContextAttributes::new();
    attributes.set_alpha(true);
    attributes.set_premultiplied_alpha(true);
    attributes.set_antialias(false);
    attributes.set_depth(false);
    attributes.set_stencil(false);
    attributes.set_preserve_drawing_buffer(false);
    attributes
}

/// A WebGL2 context on a canvas or offscreen canvas.
fn context(canvas: &wasm_bindgen::JsValue) -> Result<web_sys::WebGl2RenderingContext, String> {
    let context = if let Some(canvas) = canvas.dyn_ref::<web_sys::HtmlCanvasElement>() {
        canvas.get_context_with_context_options("webgl2", &attributes())
    } else if let Some(canvas) = canvas.dyn_ref::<web_sys::OffscreenCanvas>() {
        canvas.get_context_with_context_options("webgl2", &attributes())
    } else {
        return Err("not a canvas".into());
    }
    .map_err(|e| format!("webgl2: {e:?}"))?
    .ok_or("no WebGL2")?;
    context
        .dyn_into::<web_sys::WebGl2RenderingContext>()
        .map_err(|_| "not a WebGL2 context".to_string())
}

impl GlGpu {
    /// The executor over a new WebGL2 context on an `HTMLCanvasElement` or
    /// `OffscreenCanvas`.
    pub fn from_canvas(canvas: &wasm_bindgen::JsValue) -> Result<Self, String> {
        Self::from_webgl2(context(canvas)?)
    }

    /// The executor over a WebGL2 context.
    pub fn from_webgl2(context: web_sys::WebGl2RenderingContext) -> Result<Self, String> {
        if context.is_context_lost() {
            return Err("the WebGL2 context is lost".into());
        }
        // Unmasked renderer names, where the browser offers them.
        let _ = context.get_extension("WEBGL_debug_renderer_info");
        let _ = context.get_extension("EXT_color_buffer_float");
        let mut gpu = Self::new(glow::Context::from_webgl2_context(context.clone()))?;
        gpu.webgl = Some(context);
        Ok(gpu)
    }

    /// The WebGL2 context underneath.
    pub fn webgl2(&self) -> Option<&web_sys::WebGl2RenderingContext> {
        self.webgl.as_ref()
    }

    /// Loses the context on purpose (`WEBGL_lose_context`), to exercise
    /// recovery.
    pub fn lose_context(&self) {
        let lose = self
            .webgl
            .as_ref()
            .and_then(|gl| gl.get_extension("WEBGL_lose_context").ok().flatten());
        if let Some(lose) = lose {
            let _ = js_sys::Reflect::get(&lose, &"loseContext".into())
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .map(|f| f.call0(&lose));
        }
    }

    /// A transferred `ArrayBuffer`, for [`orb_hal::Gpu::write_buffer_from_host`].
    pub fn register_array_buffer(&mut self, buffer: js_sys::ArrayBuffer) -> HostBytesId {
        HostBytesId(self.host_bytes.insert(HostBytes::Js(buffer)))
    }

    /// A decoded image, for [`orb_hal::Gpu::write_texture_from_host`].
    pub fn register_image_bitmap(&mut self, bitmap: web_sys::ImageBitmap) -> HostImageId {
        HostImageId(self.host_images.insert(bitmap))
    }
}

/// An image's RGBA8 pixels, premultiplied if asked.
pub(crate) fn read_bitmap(bitmap: &web_sys::ImageBitmap, premultiply: bool) -> Vec<u8> {
    let (width, height) = (bitmap.width(), bitmap.height());
    let pixels = (|| -> Result<Vec<u8>, wasm_bindgen::JsValue> {
        let canvas = web_sys::OffscreenCanvas::new(width, height)?;
        let context = canvas
            .get_context("2d")?
            .ok_or("no 2d context")?
            .dyn_into::<web_sys::OffscreenCanvasRenderingContext2d>()?;
        context.draw_image_with_image_bitmap(bitmap, 0.0, 0.0)?;
        Ok(context
            .get_image_data(0.0, 0.0, f64::from(width), f64::from(height))?
            .data()
            .0)
    })();
    let mut pixels = pixels.unwrap_or_else(|_| vec![0; (width * height * 4) as usize]);
    if premultiply {
        for px in pixels.as_chunks_mut::<4>().0 {
            let a = u16::from(px[3]);
            for c in &mut px[..3] {
                *c = ((u16::from(*c) * a + 127) / 255) as u8;
            }
        }
    }
    pixels
}
