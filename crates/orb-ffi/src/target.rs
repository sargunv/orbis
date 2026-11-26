// Render targets: the `mln_render_target` an attach or a replacement names,
// checked as every build checks it, whatever kinds its devices render to,
// before the kinds this build lacks are refused.

use std::ffi::c_void;

use crate::base::{Failure, Result, UserData, sized};
use crate::gpu_device::{TARGET_KINDS, kind_flag};
use crate::session::TargetSpec;
use crate::sys::*;

/// A target whose descriptor checks out.
pub(crate) struct Described {
    pub(crate) kind: mln_render_target_kind,
    pub(crate) logical: mln_logical_size,
    shape: Shape,
}

/// What a kind Orbis renders to names.
enum Shape {
    Offscreen,
    /// A CAMetalLayer.
    MetalLayer(UserData),
    /// The host's MTLTextures, of this physical size.
    MetalTextures {
        textures: Vec<UserData>,
        physical: (u32, u32),
    },
    /// A kind no build of Orbis renders to.
    Other,
}

impl Described {
    /// What a session renders into, or why no device of this build renders
    /// to the target.
    pub(crate) fn spec(&self) -> Result<TargetSpec> {
        if TARGET_KINDS & kind_flag(self.kind) == 0 {
            return Err(Failure::unsupported(format!(
                "a target of kind {} is not supported by this build: its devices render \
                 offscreen{}",
                self.kind,
                if cfg!(target_vendor = "apple") {
                    ", to Metal layers and into Metal textures"
                } else {
                    ""
                }
            )));
        }
        match &self.shape {
            Shape::Offscreen => Ok(TargetSpec::Offscreen),
            Shape::MetalLayer(layer) => Ok(TargetSpec::Metal(*layer)),
            Shape::MetalTextures { textures, physical } => {
                crate::frames::metal_slots(textures, self.logical, *physical).map(TargetSpec::Ring)
            }
            Shape::Other => unreachable!("TARGET_KINDS names no other kind"),
        }
    }
}

fn required(object: *mut c_void, what: &str) -> Result {
    match object.is_null() {
        true => Err(Failure::argument(format!("target.{what} must not be null"))),
        false => Ok(()),
    }
}

/// Checks the rules every ring of the host's textures shares: a positive
/// physical size, one to `MLN_TEXTURE_RING_MAX_DEPTH` textures, and no
/// texture twice, by the identity `id` reads of each entry once it checks
/// the objects the entry names.
///
/// # Safety
/// `textures` is null or holds `count` entries.
unsafe fn ring<'a, T, I: PartialEq>(
    what: &str,
    physical: (u32, u32),
    textures: *const T,
    count: usize,
    id: impl Fn(usize, &T) -> Result<I>,
) -> Result<&'a [T]> {
    if physical.0 == 0 || physical.1 == 0 {
        return Err(Failure::argument(format!(
            "target.{what}: the physical texture dimensions must be positive"
        )));
    }
    if textures.is_null() || count == 0 {
        return Err(Failure::argument(format!(
            "target.{what} takes at least one texture"
        )));
    }
    if count > MLN_TEXTURE_RING_MAX_DEPTH as usize {
        return Err(Failure::argument(format!(
            "target.{what} holds at most MLN_TEXTURE_RING_MAX_DEPTH \
             ({MLN_TEXTURE_RING_MAX_DEPTH}) textures"
        )));
    }
    // SAFETY: the caller's contract.
    let textures = unsafe { std::slice::from_raw_parts(textures, count) };
    let ids = textures
        .iter()
        .enumerate()
        .map(|(i, t)| id(i, t))
        .collect::<Result<Vec<_>>>()?;
    if ids.iter().enumerate().any(|(i, a)| ids[..i].contains(a)) {
        return Err(Failure::argument(format!(
            "target.{what} takes distinct textures"
        )));
    }
    Ok(textures)
}

/// Checks the target `target` points at.
///
/// # Safety
/// `target` is null or a readable `mln_render_target`, whose pointers to
/// arrays hold the entries their counts say.
pub(crate) unsafe fn describe(target: *const mln_render_target) -> Result<Described> {
    // SAFETY: the caller's pointer.
    let t = unsafe { sized(target, "target")? };
    let mut shape = Shape::Other;
    // SAFETY: `kind` tags the union's variant, and the arrays hold their
    // counts, by the C API's contract.
    let logical = unsafe {
        let data = &t.data;
        match t.kind {
            MLN_RENDER_TARGET_KIND_OFFSCREEN => {
                shape = Shape::Offscreen;
                data.offscreen.logical_size
            }
            MLN_RENDER_TARGET_KIND_METAL_LAYER => {
                let d = data.metal_layer;
                required(d.layer, "metal_layer.layer")?;
                shape = Shape::MetalLayer(UserData(d.layer));
                d.logical_size
            }
            MLN_RENDER_TARGET_KIND_WIN32_WINDOW => {
                let d = data.win32_window;
                required(d.window, "win32_window.window")?;
                d.logical_size
            }
            MLN_RENDER_TARGET_KIND_XLIB_WINDOW => {
                let d = data.xlib_window;
                required(d.display, "xlib_window.display")?;
                if d.window == 0 {
                    return Err(Failure::argument(
                        "target.xlib_window.window must not be null",
                    ));
                }
                d.logical_size
            }
            MLN_RENDER_TARGET_KIND_WAYLAND_SURFACE => {
                let d = data.wayland_surface;
                required(d.display, "wayland_surface.display")?;
                required(d.surface, "wayland_surface.surface")?;
                d.logical_size
            }
            MLN_RENDER_TARGET_KIND_ANDROID_WINDOW => {
                let d = data.android_window;
                required(d.window, "android_window.window")?;
                d.logical_size
            }
            MLN_RENDER_TARGET_KIND_OHOS_WINDOW => {
                let d = data.ohos_window;
                required(d.window, "ohos_window.window")?;
                d.logical_size
            }
            MLN_RENDER_TARGET_KIND_WEB_GPU_SURFACE => {
                let d = data.web_gpu_surface;
                required(d.surface, "web_gpu_surface.surface")?;
                if d.format == 0 {
                    return Err(Failure::argument(
                        "target.web_gpu_surface.format must name a texture format",
                    ));
                }
                d.logical_size
            }
            MLN_RENDER_TARGET_KIND_HTML_CANVAS => {
                let d = data.html_canvas;
                if d.selector.data.is_null() || d.selector.size == 0 {
                    return Err(Failure::argument(
                        "target.html_canvas.selector must name a canvas",
                    ));
                }
                d.logical_size
            }
            MLN_RENDER_TARGET_KIND_VULKAN_SURFACE => {
                let d = data.vulkan_surface;
                if d.surface == 0 {
                    return Err(Failure::argument(
                        "target.vulkan_surface.surface must not be null",
                    ));
                }
                d.logical_size
            }
            MLN_RENDER_TARGET_KIND_OPEN_GL_SURFACE => {
                let d = data.open_gl_surface;
                required(d.surface, "open_gl_surface.surface")?;
                d.logical_size
            }
            MLN_RENDER_TARGET_KIND_METAL_BORROWED_TEXTURES => {
                let d = data.metal_borrowed_textures;
                let physical = (d.physical_width, d.physical_height);
                let what = "metal_borrowed_textures";
                let textures = ring(what, physical, d.textures, d.texture_count, |i, t| {
                    required(t.texture, &format!("{what}.textures[{i}].texture"))?;
                    Ok(t.texture)
                })?;
                shape = Shape::MetalTextures {
                    textures: textures.iter().map(|t| UserData(t.texture)).collect(),
                    physical,
                };
                d.logical_size
            }
            MLN_RENDER_TARGET_KIND_VULKAN_BORROWED_TEXTURES => {
                let d = data.vulkan_borrowed_textures;
                let physical = (d.physical_width, d.physical_height);
                let what = "vulkan_borrowed_textures";
                ring(what, physical, d.textures, d.texture_count, |i, t| {
                    if t.image == 0 || t.image_view == 0 {
                        return Err(Failure::argument(format!(
                            "target.{what}.textures[{i}] needs an image and an image view"
                        )));
                    }
                    Ok(t.image)
                })?;
                if d.format == 0 || d.final_layout == 0 {
                    return Err(Failure::argument(format!(
                        "target.{what} needs a format and final_layout"
                    )));
                }
                if d.owner > MLN_VULKAN_IMAGE_OWNER_FOREIGN {
                    return Err(Failure::argument(format!("target.{what}.owner is invalid")));
                }
                d.logical_size
            }
            MLN_RENDER_TARGET_KIND_OPEN_GL_BORROWED_TEXTURES => {
                let d = data.open_gl_borrowed_textures;
                let physical = (d.physical_width, d.physical_height);
                let what = "open_gl_borrowed_textures";
                ring(what, physical, d.textures, d.texture_count, |i, t| {
                    if t.texture == 0 {
                        return Err(Failure::argument(format!(
                            "target.{what}.textures[{i}].texture must name a texture"
                        )));
                    }
                    Ok(t.texture)
                })?;
                if d.target == 0 {
                    return Err(Failure::argument(format!(
                        "target.{what}.target must name the textures' target, such as \
                         GL_TEXTURE_2D"
                    )));
                }
                d.logical_size
            }
            MLN_RENDER_TARGET_KIND_WEB_GPU_BORROWED_TEXTURES => {
                let d = data.web_gpu_borrowed_textures;
                let physical = (d.physical_width, d.physical_height);
                let what = "web_gpu_borrowed_textures";
                ring(what, physical, d.textures, d.texture_count, |i, t| {
                    if t.texture.is_null() || t.texture_view.is_null() {
                        return Err(Failure::argument(format!(
                            "target.{what}.textures[{i}] needs a texture and a texture view"
                        )));
                    }
                    Ok(t.texture)
                })?;
                if d.format == 0 {
                    return Err(Failure::argument(format!(
                        "target.{what}.format must name a texture format"
                    )));
                }
                d.logical_size
            }
            MLN_RENDER_TARGET_KIND_EXPORTED_RING => {
                let d = data.exported_ring;
                if d.texture_count == 0 || d.texture_count > MLN_TEXTURE_RING_MAX_DEPTH {
                    return Err(Failure::argument(format!(
                        "target.exported_ring.texture_count must be from 1 to \
                         MLN_TEXTURE_RING_MAX_DEPTH ({MLN_TEXTURE_RING_MAX_DEPTH})"
                    )));
                }
                let known = (MLN_GPU_EXPORT_KIND_FLAG_OPAQUE_FD << 1) - 1;
                if d.export_kinds == 0 || d.export_kinds & !known != 0 {
                    return Err(Failure::argument(
                        "target.exported_ring.export_kinds must name ways of sharing a texture, \
                         and only those",
                    ));
                }
                if d.dma_buf_format_count > 0 && d.dma_buf_formats.is_null() {
                    return Err(Failure::argument(
                        "target.exported_ring.dma_buf_formats is null but its count isn't zero",
                    ));
                }
                if d.export_kinds & MLN_GPU_EXPORT_KIND_FLAG_DMA_BUF != 0
                    && d.dma_buf_format_count == 0
                {
                    return Err(Failure::argument(
                        "target.exported_ring exports dmabufs, so it needs dma_buf_formats",
                    ));
                }
                d.logical_size
            }
            _ => {
                return Err(Failure::argument("target.kind names no render target kind"));
            }
        }
    };
    if logical.width == 0 || logical.height == 0 {
        return Err(Failure::argument(
            "the target's logical width and height must be positive",
        ));
    }
    Ok(Described {
        kind: t.kind,
        logical,
        shape,
    })
}
