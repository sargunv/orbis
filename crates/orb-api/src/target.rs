//! What a map draws into: a window surface, a texture the runtime owns
//! and the host reads back or composites, or a texture the host owns.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use orb_hal::{Gpu, TextureDesc, TextureFormat, TextureId, TextureUsages};
use orb_hal_wgpu::{Surface, SurfaceError, WgpuGpu};
use orb_runtime::{TargetFormat, Viewport};

use crate::{Error, Runtime, Shared};

/// A render target with its size in physical pixels and its scale
/// (physical pixels per logical pixel).
#[derive(Debug)]
pub struct Target {
    shared: Arc<Shared>,
    kind: Kind,
    width: u32,
    height: u32,
    scale: f64,
    /// The runtime's device generation the target's resources are of.
    generation: u64,
    /// The last frame was read back: the next one copies its pixels out
    /// in its own submission, as the read after it will want them.
    read_back: AtomicBool,
}

/// Frames a texture target lets the GPU queue before the next one waits,
/// as a display's swapchain does: offscreen hosts never queue without
/// bound, and the GPU never idles between frames.
const IN_FLIGHT: usize = 2;

#[derive(Debug)]
enum Kind {
    Surface(Box<Surface>),
    Texture(TextureId),
    /// A texture the host made on the runtime's device, and its format.
    Borrowed(TextureId, TextureFormat),
    /// Its resources are gone ([`Target::release`]).
    Released,
}

/// The format of texture targets.
const TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

impl Target {
    /// Draws into a window surface made with the same runtime.
    pub fn surface(runtime: &Runtime, surface: Surface, scale: f64) -> Self {
        let (width, height) = surface.size();
        Self {
            shared: runtime.shared.clone(),
            kind: Kind::Surface(Box::new(surface)),
            width,
            height,
            scale: sane_scale(scale),
            generation: runtime.shared.generation(),
            read_back: Default::default(),
        }
    }

    /// Draws into an RGBA8 texture the runtime owns.
    pub fn texture(runtime: &Runtime, width: u32, height: u32, scale: f64) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        let id = create_texture(&mut runtime.shared.gpu(), width, height);
        Self {
            shared: runtime.shared.clone(),
            kind: Kind::Texture(id),
            width,
            height,
            scale: sane_scale(scale),
            generation: runtime.shared.generation(),
            read_back: Default::default(),
        }
    }

    /// Draws into a texture the host made on the runtime's device (as
    /// `WgpuGpu::import_metal_texture` makes them), in an 8-bit RGBA or
    /// BGRA format. Each frame is done on the GPU when its render returns,
    /// so the host can read it then.
    pub fn borrowed(
        runtime: &Runtime,
        texture: &orb_hal_wgpu::wgpu::Texture,
        scale: f64,
    ) -> Result<Self, Error> {
        let format = orb_hal_wgpu::from_texture_format(texture.format())
            .filter(|f| matches!(f, TextureFormat::Rgba8Unorm | TextureFormat::Bgra8Unorm))
            .ok_or_else(|| {
                Error::Unsupported(format!("Orbis can't draw into {:?}", texture.format()))
            })?;
        let view = texture.create_view(&Default::default());
        let id = runtime.shared.gpu().register_view(view);
        Ok(Self {
            shared: runtime.shared.clone(),
            kind: Kind::Borrowed(id, format),
            width: texture.width(),
            height: texture.height(),
            scale: sane_scale(scale),
            generation: runtime.shared.generation(),
            read_back: Default::default(),
        })
    }

    /// Width and height in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The size in logical pixels, which the camera and gestures use.
    pub fn viewport(&self) -> Viewport {
        Viewport::of_physical(self.width, self.height, self.scale)
    }

    /// Resizes the target, in physical pixels; a texture target's pixels
    /// are lost.
    pub fn resize(&mut self, width: u32, height: u32, scale: f64) {
        let (width, height) = (width.max(1), height.max(1));
        self.scale = sane_scale(scale);
        if (width, height) == (self.width, self.height) {
            return;
        }
        self.width = width;
        self.height = height;
        let mut device = self.shared.gpu();
        match &mut self.kind {
            Kind::Surface(surface) => device.resize_surface(surface, width, height),
            // The host gives it new textures instead.
            Kind::Released | Kind::Borrowed(..) => {}
            Kind::Texture(id) => {
                // A texture of a replaced device went with it.
                if self.generation == self.shared.generation() {
                    device.destroy_texture(*id);
                }
                *id = create_texture(&mut device, width, height);
                self.generation = self.shared.generation();
            }
        }
    }

    /// A texture target's texture, for a host that composites it with the
    /// runtime's device (`Runtime::with_device`); `None` for a surface
    /// target, or once the GPU was replaced until the next frame.
    pub fn wgpu_texture(&self) -> Option<orb_hal_wgpu::wgpu::Texture> {
        match self.kind {
            Kind::Texture(id) if self.generation == self.shared.generation() => {
                Some(self.shared.gpu().texture(id).clone())
            }
            _ => None,
        }
    }

    /// Copies a texture target's pixels out: RGBA8, rows top to bottom
    /// without padding. Blocks until the GPU is done. The pixels of a
    /// replaced GPU went with it: draw a frame first.
    pub fn read_pixels(&self) -> Result<Vec<u8>, Error> {
        match self.kind {
            Kind::Texture(_) if self.generation != self.shared.generation() => Err(
                Error::InvalidState("the GPU was replaced since the last frame".into()),
            ),
            // wgpu panics when a lost device's wait fails.
            Kind::Texture(_) if self.shared.gpu().lost() => Err(Error::InvalidState(
                "the GPU was lost: draw a frame on its replacement first".into(),
            )),
            Kind::Texture(id) => {
                self.read_back.store(true, Ordering::Relaxed);
                Ok(pollster::block_on(self.shared.gpu().read_texture(id)))
            }
            Kind::Surface(_) | Kind::Borrowed(..) => Err(Error::Unsupported(
                "surface and borrowed targets can't be read back".into(),
            )),
            Kind::Released => Err(Error::TargetLost("the GPU was replaced".into())),
        }
    }

    pub(crate) fn format(&self) -> TargetFormat {
        TargetFormat {
            color: match &self.kind {
                Kind::Surface(surface) => surface.format(),
                Kind::Borrowed(_, format) => *format,
                Kind::Texture(_) | Kind::Released => TEXTURE_FORMAT,
            },
            sample_count: 1,
        }
    }

    pub(crate) fn is_window(&self) -> bool {
        matches!(self.kind, Kind::Surface(_))
    }

    pub(crate) fn same_runtime(&self, shared: &Arc<Shared>) -> bool {
        Arc::ptr_eq(&self.shared, shared)
    }

    /// Starts a frame on `device`, the runtime's, which the frame holds:
    /// the texture to draw it into, or `None` to skip it.
    pub(crate) fn begin(&mut self, device: &mut WgpuGpu) -> Result<Option<TextureId>, Error> {
        // Made before the device was replaced: textures are made again;
        // windows need the host.
        if self.generation != self.shared.generation() {
            match &mut self.kind {
                Kind::Texture(id) => {
                    *id = create_texture(device, self.width, self.height);
                    self.generation = self.shared.generation();
                }
                Kind::Surface(_) | Kind::Borrowed(..) | Kind::Released => {
                    return Err(Error::TargetLost("the GPU was replaced".into()));
                }
            }
        }
        match &mut self.kind {
            Kind::Texture(id) => {
                // A host reading frames back reads the next one too.
                if self.read_back.swap(false, Ordering::Relaxed) {
                    device.read_with_next_submit(*id);
                }
                Ok(Some(*id))
            }
            Kind::Borrowed(id, _) => Ok(Some(*id)),
            Kind::Released => Err(Error::TargetLost("the GPU was replaced".into())),
            Kind::Surface(surface) => match device.acquire(surface) {
                Ok(id) => Ok(Some(id)),
                Err(SurfaceError::Skip) => Ok(None),
                Err(SurfaceError::Lost(message)) => Err(Error::TargetLost(message)),
            },
        }
    }

    /// Shows the frame on a surface, on the device that began it.
    pub(crate) fn end(&mut self, device: &mut WgpuGpu) {
        match &mut self.kind {
            Kind::Surface(surface) => device.present(surface),
            // The host reads it once the render returns.
            Kind::Borrowed(..) => device.end_frame(0),
            _ => device.end_frame(IN_FLIGHT),
        }
    }
}

impl Target {
    /// Frees a window target's surface, so a lost device's go before the
    /// new device's come. The target then reports [`Error::TargetLost`].
    pub(crate) fn release(&mut self) {
        let mut device = self.shared.gpu();
        let kind = std::mem::replace(&mut self.kind, Kind::Released);
        if self.generation != self.shared.generation() {
            // Its resources went with the old device.
            return;
        }
        match kind {
            Kind::Released => {}
            Kind::Texture(id) | Kind::Borrowed(id, _) => device.destroy_texture(id),
            Kind::Surface(mut surface) => device.retire_surface(&mut surface),
        }
    }
}

impl Drop for Target {
    fn drop(&mut self) {
        self.release();
    }
}

fn texture_desc(width: u32, height: u32) -> TextureDesc<'static> {
    TextureDesc {
        label: "target",
        width,
        height,
        array_layers: 1,
        dimension: orb_hal::TextureViewDimension::D2,
        format: TEXTURE_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::COPY_SRC
            | TextureUsages::TEXTURE_BINDING,
        sample_count: 1,
        mip_levels: 1,
        view_formats: &[],
    }
}

fn create_texture(device: &mut WgpuGpu, width: u32, height: u32) -> TextureId {
    device.create_texture(&texture_desc(width, height))
}

fn sane_scale(scale: f64) -> f64 {
    if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    }
}
