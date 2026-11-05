//! What a map draws into (plan §5.4): a window surface, or a texture the
//! runtime owns and the host reads back or composites.

use std::rc::Rc;

use orb_hal::{TextureDesc, TextureFormat, TextureId, TextureUsages};
use orb_hal_gl::egl::EglWindow;
use orb_hal_wgpu::SurfaceError;
use orb_runtime::{TargetFormat, Viewport};

use crate::device::Device;
use crate::{Error, Runtime, Shared};

/// A window surface: wgpu's on Modern, an EGL window on Compat.
// One per window target: the variants' sizes don't matter.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum Surface {
    Modern(orb_hal_wgpu::Surface),
    /// A native window's EGL surface, and the frame drawn for it.
    Compat {
        window: EglWindow,
        frame: Option<(TextureId, (u32, u32))>,
    },
}

impl Surface {
    /// An EGL window made with [`Runtime::with_device`]'s Compat context.
    pub fn compat(window: EglWindow) -> Self {
        Self::Compat {
            window,
            frame: None,
        }
    }
}

/// A render target with its size in physical pixels and its scale
/// (physical pixels per logical pixel).
#[derive(Debug)]
pub struct Target {
    shared: Rc<Shared>,
    kind: Kind,
    width: u32,
    height: u32,
    scale: f64,
    /// The runtime's device generation the target's resources are of.
    generation: u64,
}

#[derive(Debug)]
enum Kind {
    Surface(Box<Surface>),
    Texture(TextureId),
}

/// The format of texture targets and Compat frames.
const TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

impl Target {
    /// Draws into a window surface made with the same runtime.
    pub fn surface(runtime: &Runtime, surface: Surface, scale: f64) -> Self {
        let (width, height) = match &surface {
            Surface::Modern(s) => s.size(),
            Surface::Compat { window, .. } => match &*runtime.shared.gpu.borrow() {
                Device::Compat(c) => c.egl.window_size(window),
                Device::Modern(_) => (1, 1),
            },
        };
        Self {
            shared: runtime.shared.clone(),
            kind: Kind::Surface(Box::new(surface)),
            width,
            height,
            scale: sane_scale(scale),
            generation: runtime.shared.generation.get(),
        }
    }

    /// Draws into an RGBA8 texture the runtime owns.
    pub fn texture(runtime: &Runtime, width: u32, height: u32, scale: f64) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        let id = create_texture(&mut runtime.shared.gpu.borrow_mut(), width, height);
        Self {
            shared: runtime.shared.clone(),
            kind: Kind::Texture(id),
            width,
            height,
            scale: sane_scale(scale),
            generation: runtime.shared.generation.get(),
        }
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
        Viewport {
            width: f64::from(self.width) / self.scale,
            height: f64::from(self.height) / self.scale,
        }
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
        let mut device = self.shared.gpu.borrow_mut();
        match (&mut self.kind, &mut *device) {
            (Kind::Surface(surface), Device::Modern(gpu)) => {
                if let Surface::Modern(surface) = &mut **surface {
                    gpu.resize_surface(surface, width, height);
                }
            }
            // An EGL window follows its native window; its frame follows
            // the size on the next frame.
            (Kind::Surface(_), Device::Compat(_)) => {}
            (Kind::Texture(id), device) => {
                // A texture of a replaced device went with it.
                if self.generation == self.shared.generation.get() {
                    device.gpu().destroy_texture(*id);
                }
                *id = create_texture(device, width, height);
                self.generation = self.shared.generation.get();
            }
        }
    }

    /// Copies a texture target's pixels out: RGBA8, rows top to bottom
    /// without padding. Blocks until the GPU is done. The pixels of a
    /// replaced GPU went with it: draw a frame first.
    pub fn read_pixels(&self) -> Result<Vec<u8>, Error> {
        match self.kind {
            Kind::Texture(_) if self.generation != self.shared.generation.get() => Err(
                Error::InvalidState("the GPU was replaced since the last frame".into()),
            ),
            Kind::Texture(id) => Ok(self.shared.gpu.borrow_mut().read_texture(id)),
            Kind::Surface(_) => Err(Error::Unsupported(
                "surface targets can't be read back".into(),
            )),
        }
    }

    pub(crate) fn format(&self) -> TargetFormat {
        TargetFormat {
            color: match &self.kind {
                Kind::Surface(surface) => match &**surface {
                    Surface::Modern(s) => s.format(),
                    Surface::Compat { .. } => TEXTURE_FORMAT,
                },
                Kind::Texture(_) => TEXTURE_FORMAT,
            },
            sample_count: 1,
        }
    }

    pub(crate) fn same_runtime(&self, shared: &Rc<Shared>) -> bool {
        Rc::ptr_eq(&self.shared, shared)
    }

    /// Starts a frame on `device`, the runtime's, which the frame holds:
    /// the texture to draw it into, or `None` to skip it.
    pub(crate) fn begin(&mut self, device: &mut Device) -> Result<Option<TextureId>, Error> {
        if device.lost() {
            return Err(Error::DeviceLost);
        }
        // Made before the device was replaced: textures are made again;
        // windows need the host.
        if self.generation != self.shared.generation.get() {
            match &mut self.kind {
                Kind::Texture(id) => {
                    *id = create_texture(device, self.width, self.height);
                    self.generation = self.shared.generation.get();
                }
                Kind::Surface(_) => {
                    return Err(Error::TargetLost("the GPU was replaced".into()));
                }
            }
        }
        match (&mut self.kind, device) {
            (Kind::Texture(id), _) => Ok(Some(*id)),
            (Kind::Surface(surface), Device::Modern(gpu)) => match &mut **surface {
                Surface::Modern(surface) => match gpu.acquire(surface) {
                    Ok(id) => Ok(Some(id)),
                    Err(SurfaceError::Skip) => Ok(None),
                    Err(SurfaceError::Lost(message)) => Err(Error::TargetLost(message)),
                },
                Surface::Compat { .. } => Err(Error::TargetLost("an EGL window on Modern".into())),
            },
            (Kind::Surface(surface), Device::Compat(c)) => {
                let Surface::Compat { window, frame } = &mut **surface else {
                    return Err(Error::TargetLost("a wgpu surface on Compat".into()));
                };
                c.egl
                    .make_current(Some(window))
                    .map_err(Error::TargetLost)?;
                let size = c.egl.window_size(window);
                (self.width, self.height) = size;
                let id = match *frame {
                    Some((id, made)) if made == size => id,
                    stale => {
                        if let Some((id, _)) = stale {
                            orb_hal::Gpu::destroy_texture(&mut c.gpu, id);
                        }
                        let id = frame_texture(&mut c.gpu, size);
                        *frame = Some((id, size));
                        id
                    }
                };
                Ok(Some(id))
            }
        }
    }

    /// Shows the frame on a surface, on the device that began it.
    pub(crate) fn end(&mut self, device: &mut Device) {
        let Kind::Surface(surface) = &mut self.kind else {
            return;
        };
        match (&mut **surface, device) {
            (Surface::Modern(surface), Device::Modern(gpu)) => gpu.present(surface),
            (Surface::Compat { window, frame }, Device::Compat(c)) => {
                if let Some((id, (w, h))) = *frame {
                    c.gpu.present(id, w, h);
                }
                // A failed swap shows as a lost window next frame.
                let _ = c.egl.swap(window);
            }
            _ => {}
        }
    }
}

impl Drop for Target {
    fn drop(&mut self) {
        let mut device = self.shared.gpu.borrow_mut();
        if self.generation != self.shared.generation.get() {
            // Its resources went with the old device.
            return;
        }
        let kind = std::mem::replace(&mut self.kind, Kind::Texture(TextureId(u32::MAX)));
        match (kind, &mut *device) {
            (Kind::Texture(id), device) => device.gpu().destroy_texture(id),
            (Kind::Surface(surface), Device::Modern(gpu)) => {
                if let Surface::Modern(mut surface) = *surface {
                    gpu.retire_surface(&mut surface);
                }
            }
            (Kind::Surface(surface), Device::Compat(c)) => {
                if let Surface::Compat { window, frame } = *surface {
                    if let Some((id, _)) = frame {
                        orb_hal::Gpu::destroy_texture(&mut c.gpu, id);
                    }
                    c.egl.destroy_window(window);
                }
            }
        }
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
    }
}

fn create_texture(device: &mut Device, width: u32, height: u32) -> TextureId {
    device.gpu().create_texture(&texture_desc(width, height))
}

fn frame_texture(gpu: &mut orb_hal_gl::GlGpu, (width, height): (u32, u32)) -> TextureId {
    orb_hal::Gpu::create_texture(gpu, &texture_desc(width, height))
}

fn sane_scale(scale: f64) -> f64 {
    if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    }
}
