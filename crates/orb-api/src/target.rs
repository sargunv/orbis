//! What a map draws into (plan §5.4): a window surface, or a texture the
//! runtime owns and the host reads back or composites.

use std::rc::Rc;

use orb_hal::{Gpu, TextureDesc, TextureFormat, TextureId, TextureUsages};
use orb_hal_wgpu::{Surface, SurfaceError};
use orb_runtime::{TargetFormat, Viewport};

use crate::{Error, Runtime, Shared};

/// A render target with its size in physical pixels and its scale
/// (physical pixels per logical pixel).
#[derive(Debug)]
pub struct Target {
    shared: Rc<Shared>,
    kind: Kind,
    width: u32,
    height: u32,
    scale: f64,
}

#[derive(Debug)]
enum Kind {
    Surface(Box<Surface>),
    Texture(TextureId),
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
        let mut gpu = self.shared.gpu.borrow_mut();
        match &mut self.kind {
            Kind::Surface(surface) => gpu.resize_surface(surface, width, height),
            Kind::Texture(id) => {
                gpu.destroy_texture(*id);
                *id = create_texture(&mut gpu, width, height);
            }
        }
    }

    /// Copies a texture target's pixels out: RGBA8, rows top to bottom
    /// without padding. Blocks until the GPU is done.
    pub fn read_pixels(&self) -> Result<Vec<u8>, Error> {
        match self.kind {
            Kind::Texture(id) => Ok(pollster::block_on(
                self.shared.gpu.borrow().read_texture(id),
            )),
            Kind::Surface(_) => Err(Error::Unsupported(
                "surface targets can't be read back".into(),
            )),
        }
    }

    pub(crate) fn format(&self) -> TargetFormat {
        TargetFormat {
            color: match &self.kind {
                Kind::Surface(surface) => surface.format(),
                Kind::Texture(_) => TEXTURE_FORMAT,
            },
            sample_count: 1,
        }
    }

    pub(crate) fn same_runtime(&self, shared: &Rc<Shared>) -> bool {
        Rc::ptr_eq(&self.shared, shared)
    }

    /// Starts a frame: the texture to draw it into, or `None` to skip it.
    pub(crate) fn begin(&mut self) -> Result<Option<TextureId>, Error> {
        match &mut self.kind {
            Kind::Texture(id) => Ok(Some(*id)),
            Kind::Surface(surface) => match self.shared.gpu.borrow_mut().acquire(surface) {
                Ok(id) => Ok(Some(id)),
                Err(SurfaceError::Skip) => Ok(None),
                Err(SurfaceError::Lost(message)) => Err(Error::TargetLost(message)),
            },
        }
    }

    /// Shows the frame on a surface.
    pub(crate) fn end(&mut self) {
        if let Kind::Surface(surface) = &mut self.kind {
            self.shared.gpu.borrow_mut().present(surface);
        }
    }
}

impl Drop for Target {
    fn drop(&mut self) {
        if let Kind::Texture(id) = self.kind {
            self.shared.gpu.borrow_mut().destroy_texture(id);
        }
    }
}

fn create_texture(gpu: &mut orb_hal_wgpu::WgpuGpu, width: u32, height: u32) -> TextureId {
    gpu.create_texture(&TextureDesc {
        label: "target",
        width,
        height,
        array_layers: 1,
        format: TEXTURE_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::COPY_SRC
            | TextureUsages::TEXTURE_BINDING,
        sample_count: 1,
    })
}

fn sane_scale(scale: f64) -> f64 {
    if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    }
}
