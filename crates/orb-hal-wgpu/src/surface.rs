//! Window surfaces: a canvas, a window, a `CAMetalLayer` or an Android
//! `ANativeWindow` as a render target.
//!
//! A frame into a surface is a target registered for it
//! ([`WgpuGpu::register_surface`]), whose texture is acquired when a pass
//! first draws into it, so the CPU's work on the frame holds no image and
//! a frame that draws nothing (one held while compiles run,
//! [`orb_hal::Gpu::presents_wait`]) presents nothing.
//! [`WgpuGpu::take_surface`] takes the texture to present.
//!
//! Natively a [`Surface`] is made from the device's own instance,
//! configured to a non-sRGB 8-bit format (GL JS blends in the style's
//! colour space).

#[cfg(not(target_arch = "wasm32"))]
use std::ffi::c_void;
#[cfg(not(target_arch = "wasm32"))]
use std::ptr::NonNull;

#[cfg(not(target_arch = "wasm32"))]
use orb_hal::TextureFormat;
use orb_hal::TextureId;

#[cfg(not(target_arch = "wasm32"))]
use crate::from_texture_format;
use crate::{TextureEntry, WgpuGpu};

/// A surface, shared between its host and the frame drawing into it.
#[cfg(not(target_arch = "wasm32"))]
pub type SharedSurface = std::sync::Arc<wgpu::Surface<'static>>;
#[cfg(target_arch = "wasm32")]
pub type SharedSurface = std::rc::Rc<wgpu::Surface<'static>>;

/// Why a frame couldn't take a surface's texture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SurfaceError {
    /// Nothing to draw into this time (the window is hidden, or acquiring
    /// timed out); try again on the next frame.
    Skip,
    /// The surface can't be drawn to again; make a new one.
    Lost(String),
}

/// The surface a frame draws into ([`WgpuGpu::register_surface`]).
#[derive(Debug)]
pub(crate) struct Pending {
    id: TextureId,
    surface: SharedSurface,
    config: wgpu::SurfaceConfiguration,
    /// The surface takes `config` before its texture.
    configure: bool,
    /// Once a pass drew into it.
    frame: Option<Result<wgpu::SurfaceTexture, SurfaceError>>,
}

impl WgpuGpu {
    /// A target for a frame into `surface`, configured as `config`, which
    /// it takes first if `configure`: its texture is acquired when a pass
    /// first draws into it, for [`take_surface`](Self::take_surface).
    pub fn register_surface(
        &mut self,
        surface: SharedSurface,
        config: &wgpu::SurfaceConfiguration,
        configure: bool,
    ) -> TextureId {
        let id = TextureId(self.textures.insert(TextureEntry::Surface(None)));
        self.surface = Some(Pending {
            id,
            surface,
            config: config.clone(),
            configure,
            frame: None,
        });
        id
    }

    /// The texture the frame acquired for `id`, to present (`None` if no
    /// pass drew into it), or why acquiring it failed; `id` is released.
    pub fn take_surface(
        &mut self,
        id: TextureId,
    ) -> Option<Result<wgpu::SurfaceTexture, SurfaceError>> {
        let pending = self.surface.take().filter(|s| s.id == id);
        self.textures.remove(id.0);
        pending.and_then(|s| s.frame)
    }

    /// Acquires the pending surface's texture if a pass draws into it
    /// (`id`) and it wasn't yet, configuring the surface again once where
    /// it is outdated.
    pub(crate) fn acquire_surface(&mut self, id: TextureId) {
        let Some(pending) = (self.surface.as_mut()).filter(|s| s.id == id && s.frame.is_none())
        else {
            return;
        };
        if std::mem::take(&mut pending.configure) {
            pending.surface.configure(&self.device, &pending.config);
        }
        let mut configured = false;
        let frame = loop {
            match pending.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(frame)
                | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => break Ok(frame),
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost
                    if !configured =>
                {
                    pending.surface.configure(&self.device, &pending.config);
                    configured = true;
                }
                wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                    break Err(SurfaceError::Skip);
                }
                other => break Err(SurfaceError::Lost(format!("{other:?}"))),
            }
        };
        if let Ok(frame) = &frame {
            let view = (frame.texture).create_view(&wgpu::TextureViewDescriptor::default());
            self.textures[id.0] = TextureEntry::Surface(Some(view));
        }
        pending.frame = Some(frame);
    }
}

/// A platform layer or window to draw into, by raw handle.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug)]
pub enum RawSurface {
    /// A `CAMetalLayer*` (macOS, iOS).
    MetalLayer(NonNull<c_void>),
    /// An `ANativeWindow*` (Android), with a reference the caller holds.
    AndroidWindow(NonNull<c_void>),
}

/// A configured window surface.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct Surface {
    surface: SharedSurface,
    config: wgpu::SurfaceConfiguration,
    format: TextureFormat,
}

#[cfg(not(target_arch = "wasm32"))]
impl Surface {
    /// The color format frames are drawn in.
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// Width and height in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl WgpuGpu {
    /// A surface on a window or view that the safe wgpu API accepts, such
    /// as a winit window, sized in physical pixels.
    pub fn create_surface(
        &self,
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<Surface, String> {
        let surface = self
            .native()?
            .0
            .create_surface(target)
            .map_err(|e| e.to_string())?;
        self.configure_new(surface, width, height)
    }

    /// A surface on a platform layer or window, sized in physical pixels.
    ///
    /// # Safety
    ///
    /// `raw` must point to a live object of its kind, and the object must
    /// outlive the returned surface.
    #[allow(unsafe_code)]
    // Elsewhere (Windows, Linux) no raw handle is a surface here.
    #[cfg_attr(
        not(any(target_os = "macos", target_os = "ios", target_os = "android")),
        allow(unused_variables, unreachable_code)
    )]
    pub unsafe fn create_surface_raw(
        &self,
        raw: RawSurface,
        width: u32,
        height: u32,
    ) -> Result<Surface, String> {
        let target = match raw {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            RawSurface::MetalLayer(layer) => {
                wgpu::SurfaceTargetUnsafe::CoreAnimationLayer(layer.as_ptr())
            }
            #[cfg(target_os = "android")]
            RawSurface::AndroidWindow(window) => wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(wgpu::rwh::RawDisplayHandle::Android(
                    wgpu::rwh::AndroidDisplayHandle::new(),
                )),
                raw_window_handle: wgpu::rwh::RawWindowHandle::AndroidNdk(
                    wgpu::rwh::AndroidNdkWindowHandle::new(window),
                ),
            },
            #[allow(unreachable_patterns)]
            other => return Err(format!("{other:?} isn't a surface on this platform")),
        };
        // SAFETY: the caller guarantees the handle is live and outlives the
        // surface, which is all wgpu requires of it.
        let surface =
            unsafe { self.native()?.0.create_surface_unsafe(target) }.map_err(|e| e.to_string())?;
        self.configure_new(surface, width, height)
    }

    fn configure_new(
        &self,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> Result<Surface, String> {
        let caps = surface.get_capabilities(self.native()?.1);
        let (wgpu_format, format) = caps
            .formats
            .iter()
            .filter(|f| !f.is_srgb())
            .find_map(|&f| Some((f, from_texture_format(f)?)))
            .ok_or_else(|| format!("no non-sRGB 8-bit surface format in {:?}", caps.formats))?;
        let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes[0]
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu_format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&self.device, &config);
        Ok(Surface {
            surface: SharedSurface::new(surface),
            config,
            format,
        })
    }

    /// Resizes a surface, in physical pixels.
    pub fn resize_surface(&mut self, surface: &mut Surface, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == surface.size() {
            return;
        }
        surface.config.width = width;
        surface.config.height = height;
        surface.surface.configure(&self.device, &surface.config);
    }

    /// Starts a frame into `surface` ([`register_surface`](Self::register_surface)).
    pub fn begin_surface(&mut self, surface: &Surface) -> TextureId {
        self.register_surface(surface.surface.clone(), &surface.config, false)
    }

    /// Ends a frame begun with [`begin_surface`](Self::begin_surface):
    /// presents what it drew, if it drew anything.
    pub fn present_surface(&mut self, id: TextureId) -> Result<(), SurfaceError> {
        if let Some(frame) = self.take_surface(id).transpose()? {
            self.queue.present(frame);
        }
        Ok(())
    }

    /// Readies a surface to be dropped: waits until the GPU is done with
    /// submitted work, which holds its images until retired (Vulkan can't
    /// release a swapchain whose images are held).
    pub fn retire_surface(&mut self) {
        // A lost device holds nothing.
        let _ = self.wait(None);
    }

    fn native(&self) -> Result<(&wgpu::Instance, &wgpu::Adapter), String> {
        self.native
            .as_ref()
            .map(|(instance, adapter)| (instance, adapter))
            .ok_or_else(|| "this device was made without an instance for surfaces".into())
    }
}
