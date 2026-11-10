//! Native window surfaces: a window, a `CAMetalLayer` or an Android
//! `ANativeWindow` as a render target.
//!
//! A [`Surface`] is made from the device's own instance, configured to a
//! non-sRGB 8-bit format (GL JS blends in the style's colour space) and
//! drawn one frame at a time: [`WgpuGpu::acquire`] registers the frame as a
//! texture to render into, [`WgpuGpu::present`] shows it.

use std::ffi::c_void;
use std::ptr::NonNull;

use orb_hal::{Gpu, TextureFormat, TextureId};

use crate::{WgpuGpu, from_texture_format};

/// A platform layer or window to draw into, by raw handle.
#[derive(Clone, Copy, Debug)]
pub enum RawSurface {
    /// A `CAMetalLayer*` (macOS, iOS).
    MetalLayer(NonNull<c_void>),
    /// An `ANativeWindow*` (Android), with a reference the caller holds.
    AndroidWindow(NonNull<c_void>),
}

/// A configured window surface and the frame being drawn, if any.
#[derive(Debug)]
pub struct Surface {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    format: TextureFormat,
    frame: Option<(wgpu::SurfaceTexture, TextureId)>,
}

/// Why a frame can't be drawn now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SurfaceError {
    /// Nothing to draw into this time (the window is hidden, or acquiring
    /// timed out); try again on the next frame.
    Skip,
    /// The surface can't be drawn to again; make a new one.
    Lost(String),
}

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
            surface,
            config,
            format,
            frame: None,
        })
    }

    /// Resizes a surface, in physical pixels. A frame being drawn is
    /// dropped unshown.
    pub fn resize_surface(&mut self, surface: &mut Surface, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if (width, height) == surface.size() {
            return;
        }
        if let Some((_, id)) = surface.frame.take() {
            self.destroy_texture(id);
        }
        surface.config.width = width;
        surface.config.height = height;
        surface.surface.configure(&self.device, &surface.config);
    }

    /// Starts a frame: the texture to render it into.
    pub fn acquire(&mut self, surface: &mut Surface) -> Result<TextureId, SurfaceError> {
        if let Some((_, id)) = &surface.frame {
            return Ok(*id);
        }
        let mut reconfigured = false;
        let frame = loop {
            match surface.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(frame)
                | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => break frame,
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost
                    if !reconfigured =>
                {
                    surface.surface.configure(&self.device, &surface.config);
                    reconfigured = true;
                }
                wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                    return Err(SurfaceError::Skip);
                }
                other => return Err(SurfaceError::Lost(format!("{other:?}"))),
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let id = self.register_view(view);
        surface.frame = Some((frame, id));
        Ok(id)
    }

    /// Readies a surface to be dropped: drops its frame and waits until
    /// the GPU is done with submitted work, which holds its images until
    /// retired (Vulkan can't release a swapchain whose images are held).
    pub fn retire_surface(&mut self, surface: &mut Surface) {
        if let Some((_, id)) = surface.frame.take() {
            self.destroy_texture(id);
        }
        // A lost device holds nothing, and wgpu panics when its wait fails.
        if !self.lost() {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        }
    }

    /// Shows the frame started with [`acquire`](Self::acquire).
    pub fn present(&mut self, surface: &mut Surface) {
        if let Some((frame, id)) = surface.frame.take() {
            self.destroy_texture(id);
            self.queue.present(frame);
        }
    }

    fn native(&self) -> Result<(&wgpu::Instance, &wgpu::Adapter), String> {
        self.native
            .as_ref()
            .map(|(instance, adapter)| (instance, adapter))
            .ok_or_else(|| "this device was made without an instance for surfaces".into())
    }
}
