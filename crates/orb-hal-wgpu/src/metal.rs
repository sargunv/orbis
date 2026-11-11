//! Textures a host made with Metal, drawn into in place: a borrowed ring
//! of `MTLTexture`s on the device's own `MTLDevice`.

#![allow(unsafe_code)]

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLPixelFormat, MTLResource, MTLTexture, MTLTextureType, MTLTextureUsage};

use crate::WgpuGpu;

/// What a host's `MTLTexture` is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalTexture {
    pub width: u32,
    pub height: u32,
    /// Its `MTLPixelFormat`.
    pub pixel_format: u64,
    /// Its `MTLDevice`.
    pub device: NonNull<c_void>,
    /// A 2D texture of one sample, layer and level, usable as a render
    /// target.
    pub render_target: bool,
}

/// Reads a host's `MTLTexture`.
///
/// # Safety
/// `texture` is a live `MTLTexture`.
pub unsafe fn metal_texture(texture: NonNull<c_void>) -> MetalTexture {
    // SAFETY: the caller's contract.
    let raw: &ProtocolObject<dyn MTLTexture> = unsafe { texture.cast().as_ref() };
    let device = raw.device();
    MetalTexture {
        width: u32::try_from(raw.width()).unwrap_or(u32::MAX),
        height: u32::try_from(raw.height()).unwrap_or(u32::MAX),
        pixel_format: raw.pixelFormat().0 as u64,
        device: NonNull::from(&*device).cast(),
        render_target: raw.textureType() == MTLTextureType::Type2D
            && raw.sampleCount() == 1
            && raw.mipmapLevelCount() == 1
            && raw.arrayLength() == 1
            && raw.usage().contains(MTLTextureUsage::RenderTarget),
    }
}

/// The format Orbis draws a host's texture in, as its surfaces' (8-bit,
/// not sRGB: GL JS blends in the style's colour space).
fn format(pixel_format: MTLPixelFormat) -> Option<wgpu::TextureFormat> {
    match pixel_format {
        MTLPixelFormat::BGRA8Unorm => Some(wgpu::TextureFormat::Bgra8Unorm),
        MTLPixelFormat::RGBA8Unorm => Some(wgpu::TextureFormat::Rgba8Unorm),
        _ => None,
    }
}

impl WgpuGpu {
    /// The device's `MTLDevice`, when it renders through Metal.
    pub fn metal_device(&self) -> Option<NonNull<c_void>> {
        // SAFETY: only reads the device's handle.
        let device = unsafe { self.device.as_hal::<wgpu::hal::api::Metal>() }?;
        Some(NonNull::from(&**device.raw_device()).cast())
    }

    /// A host's `MTLTexture` as a texture of this device to draw into,
    /// holding a retain of it. Fails when it's of another `MTLDevice`, not
    /// a [render target](MetalTexture::render_target), or in a format
    /// Orbis doesn't draw into.
    ///
    /// # Safety
    /// `texture` is a live `MTLTexture`.
    pub unsafe fn import_metal_texture(
        &self,
        texture: NonNull<c_void>,
    ) -> Result<wgpu::Texture, String> {
        // SAFETY: the caller's contract.
        let info = unsafe { metal_texture(texture) };
        if Some(info.device) != self.metal_device() {
            return Err("the texture is of another MTLDevice than the runtime renders on".into());
        }
        if !info.render_target {
            return Err(
                "the texture isn't a 2D render target of one sample, layer and level".into(),
            );
        }
        let pixel_format = MTLPixelFormat(info.pixel_format as usize);
        let format = format(pixel_format).ok_or_else(|| {
            format!(
                "Orbis draws into BGRA8Unorm and RGBA8Unorm textures, not MTLPixelFormat {}",
                info.pixel_format
            )
        })?;
        // SAFETY: the caller's contract; the retain is wgpu's to release.
        let raw =
            unsafe { Retained::retain(texture.cast::<ProtocolObject<dyn MTLTexture>>().as_ptr()) }
                .ok_or("the texture is null")?;
        let size = wgpu::Extent3d {
            width: info.width,
            height: info.height,
            depth_or_array_layers: 1,
        };
        // SAFETY: a texture of this device (checked above) of the size,
        // format, type and levels it reports.
        let hal = unsafe {
            wgpu::hal::metal::Device::texture_from_raw(
                raw,
                format,
                MTLTextureType::Type2D,
                1,
                1,
                wgpu::hal::CopyExtent {
                    width: info.width,
                    height: info.height,
                    depth: 1,
                },
                None,
            )
        };
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        // SAFETY: as above; the host's frames before are of no interest.
        Ok(unsafe {
            self.device
                .create_texture_from_hal::<wgpu::hal::api::Metal>(
                    hal,
                    &wgpu::TextureDescriptor {
                        label: Some("borrowed target"),
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage,
                        view_formats: &[],
                    },
                    wgpu::TextureUses::UNINITIALIZED,
                )
        })
    }
}
