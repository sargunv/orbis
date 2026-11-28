//! Devices without a surface (native only), and readback and waiting for
//! the GPU on both targets: for tests, the fixture runner, snapshots and
//! the bench.
//!
//! Readback is async on both targets. Natively the futures block on
//! `Device::poll` and are ready when first polled (drive them with any
//! executor, such as `pollster`); in the browser they resolve from the
//! event loop. Each resolves to [`Lost`] if the device goes first.

use std::future::Future;

use orb_hal::{BufferId, TextureId};

use crate::{Lost, WgpuGpu};

/// Readbacks' buffers not in use, each readback's bands together, shared
/// with the readbacks' futures.
pub(crate) type Readbacks = std::sync::Arc<std::sync::Mutex<Vec<Vec<wgpu::Buffer>>>>;

/// The most readbacks' buffers kept.
const READBACKS: usize = 2;

/// A texture's pixels copied into readback buffers, rows padded to wgpu's
/// copy alignment: in bands of rows, each buffer within the device's
/// largest, since a frame as wide as the device's textures can
/// take more bytes than one buffer holds (16384² takes 1 GiB).
#[derive(Debug)]
pub(crate) struct Readback {
    bands: Vec<wgpu::Buffer>,
    row: u32,
    padded_row: u32,
}

/// An instance, configured from the environment ([`with_env`]), and its default adapter.
///
/// Without wgpu's indirect-call validation unless the environment asks
/// for it (`WGPU_VALIDATION_INDIRECT_CALL=1`): Orbis's indirect arguments
/// are its cull shader's, bounded by the scene's own buffers, and the
/// validation costs every frame a compute pass, a buffer and a copy (on
/// sparse 512 px views a sixth of the frame: 0.06 ms of CPU, 0.15 ms of
/// wall). After it, Qualcomm's Vulkan driver also draws nothing from
/// drawIndexedIndirect (Adreno 619 measured; gfx-rs/wgpu#8801 on 650/740).
///
/// On Windows, D3D12's shader compiler too ([`crate::dx12`]): what it is,
/// when the adapter is D3D12's.
#[cfg(not(target_arch = "wasm32"))]
async fn adapter() -> Result<(wgpu::Instance, wgpu::Adapter, Option<String>), String> {
    let described = || {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor
            .flags
            .remove(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL);
        with_env(descriptor)
    };
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut descriptor = described();
    #[cfg(windows)]
    let compiler = Some(crate::dx12::choose(&mut descriptor));
    #[cfg(not(windows))]
    let compiler: Option<String> = None;
    let backends = descriptor.backends;
    let mut instance = wgpu::Instance::new(descriptor);
    let mut adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .map_err(|e| format!("no adapter: {e}"))?;
    // GL taken where another backend was asked for too (Android 8's
    // emulator, which has a Vulkan loader but no Vulkan device): an
    // instance of GL alone, whose window surfaces are GL's. Beside the
    // others, its surface on an Android window failed to configure
    // ("Invalid surface").
    if adapter.get_info().backend == wgpu::Backend::Gl && backends != wgpu::Backends::GL {
        let mut descriptor = described();
        descriptor.backends = wgpu::Backends::GL;
        instance = wgpu::Instance::new(descriptor);
        adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|e| format!("no adapter: {e}"))?;
    }
    #[cfg(windows)]
    let compiler = compiler
        .filter(|_| adapter.get_info().backend == wgpu::Backend::Dx12)
        .map(|choice| {
            if let Some(warning) = &choice.warning {
                eprintln!("{warning}");
            }
            choice.described
        });
    Ok((instance, adapter, compiler))
}

impl WgpuGpu {
    /// A native device on the default adapter ([`open`](Self::open)),
    /// without a surface, taking at most the frame path `ORB_DRIVER` asks
    /// for (`gpu`, `cpu`, `cpu-tex` or `webgl2`). `ORB_BACKEND`,
    /// `WGPU_BACKEND` and wgpu's other variables choose the adapter. The
    /// device keeps its instance, so it can also draw to window surfaces.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn offscreen() -> Result<Self, String> {
        Self::offscreen_on(crate::driver_from_env()).await
    }

    /// [`offscreen`](Self::offscreen), taking at most frame path `most`.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn offscreen_on(most: orb_hal::FramePath) -> Result<Self, String> {
        let (instance, adapter, compiler) = adapter().await?;
        let mut gpu = Self::open(&adapter, most, |_| {})
            .await
            .map_err(|e| format!("no device: {e}"))?;
        gpu.native = Some((instance, adapter));
        gpu.shader_compiler = compiler;
        Ok(gpu)
    }

    /// Resolves when the GPU has finished all work submitted so far.
    pub(crate) fn work_done(&mut self) -> impl Future<Output = Result<(), Lost>> + use<> {
        // Gathered writes are work submitted too.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("work done"),
            });
        self.encode_writes(&mut encoder);
        let index = self.queue.submit([encoder.finish()]);
        let (sender, receiver) = futures_channel::oneshot::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = sender.send(());
        });
        let waiter = self.waiter();
        async move { waiter.done(index, receiver).await }
    }

    /// Copies the pixels of a 4-byte-per-texel texture out, row by row
    /// without padding. The copy is submitted now, unless the last
    /// submission made it ([`read_with_next_submit`](Self::read_with_next_submit));
    /// the future holds no borrow of the device, so the texture may go
    /// before it resolves.
    pub fn read_texture(
        &mut self,
        id: TextureId,
    ) -> impl Future<Output = Result<Vec<u8>, Lost>> + use<> {
        let copy = match self.copied.take() {
            Some((texture, copy)) if texture == id => copy,
            _ => {
                let texture = self.texture(id).clone();
                self.copy_now(&texture)
            }
        };
        self.finish_read(copy)
    }

    /// [`read_texture`](Self::read_texture) for a wgpu texture the host
    /// owns, such as a canvas's current texture (made with `COPY_SRC`).
    pub fn read_wgpu_texture(
        &mut self,
        texture: &wgpu::Texture,
    ) -> impl Future<Output = Result<Vec<u8>, Lost>> + use<> {
        let copy = self.copy_now(texture);
        self.finish_read(copy)
    }

    /// Has the next [`submit`](orb_hal::Gpu::submit) copy texture `id` out
    /// after its passes, for a [`read_texture`](Self::read_texture) to
    /// map: a frame read back costs one submission rather than two, and
    /// waits once.
    pub fn read_with_next_submit(&mut self, id: TextureId) {
        self.copy_next = Some(id);
    }

    /// Encodes the copy armed by [`read_with_next_submit`](Self::read_with_next_submit)
    /// at the end of a submission, or forgets an earlier one's: the
    /// texture may be drawn again.
    pub(crate) fn encode_armed_copy(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let copy = (self.copy_next.take())
            .filter(|id| (self.textures.get(id.0)).is_some_and(|t| t.texture().is_some()))
            .map(|id| (id, self.encode_copy(encoder, self.texture(id))));
        let unread = std::mem::replace(&mut self.copied, copy);
        if let Some((_, unread)) = unread {
            keep(&self.readbacks, unread.bands);
        }
    }

    /// Submits a copy of `texture` into a readback buffer.
    fn copy_now(&mut self, texture: &wgpu::Texture) -> Readback {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        self.encode_writes(&mut encoder);
        let copy = self.encode_copy(&mut encoder, texture);
        self.queue.submit([encoder.finish()]);
        copy
    }

    /// Encodes a copy of `texture` into readback buffers, kept ones of
    /// their sizes if there are some.
    fn encode_copy(&self, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture) -> Readback {
        let (width, height) = (texture.width(), texture.height());
        let row = width * 4;
        let padded_row = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let largest = self.device.limits().max_buffer_size;
        let band_rows = (largest / u64::from(padded_row)).clamp(1, u64::from(height)) as u32;
        let sizes: Vec<u64> = (0..height)
            .step_by(band_rows as usize)
            .map(|y| u64::from(padded_row) * u64::from(band_rows.min(height - y)))
            .collect();
        let kept = {
            let mut kept = self.readbacks.lock().expect("readbacks poisoned");
            let at = (kept.iter()).position(|bands| {
                bands
                    .iter()
                    .map(wgpu::Buffer::size)
                    .eq(sizes.iter().copied())
            });
            at.map(|at| kept.swap_remove(at))
        };
        let bands = kept.unwrap_or_else(|| {
            (sizes.iter())
                .map(|&size| {
                    self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("readback"),
                        size,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    })
                })
                .collect()
        });
        for (band, y) in bands.iter().zip((0..height).step_by(band_rows as usize)) {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: band,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width,
                    height: band_rows.min(height - y),
                    depth_or_array_layers: 1,
                },
            );
        }
        Readback {
            bands,
            row,
            padded_row,
        }
    }

    /// Maps a copy's buffer once the GPU has made it, and unpads its rows.
    fn finish_read(&self, copy: Readback) -> impl Future<Output = Result<Vec<u8>, Lost>> + use<> {
        let (waiter, readbacks) = (self.waiter(), self.readbacks.clone());
        async move {
            let Readback {
                bands,
                row,
                padded_row,
            } = copy;
            let (row, padded_row) = (row as usize, padded_row as usize);
            let bytes: u64 = bands.iter().map(wgpu::Buffer::size).sum();
            let mut pixels = Vec::with_capacity(bytes as usize / padded_row * row);
            for band in &bands {
                let mapped = waiter.map_read(band).await?;
                for padded in mapped.chunks_exact(padded_row) {
                    pixels.extend_from_slice(&padded[..row]);
                }
                drop(mapped);
                band.unmap();
            }
            keep(&readbacks, bands);
            Ok(pixels)
        }
    }

    /// Copies a buffer's bytes out. The buffer needs `COPY_SRC`. The
    /// future holds no borrow of the device.
    pub fn read_buffer(
        &mut self,
        id: BufferId,
    ) -> impl Future<Output = Result<Vec<u8>, Lost>> + use<> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("buffer readback"),
            });
        self.encode_writes(&mut encoder);
        let source = self.buffer(id);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: source.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, source.size());
        self.queue.submit([encoder.finish()]);
        let waiter = self.waiter();
        async move { Ok(waiter.map_read(&buffer).await?.to_vec()) }
    }
}

/// Keeps a readback's unmapped buffers for the next readback of their sizes.
fn keep(readbacks: &Readbacks, bands: Vec<wgpu::Buffer>) {
    let mut kept = readbacks.lock().expect("readbacks poisoned");
    if kept.len() == READBACKS {
        kept.remove(0);
    }
    kept.push(bands);
}

/// Picks the backends a native device may run on, over wgpu's own
/// `WGPU_BACKEND`: a comma list of `vulkan`, `metal`, `dx12` and `gl`, for
/// testing one a platform has beside its default (GLES on Android).
#[cfg(not(target_arch = "wasm32"))]
const BACKEND: &str = "ORB_BACKEND";

/// `descriptor` configured from wgpu's environment variables and
/// [`BACKEND`].
///
/// On Linux, without wgpu's debug object names (`InstanceFlags::DEBUG`, on
/// in debug builds) where the Vulkan loader is older than 1.4.345, unless
/// `WGPU_DEBUG=1` asks for them. Those loaders look up a device's names
/// under a different lock than the one creating and destroying devices
/// holds, so naming an object while another thread makes or drops a device
/// can crash in `loader_get_icd_and_device` (#169; fixed by
/// KhronosGroup/Vulkan-Loader@5ee27b30c). Fedora 44 ships 1.4.341, where
/// parallel test binaries crashed four runs in five.
#[cfg(not(target_arch = "wasm32"))]
fn with_env(descriptor: wgpu::InstanceDescriptor) -> wgpu::InstanceDescriptor {
    let mut descriptor = descriptor.with_env();
    if let Ok(names) = std::env::var(BACKEND) {
        descriptor.backends = wgpu::Backends::from_comma_list(&names);
    }
    #[cfg(target_os = "linux")]
    if descriptor.flags.contains(wgpu::InstanceFlags::DEBUG)
        && descriptor.backends.contains(wgpu::Backends::VULKAN)
        && std::env::var_os("WGPU_DEBUG").is_none()
        && racy_vulkan_loader()
    {
        descriptor.flags.remove(wgpu::InstanceFlags::DEBUG);
    }
    descriptor
}

/// Whether the system's Vulkan loader predates 1.4.345, whose debug object
/// names race with device creation on other threads ([`with_env`]).
#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
fn racy_vulkan_loader() -> bool {
    static RACY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *RACY.get_or_init(|| {
        // SAFETY: loading the system's loader runs no code of ours; wgpu
        // loads the same library.
        let Ok(entry) = (unsafe { ash::Entry::load() }) else {
            return false;
        };
        // SAFETY: a global command, valid with no instance.
        let version = unsafe { entry.try_enumerate_instance_version() }
            .ok()
            .flatten()
            .unwrap_or(ash::vk::API_VERSION_1_0);
        version < ash::vk::make_api_version(0, 1, 4, 345)
    })
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use orb_hal::{Gpu, TextureDesc, TextureFormat, TextureRegion, TextureUsages};

    use super::*;

    #[test]
    fn a_texture_past_the_largest_buffer_reads_back_in_bands() {
        let (_instance, adapter, _) = pollster::block_on(adapter()).expect("a GPU adapter");
        // Rows of 16 KiB, 64 to a buffer: 150 rows read as 64, 64 and 22.
        const LARGEST: u64 = 1 << 20;
        let required_limits = wgpu::Limits {
            max_buffer_size: LARGEST,
            ..crate::device_limits(&adapter)
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits,
            ..Default::default()
        }))
        .expect("a device");
        assert_eq!(device.limits().max_buffer_size, LARGEST);
        let flags = adapter.get_downlevel_capabilities().flags;
        let mut gpu = WgpuGpu::new(device, queue, flags, orb_hal::FramePath::Gpu);
        let (width, height) = (4096, 150);
        let texture = gpu.create_texture(&TextureDesc {
            label: "banded",
            width,
            height,
            array_layers: 1,
            dimension: orb_hal::TextureViewDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
            sample_count: 1,
            mip_levels: 1,
        });
        let texels: Vec<u8> = (0..width * height * 4).map(|i| (i % 251) as u8).collect();
        // In strips, each within the largest staging buffer.
        let strip = 30;
        for (y, texels) in (0..height)
            .step_by(strip)
            .zip(texels.chunks(strip * width as usize * 4))
        {
            gpu.write_texture(
                &TextureRegion {
                    texture,
                    x: 0,
                    y,
                    layer: 0,
                    width,
                    height: strip as u32,
                    level: 0,
                },
                texels,
            );
        }
        assert_eq!(
            pollster::block_on(gpu.read_texture(texture)),
            Ok(texels.clone())
        );
        // Again into the kept buffers.
        assert_eq!(pollster::block_on(gpu.read_texture(texture)), Ok(texels));
    }
}
