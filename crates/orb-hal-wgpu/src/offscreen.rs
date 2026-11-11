//! Devices without a surface, readback and waiting for the GPU: for tests,
//! the fixture runner, snapshots and the bench.
//!
//! Readback is async on both targets. Natively the futures block on
//! `Device::poll` and are ready when first polled (drive them with any
//! executor, such as `pollster`); in the browser they resolve from the
//! event loop.

use std::future::Future;

use orb_hal::{BufferId, TextureId};

use crate::WgpuGpu;

/// Readback buffers not in use, shared with the readbacks' futures.
pub(crate) type Readbacks = std::sync::Arc<std::sync::Mutex<Vec<wgpu::Buffer>>>;

/// The most readback buffers kept.
const READBACKS: usize = 2;

/// A texture's pixels copied into a readback buffer, rows padded to
/// wgpu's copy alignment.
#[derive(Debug)]
pub(crate) struct Readback {
    buffer: wgpu::Buffer,
    row: u32,
    padded_row: u32,
    height: u32,
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
    /// A device on the default adapter with WebGPU's default limits, without
    /// a surface. `ORB_BACKEND`, `WGPU_BACKEND` and wgpu's other variables
    /// choose the adapter. Natively the device keeps its instance, so it can
    /// also draw to window surfaces.
    pub async fn offscreen() -> Result<Self, String> {
        let (instance, adapter, compiler) = adapter().await?;
        // Orbis draws indirectly from compute; devices without that (the
        // iOS simulator's Metal) can't draw it; GL boots to flat frames (#140).
        let downlevel = adapter.get_downlevel_capabilities().flags;
        let needed = wgpu::DownlevelFlags::COMPUTE_SHADERS
            | wgpu::DownlevelFlags::INDIRECT_EXECUTION
            | wgpu::DownlevelFlags::VERTEX_STORAGE;
        if !downlevel.contains(needed) && adapter.get_info().backend != wgpu::Backend::Gl {
            return Err(format!(
                "{} lacks {:?}, which Orbis needs",
                adapter.get_info().name,
                needed - downlevel
            ));
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("orbis"),
                // Natively, warm starts reuse pipelines through a cache;
                // timestamps time passes for measurements (`set_timing`);
                // clip distances clip cut buildings without discarding.
                required_features: adapter.features()
                    & (wgpu::Features::PIPELINE_CACHE
                        | wgpu::Features::TIMESTAMP_QUERY
                        | wgpu::Features::CLIP_DISTANCES),
                required_limits: crate::device_limits(&adapter),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no device: {e}"))?;
        #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
        let mut gpu = Self::new(device, queue);
        #[cfg(not(webgpu))]
        gpu.read_formats(&adapter);
        #[cfg(not(target_arch = "wasm32"))]
        {
            gpu.watch_loss();
            gpu.native = Some((instance, adapter));
            gpu.shader_compiler = compiler;
        }
        #[cfg(target_arch = "wasm32")]
        let _ = (instance, compiler);
        Ok(gpu)
    }

    /// Resolves when the GPU has finished all work submitted so far.
    pub(crate) fn work_done(&self) -> impl Future<Output = ()> + use<> {
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
        let device = self.device.clone();
        async move {
            #[cfg(webgl)]
            {
                let _ = index;
                let mut receiver = receiver;
                polled(&device, || {
                    receiver.try_recv().expect("work-done callback dropped")
                })
                .await;
            }
            #[cfg(not(webgl))]
            {
                device
                    .poll(wgpu::PollType::Wait {
                        submission_index: Some(index),
                        timeout: None,
                    })
                    .expect("device lost");
                receiver.await.expect("work-done callback dropped");
            }
        }
    }

    /// Copies the pixels of a 4-byte-per-texel texture out, row by row
    /// without padding. The copy is submitted now, unless the last
    /// submission made it ([`read_with_next_submit`](Self::read_with_next_submit));
    /// the future holds no borrow of the device, so the texture may go
    /// before it resolves.
    pub fn read_texture(&self, id: TextureId) -> impl Future<Output = Vec<u8>> + use<> {
        let copied = self.copied.lock().expect("copied poisoned").take();
        let copy = match copied {
            Some((texture, copy)) if texture == id => copy,
            _ => self.copy_now(self.texture(id)),
        };
        self.finish_read(copy)
    }

    /// [`read_texture`](Self::read_texture) for a wgpu texture the host
    /// owns, such as a canvas's current texture (made with `COPY_SRC`).
    pub fn read_wgpu_texture(
        &self,
        texture: &wgpu::Texture,
    ) -> impl Future<Output = Vec<u8>> + use<> {
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
            .filter(|id| self.textures.contains(id.0) && self.textures[id.0].texture.is_some())
            .map(|id| (id, self.encode_copy(encoder, self.texture(id))));
        let unread = std::mem::replace(&mut *self.copied.lock().expect("copied poisoned"), copy);
        if let Some((_, unread)) = unread {
            keep(&self.readbacks, unread.buffer);
        }
    }

    /// Submits a copy of `texture` into a readback buffer.
    fn copy_now(&self, texture: &wgpu::Texture) -> Readback {
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

    /// Encodes a copy of `texture` into a readback buffer, a kept one of
    /// its size if there is one.
    fn encode_copy(&self, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture) -> Readback {
        let (width, height) = (texture.width(), texture.height());
        let row = width * 4;
        let padded_row = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let size = u64::from(padded_row * height);
        let kept = {
            let mut kept = self.readbacks.lock().expect("readbacks poisoned");
            let at = kept.iter().position(|b| b.size() == size);
            at.map(|at| kept.swap_remove(at))
        };
        let buffer = kept.unwrap_or_else(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: None,
                },
            },
            texture.size(),
        );
        Readback {
            buffer,
            row,
            padded_row,
            height,
        }
    }

    /// Maps a copy's buffer once the GPU has made it, and unpads its rows.
    fn finish_read(&self, copy: Readback) -> impl Future<Output = Vec<u8>> + use<> {
        let (device, readbacks) = (self.device.clone(), self.readbacks.clone());
        async move {
            let Readback {
                buffer,
                row,
                padded_row,
                height,
            } = copy;
            let mapped = map_read(&device, &buffer).await;
            let mut pixels = Vec::with_capacity((row * height) as usize);
            for y in 0..height {
                let start = (y * padded_row) as usize;
                pixels.extend_from_slice(&mapped[start..start + row as usize]);
            }
            drop(mapped);
            buffer.unmap();
            keep(&readbacks, buffer);
            pixels
        }
    }

    /// Copies a buffer's bytes out. The buffer needs `COPY_SRC`.
    pub async fn read_buffer(&self, id: BufferId) -> Vec<u8> {
        let source = self.buffer(id);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: source.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("buffer readback"),
            });
        self.encode_writes(&mut encoder);
        encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, source.size());
        self.queue.submit([encoder.finish()]);
        map_read(&self.device, &buffer).await.to_vec()
    }
}

/// Keeps an unmapped readback buffer for the next readback of its size.
fn keep(readbacks: &Readbacks, buffer: wgpu::Buffer) {
    let mut kept = readbacks.lock().expect("readbacks poisoned");
    if kept.len() == READBACKS {
        kept.remove(0);
    }
    kept.push(buffer);
}

/// What `ready` gives once it gives something, the device polled before
/// each try: WebGL 2 can't wait for its fence, so each task polls it once
/// until the callback that `ready` watches has run.
#[cfg(webgl)]
async fn polled<T>(device: &wgpu::Device, mut ready: impl FnMut() -> Option<T>) -> T {
    loop {
        let _ = device.poll(wgpu::PollType::Poll);
        if let Some(value) = ready() {
            return value;
        }
        let tick = js_sys::Promise::new(&mut |resolve, _| {
            let global = js_sys::global();
            let set_timeout = js_sys::Reflect::get(&global, &"setTimeout".into())
                .ok()
                .and_then(|f| wasm_bindgen::JsCast::dyn_into::<js_sys::Function>(f).ok())
                .expect("setTimeout");
            let _ = set_timeout.call2(&global, &resolve, &1.into());
        });
        let _ = wasm_bindgen_futures::JsFuture::from(tick).await;
    }
}

pub(crate) async fn map_read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> wgpu::BufferView {
    #[cfg_attr(not(webgl), allow(unused_mut))]
    let (sender, mut receiver) = futures_channel::oneshot::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    #[cfg(webgl)]
    let mapped = polled(device, || {
        receiver.try_recv().expect("mapping callback dropped")
    })
    .await;
    #[cfg(not(webgl))]
    let mapped = {
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device lost during readback");
        receiver.await.expect("mapping callback dropped")
    };
    mapped.expect("readback mapping failed");
    buffer
        .slice(..)
        .get_mapped_range()
        .expect("readback buffer is mapped")
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
fn with_env(descriptor: wgpu::InstanceDescriptor) -> wgpu::InstanceDescriptor {
    #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
    let mut descriptor = descriptor.with_env();
    #[cfg(not(target_arch = "wasm32"))]
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
