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

/// An instance, configured from the environment, and its default adapter.
///
/// Without wgpu's indirect-call validation unless the environment asks
/// for it (`WGPU_VALIDATION_INDIRECT_CALL=1`): Orbis's indirect arguments
/// are its cull shader's, bounded by the scene's own buffers, and the
/// validation costs every frame a compute pass, a buffer and a copy (on
/// sparse 512 px views a sixth of the frame: 0.06 ms of CPU, 0.15 ms of
/// wall). After it, Qualcomm's Vulkan driver also draws nothing from
/// drawIndexedIndirect (Adreno 619 measured; gfx-rs/wgpu#8801 on 650/740).
async fn adapter() -> Result<(wgpu::Instance, wgpu::Adapter), String> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor
        .flags
        .remove(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL);
    let instance = wgpu::Instance::new(descriptor.with_env());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .map_err(|e| format!("no adapter: {e}"))?;
    Ok((instance, adapter))
}

impl WgpuGpu {
    /// A device on the default adapter with WebGPU's default limits, without
    /// a surface. `WGPU_BACKEND` and the other wgpu environment variables
    /// choose the adapter. Natively the device keeps its instance, so it can
    /// also draw to window surfaces.
    pub async fn offscreen() -> Result<Self, String> {
        let (instance, adapter) = adapter().await?;
        // Orbis draws indirectly from compute; devices without that (the
        // iOS simulator's Metal) can't draw it.
        let downlevel = adapter.get_downlevel_capabilities().flags;
        let needed = wgpu::DownlevelFlags::COMPUTE_SHADERS
            | wgpu::DownlevelFlags::INDIRECT_EXECUTION
            | wgpu::DownlevelFlags::VERTEX_STORAGE;
        if !downlevel.contains(needed) {
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
                // WebGPU's defaults, lowered where the adapter falls short
                // of them: the iOS simulator has 15 inter-stage variables.
                required_limits: wgpu::Limits::default().or_worse_values_from(&adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no device: {e}"))?;
        #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
        let mut gpu = Self::new(device, queue);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let lost = std::sync::Arc::clone(&gpu.lost);
            gpu.device.set_device_lost_callback(move |_, _| {
                lost.store(true, std::sync::atomic::Ordering::Relaxed)
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            gpu.native = Some((instance, adapter));
        }
        #[cfg(target_arch = "wasm32")]
        let _ = instance;
        Ok(gpu)
    }

    /// Resolves when the GPU has finished all work submitted so far.
    pub(crate) fn work_done(&self) -> impl Future<Output = ()> + use<> {
        // Gathered writes are work submitted too.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_writes(&mut encoder);
        let index = self.queue.submit([encoder.finish()]);
        let (sender, receiver) = futures_channel::oneshot::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = sender.send(());
        });
        let device = self.device.clone();
        async move {
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(index),
                    timeout: None,
                })
                .expect("device lost");
            receiver.await.expect("work-done callback dropped");
        }
    }

    /// Copies the pixels of a 4-byte-per-texel texture out, row by row
    /// without padding. The copy is submitted now; the future holds no
    /// borrow of the device, so the texture may go before it resolves.
    pub fn read_texture(&self, id: TextureId) -> impl Future<Output = Vec<u8>> + use<> {
        self.read_wgpu_texture(self.texture(id))
    }

    /// [`read_texture`](Self::read_texture) for a wgpu texture the host
    /// owns, such as a canvas's current texture (made with `COPY_SRC`).
    pub fn read_wgpu_texture(
        &self,
        texture: &wgpu::Texture,
    ) -> impl Future<Output = Vec<u8>> + use<> {
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_writes(&mut encoder);
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
        self.queue.submit([encoder.finish()]);
        let (device, readbacks) = (self.device.clone(), self.readbacks.clone());
        async move {
            let mapped = map_read(&device, &buffer).await;
            let mut pixels = Vec::with_capacity((row * height) as usize);
            for y in 0..height {
                let start = (y * padded_row) as usize;
                pixels.extend_from_slice(&mapped[start..start + row as usize]);
            }
            drop(mapped);
            buffer.unmap();
            let mut kept = readbacks.lock().expect("readbacks poisoned");
            if kept.len() == READBACKS {
                kept.remove(0);
            }
            kept.push(buffer);
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
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_writes(&mut encoder);
        encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, source.size());
        self.queue.submit([encoder.finish()]);
        map_read(&self.device, &buffer).await.to_vec()
    }
}

pub(crate) async fn map_read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> wgpu::BufferView {
    let (sender, receiver) = futures_channel::oneshot::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device lost during readback");
    receiver
        .await
        .expect("mapping callback dropped")
        .expect("readback mapping failed");
    buffer
        .slice(..)
        .get_mapped_range()
        .expect("readback buffer is mapped")
}
