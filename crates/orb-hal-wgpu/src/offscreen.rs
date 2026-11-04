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

/// Qualcomm's PCI vendor id, as Vulkan reports it.
const QUALCOMM: u32 = 0x5143;

/// An instance, configured from the environment, and its default adapter.
async fn adapter(
    skip_indirect_validation: bool,
) -> Result<(wgpu::Instance, wgpu::Adapter), String> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    if skip_indirect_validation {
        descriptor
            .flags
            .remove(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL);
    }
    let instance = wgpu::Instance::new(descriptor);
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
        let (mut instance, mut adapter) = adapter(false).await?;
        // Quirk (plan §8.8): after wgpu's indirect-call validation pass,
        // Qualcomm's Vulkan driver draws nothing from drawIndexedIndirect
        // (Adreno 619 measured; gfx-rs/wgpu#8801 on the 650 and 740), while
        // direct draws and drawIndirect work. Orbis's indirect arguments
        // are its cull shader's, bounded by the scene's own buffers, so on
        // those drivers the device skips that validation.
        let info = adapter.get_info();
        if info.backend == wgpu::Backend::Vulkan && info.vendor == QUALCOMM {
            (instance, adapter) = self::adapter(true).await?;
        }
        // The Modern tier draws indirectly from compute (plan §4.2); devices
        // without that (the iOS simulator's Metal) are Compat's, from M4.
        let downlevel = adapter.get_downlevel_capabilities().flags;
        let modern = wgpu::DownlevelFlags::COMPUTE_SHADERS
            | wgpu::DownlevelFlags::INDIRECT_EXECUTION
            | wgpu::DownlevelFlags::VERTEX_STORAGE;
        if !downlevel.contains(modern) {
            return Err(format!(
                "{} lacks {:?}, which the Modern tier needs",
                adapter.get_info().name,
                modern - downlevel
            ));
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("orbis"),
                // Natively, warm starts reuse pipelines through a cache.
                required_features: adapter.features() & wgpu::Features::PIPELINE_CACHE,
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
            gpu.native = Some((instance, adapter));
        }
        #[cfg(target_arch = "wasm32")]
        let _ = instance;
        Ok(gpu)
    }

    /// Resolves when the GPU has finished all work submitted so far.
    pub fn work_done(&self) -> impl Future<Output = ()> + 'static {
        let index = self.queue.submit([]);
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
    /// without padding.
    pub async fn read_texture(&self, id: TextureId) -> Vec<u8> {
        let texture = self.texture(id);
        let (width, height) = (texture.width(), texture.height());
        let row = width * 4;
        let padded_row = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(padded_row * height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
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
        let mapped = self.map_read(&buffer).await;
        let mut pixels = Vec::with_capacity((row * height) as usize);
        for y in 0..height {
            let start = (y * padded_row) as usize;
            pixels.extend_from_slice(&mapped[start..start + row as usize]);
        }
        pixels
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
        encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, source.size());
        self.queue.submit([encoder.finish()]);
        self.map_read(&buffer).await.to_vec()
    }

    async fn map_read(&self, buffer: &wgpu::Buffer) -> wgpu::BufferView {
        let (sender, receiver) = futures_channel::oneshot::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        self.device
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
}
