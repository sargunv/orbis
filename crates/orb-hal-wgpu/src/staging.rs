//! Natively, every write from core memory reaches the GPU through Orbis's
//! own staging: mapped buffers, filled on the CPU and copied from on the
//! GPU, that come back mapped once the GPU is done with them. wgpu's
//! `Queue::write_buffer` and `write_texture` make a staging buffer for every
//! call, which on Metal is a trip into the kernel, and add a command buffer
//! of their own to the submission after them.
//!
//! Writes staged here are copied into place at the start of the next
//! submission, ahead of the gathered small writes (`writes`), as the
//! queue's would land.

use std::sync::mpsc;

use orb_hal::{BufferId, TextureId, TextureRegion};

/// The size of the belt's chunks; a larger staging gets a chunk its size.
const CHUNK: u64 = 256 << 10;

/// Bytes of free chunks kept: those that come back past it are dropped, so
/// a burst of uploads (a style's first tiles) doesn't keep its staging.
const KEPT: u64 = 16 << 20;

/// A chunk filled since the last submission, mapped.
#[derive(Debug)]
struct Chunk {
    buffer: wgpu::Buffer,
    /// The whole chunk's mapping, made at its first staging.
    view: Option<wgpu::BufferViewMut>,
    used: u64,
}

/// Mapped buffers that stagings are cut from, each back once the
/// submission that copies from it is done.
#[derive(Debug)]
pub(crate) struct Belt {
    active: Vec<Chunk>,
    free: Vec<wgpu::Buffer>,
    sender: mpsc::Sender<wgpu::Buffer>,
    returned: mpsc::Receiver<wgpu::Buffer>,
}

impl Default for Belt {
    fn default() -> Self {
        let (sender, returned) = mpsc::channel();
        Self {
            active: Vec::new(),
            free: Vec::new(),
            sender,
            returned,
        }
    }
}

/// Writes bytes into a staging.
pub(crate) struct Fill<'a> {
    view: &'a mut wgpu::BufferViewMut,
    start: usize,
}

impl Fill<'_> {
    /// Writes `bytes` at `at` in the staging.
    pub(crate) fn put(&mut self, at: usize, bytes: &[u8]) {
        let start = self.start + at;
        (self.view.slice(start..start + bytes.len())).copy_from_slice(bytes);
    }
}

impl Belt {
    /// Stages `size` bytes, a multiple of 4, at an offset that is a
    /// multiple of `align`, written by `fill`: the buffer and offset to
    /// copy from. `None` when a chunk's mapping fails, which only a lost
    /// device (or one out of memory) explains.
    pub(crate) fn stage(
        &mut self,
        device: &wgpu::Device,
        size: u64,
        align: u64,
        fill: impl FnOnce(&mut Fill<'_>),
    ) -> Option<(wgpu::Buffer, u64)> {
        let room = |c: &Chunk| c.used.next_multiple_of(align) + size <= c.buffer.size();
        let i = match self.active.iter().position(room) {
            Some(i) => i,
            None => {
                self.take_returned();
                let buffer = match self.free.iter().position(|b| b.size() >= size) {
                    Some(i) => self.free.swap_remove(i),
                    None => device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("staging"),
                        size: size.next_multiple_of(CHUNK),
                        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: true,
                    }),
                };
                self.active.push(Chunk {
                    buffer,
                    view: None,
                    used: 0,
                });
                self.active.len() - 1
            }
        };
        let chunk = &mut self.active[i];
        let at = chunk.used.next_multiple_of(align);
        chunk.used = at + size;
        let view = match &mut chunk.view {
            Some(view) => view,
            None => chunk
                .view
                .insert(chunk.buffer.slice(..).get_mapped_range_mut().ok()?),
        };
        fill(&mut Fill {
            view,
            start: at as usize,
        });
        Some((chunk.buffer.clone(), at))
    }

    /// Unmaps the chunks filled since the last submission, each to come
    /// back mapped once `encoder`'s submission is done.
    pub(crate) fn finish(&mut self, encoder: &wgpu::CommandEncoder) {
        for chunk in self.active.drain(..) {
            drop(chunk.view);
            chunk.buffer.unmap();
            let sender = self.sender.clone();
            let buffer = chunk.buffer.clone();
            encoder.map_buffer_on_submit(&chunk.buffer, wgpu::MapMode::Write, .., move |r| {
                if r.is_ok() {
                    let _ = sender.send(buffer);
                }
            });
        }
    }

    /// Takes the chunks that came back, keeping up to [`KEPT`] bytes.
    fn take_returned(&mut self) {
        while let Ok(buffer) = self.returned.try_recv() {
            let kept: u64 = self.free.iter().map(wgpu::Buffer::size).sum();
            if kept + buffer.size() <= KEPT {
                self.free.push(buffer);
            }
        }
    }
}

/// A write staged when it was made, waiting for the next submission.
#[derive(Debug)]
pub(crate) enum Staged {
    Buffer {
        from: (wgpu::Buffer, u64),
        to: BufferId,
        offset: u64,
        len: u64,
    },
    /// Rows `bytes_per_row` apart, a multiple of 256 as copies need them,
    /// in the wave of texture writes (`texture_writes`) it lands in.
    Texture {
        from: (wgpu::Buffer, u64),
        bytes_per_row: u32,
        region: TextureRegion,
        wave: u32,
    },
}

impl Staged {
    /// Stages a buffer write.
    pub(crate) fn buffer(
        belt: &mut Belt,
        device: &wgpu::Device,
        (to, offset): (BufferId, u64),
        data: &[u8],
    ) -> Option<Self> {
        let len = data.len() as u64;
        let from = belt.stage(device, len, wgpu::COPY_BUFFER_ALIGNMENT, |f| f.put(0, data))?;
        Some(Self::Buffer {
            from,
            to,
            offset,
            len,
        })
    }

    /// Stages a texture write of rows `bytes_per_row` apart in `data`.
    pub(crate) fn texture(
        belt: &mut Belt,
        device: &wgpu::Device,
        (region, wave): (&TextureRegion, u32),
        data: &[u8],
        bytes_per_row: u32,
    ) -> Option<Self> {
        let row = bytes_per_row as usize;
        let padded = bytes_per_row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let size = u64::from(padded) * u64::from(region.height);
        let from = belt.stage(
            device,
            size,
            u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
            |f| {
                if padded == bytes_per_row {
                    f.put(0, data);
                } else {
                    for (y, texels) in data.chunks_exact(row).enumerate() {
                        f.put(y * padded as usize, texels);
                    }
                }
            },
        )?;
        Some(Self::Texture {
            from,
            bytes_per_row: padded,
            region: *region,
            wave,
        })
    }

    pub(crate) fn wave(&self) -> u32 {
        match self {
            Self::Buffer { .. } => 0,
            Self::Texture { wave, .. } => *wave,
        }
    }

    /// Whether it writes buffer `id`.
    pub(crate) fn writes_buffer(&self, id: BufferId) -> bool {
        matches!(self, Self::Buffer { to, .. } if *to == id)
    }

    /// Whether it writes texture `id`.
    pub(crate) fn writes_texture(&self, id: TextureId) -> bool {
        matches!(self, Self::Texture { region, .. } if region.texture == id)
    }

    /// Records its copy into place.
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        buffer: impl Fn(BufferId) -> wgpu::Buffer,
        texture: impl Fn(TextureId) -> wgpu::Texture,
    ) {
        match self {
            Self::Buffer {
                from: (from, at),
                to,
                offset,
                len,
            } => encoder.copy_buffer_to_buffer(from, *at, &buffer(*to), *offset, *len),
            Self::Texture {
                from: (from, at),
                bytes_per_row,
                region,
                ..
            } => encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: from,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: *at,
                        bytes_per_row: Some(*bytes_per_row),
                        rows_per_image: None,
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &texture(region.texture),
                    mip_level: region.level,
                    origin: wgpu::Origin3d {
                        x: region.x,
                        y: region.y,
                        z: region.layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: region.width,
                    height: region.height,
                    depth_or_array_layers: 1,
                },
            ),
        }
    }
}
