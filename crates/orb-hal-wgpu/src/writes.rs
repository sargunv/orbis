//! Small buffer writes, gathered into one upload and copied into place at
//! the start of the next submission (#21). `GPUQueue.writeBuffer` costs
//! some microseconds a call however few bytes it carries, and a frame
//! writes hundreds of small runs: symbol state words, tile rows, paint.

use std::sync::Mutex;

use orb_hal::BufferId;

/// Writes up to this size are gathered; larger ones go to the queue.
pub(crate) const GATHERED: usize = 64 << 10;

#[derive(Debug, Default)]
struct Pending {
    data: Vec<u8>,
    /// Buffer, offset in it, offset in `data`, length.
    copies: Vec<(BufferId, u64, u64, u64)>,
    /// The upload, reused while it's big enough: writes to it are ordered
    /// after the submissions that copied from it.
    staging: Option<wgpu::Buffer>,
}

/// The writes gathered since the last submission.
#[derive(Debug, Default)]
pub(crate) struct Writes(Mutex<Pending>);

impl Writes {
    /// Gathers a write of `data` at `offset` in buffer `id`.
    pub(crate) fn gather(&mut self, id: BufferId, offset: u64, data: &[u8]) {
        let pending = self.0.get_mut().expect("writes poisoned");
        let at = pending.data.len() as u64;
        pending.data.extend_from_slice(data);
        // Copies keep wgpu's 4-byte alignment, as writes already do.
        pending
            .data
            .resize(pending.data.len().next_multiple_of(4), 0);
        pending.copies.push((id, offset, at, data.len() as u64));
    }

    /// Whether a gathered write touches `len` bytes at `offset` of `id`: a
    /// write that goes to the queue directly would land before it.
    pub(crate) fn overlaps(&mut self, id: BufferId, offset: u64, len: u64) -> bool {
        let pending = self.0.get_mut().expect("writes poisoned");
        pending
            .copies
            .iter()
            .any(|&(b, o, _, l)| b == id && o < offset + len && offset < o + l)
    }

    /// Drops the gathered writes to a buffer being destroyed.
    pub(crate) fn forget(&mut self, id: BufferId) {
        let pending = self.0.get_mut().expect("writes poisoned");
        pending.copies.retain(|&(b, ..)| b != id);
    }

    /// Uploads the gathered writes and copies them into place in
    /// `encoder`, ahead of anything else it records.
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: impl Fn(BufferId) -> wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let mut pending = self.0.lock().expect("writes poisoned");
        if pending.copies.is_empty() {
            pending.data.clear();
            return;
        }
        let size = pending.data.len() as u64;
        if pending.staging.as_ref().is_none_or(|s| s.size() < size) {
            pending.staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gathered writes"),
                size: size.next_power_of_two().max(GATHERED as u64),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let Pending {
            data,
            copies,
            staging,
        } = &mut *pending;
        let staging = staging.as_ref().expect("made above");
        queue.write_buffer(staging, 0, data);
        for (id, offset, at, len) in copies.drain(..) {
            encoder.copy_buffer_to_buffer(staging, at, &buffer(id), offset, len);
        }
        data.clear();
    }
}
