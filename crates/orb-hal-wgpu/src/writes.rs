//! Small buffer writes, gathered into one upload and copied into place at
//! the start of the next submission. `GPUQueue.writeBuffer` costs
//! some microseconds a call however few bytes it carries, and a frame
//! writes hundreds of small runs: symbol state words, tile rows, paint.
//!
//! Writes that touch or overlap in one buffer are merged into one copy, the
//! later write's bytes winning where they overlap. A frame's ~800 writes
//! land in a handful of tables, mostly side by side, and each copy is a
//! command for wgpu to track and the driver to encode (on Metal, a blit
//! `copyFromBuffer`): merged, they come to a fifth as many.
//!
//! Natively the upload goes through a staging belt: mapped buffers that
//! return once the GPU has copied from them. wgpu's `write_buffer` makes
//! a staging buffer for every call, and making a Metal buffer is a trip
//! into the kernel: on sparse views, a tenth of a frame.

use std::ops::Range;
use std::sync::Mutex;

use orb_hal::BufferId;

/// Writes up to this size are gathered; larger ones go to the queue.
pub(crate) const GATHERED: usize = 64 << 10;

/// A gathered write: `len` bytes at `offset` in `buffer`, from `at` in the
/// gathered data.
#[derive(Clone, Copy, Debug)]
struct Write {
    buffer: BufferId,
    offset: u64,
    at: u64,
    len: u64,
}

/// One copy into place: `len` bytes at `offset` in `buffer`, from `at` in
/// the upload, made of the writes `order[writes]`.
#[derive(Clone, Debug)]
struct Run {
    buffer: BufferId,
    offset: u64,
    at: u64,
    len: u64,
    writes: Range<usize>,
    /// Whether its writes overlap, so must land in the order they came.
    overlapping: bool,
}

#[derive(Debug, Default)]
struct Pending {
    data: Vec<u8>,
    writes: Vec<Write>,
    /// The writes' indices by buffer and offset, and the runs they merge
    /// into: kept for their allocations.
    order: Vec<usize>,
    runs: Vec<Run>,
    /// The upload, reused while it's big enough: writes to it are ordered
    /// after the submissions that copied from it.
    #[cfg(target_arch = "wasm32")]
    staging: Option<wgpu::Buffer>,
    #[cfg(target_arch = "wasm32")]
    upload: Vec<u8>,
    #[cfg(not(target_arch = "wasm32"))]
    belt: Option<wgpu::util::StagingBelt>,
}

impl Pending {
    /// Merges the writes into runs and returns the upload's size.
    fn plan(&mut self) -> u64 {
        let Self {
            writes,
            order,
            runs,
            ..
        } = self;
        order.clear();
        order.extend(0..writes.len());
        order.sort_unstable_by_key(|&i| (writes[i].buffer, writes[i].offset, i));
        runs.clear();
        let mut size = 0;
        for (n, &i) in order.iter().enumerate() {
            let w = writes[i];
            if let Some(run) = runs.last_mut()
                && run.buffer == w.buffer
                && w.offset <= run.offset + run.len
            {
                let end = run.offset + run.len;
                run.overlapping |= w.offset < end;
                let grown = (w.offset + w.len).saturating_sub(end);
                run.len += grown;
                run.writes.end = n + 1;
                size += grown;
            } else {
                runs.push(Run {
                    buffer: w.buffer,
                    offset: w.offset,
                    at: size,
                    len: w.len,
                    writes: n..n + 1,
                    overlapping: false,
                });
                size += w.len;
            }
        }
        size
    }

    /// Lays the planned runs out in the upload, `plan`'s size long: `put`
    /// writes bytes at an offset in it.
    fn fill(&mut self, mut put: impl FnMut(usize, &[u8])) {
        let Self {
            data,
            writes,
            order,
            runs,
            ..
        } = self;
        for run in runs.iter() {
            let members = &mut order[run.writes.clone()];
            if run.overlapping {
                // Gathered order, so later writes land over earlier ones.
                members.sort_unstable();
            }
            for &i in members.iter() {
                let w = writes[i];
                let (at, len) = (w.at as usize, w.len as usize);
                put(
                    (run.at + w.offset - run.offset) as usize,
                    &data[at..at + len],
                );
            }
        }
    }

    fn clear(&mut self) {
        self.data.clear();
        self.writes.clear();
        self.runs.clear();
    }
}

/// The writes gathered since the last submission.
#[derive(Debug, Default)]
pub(crate) struct Writes(Mutex<Pending>);

impl Writes {
    /// Gathers a write of `data` at `offset` in buffer `id`.
    pub(crate) fn gather(&mut self, id: BufferId, offset: u64, data: &[u8]) {
        let pending = self.0.get_mut().expect("writes poisoned");
        // Copies take 4-byte offsets and sizes, as writes already do.
        debug_assert!(
            offset.is_multiple_of(4) && data.len().is_multiple_of(4),
            "unaligned write"
        );
        let at = pending.data.len() as u64;
        pending.data.extend_from_slice(data);
        pending.writes.push(Write {
            buffer: id,
            offset,
            at,
            len: data.len() as u64,
        });
    }

    /// Whether a gathered write touches `len` bytes at `offset` of `id`: a
    /// write that goes to the queue directly would land before it.
    pub(crate) fn overlaps(&mut self, id: BufferId, offset: u64, len: u64) -> bool {
        let pending = self.0.get_mut().expect("writes poisoned");
        pending
            .writes
            .iter()
            .any(|w| w.buffer == id && w.offset < offset + len && offset < w.offset + w.len)
    }

    /// Drops the gathered writes to a buffer being destroyed.
    pub(crate) fn forget(&mut self, id: BufferId) {
        let pending = self.0.get_mut().expect("writes poisoned");
        pending.writes.retain(|w| w.buffer != id);
    }

    /// Drops the gathered writes: their device is gone.
    pub(crate) fn discard(&self) {
        let mut pending = self.0.lock().expect("writes poisoned");
        pending.clear();
        #[cfg(not(target_arch = "wasm32"))]
        {
            pending.belt = None;
        }
    }

    /// Uploads the gathered writes and copies them into place in
    /// `encoder`, ahead of anything else it records. `false` when the
    /// staging belt's buffer is invalid, which only a lost device (or one
    /// out of memory) explains: the writes and the belt are dropped.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        buffer: impl Fn(BufferId) -> wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
    ) -> bool {
        let mut pending = self.0.lock().expect("writes poisoned");
        if pending.writes.is_empty() {
            pending.clear();
            return true;
        }
        let size = wgpu::BufferSize::new(pending.plan()).expect("writes gathered");
        let mut belt = pending
            .belt
            .take()
            .unwrap_or_else(|| wgpu::util::StagingBelt::new(device.clone(), GATHERED as u64));
        let align = wgpu::BufferSize::new(wgpu::COPY_BUFFER_ALIGNMENT).expect("nonzero");
        let slice = belt.allocate(size, align);
        let Ok(mut view) = slice.get_mapped_range_mut() else {
            pending.clear();
            return false;
        };
        pending.fill(|to, bytes| view.slice(to..to + bytes.len()).copy_from_slice(bytes));
        drop(view);
        let (staging, start) = (slice.buffer(), slice.offset());
        for run in &pending.runs {
            encoder.copy_buffer_to_buffer(
                staging,
                start + run.at,
                &buffer(run.buffer),
                run.offset,
                run.len,
            );
        }
        belt.finish_and_recall_on_submit(encoder);
        pending.belt = Some(belt);
        pending.clear();
        true
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: impl Fn(BufferId) -> wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
    ) -> bool {
        let mut pending = self.0.lock().expect("writes poisoned");
        if pending.writes.is_empty() {
            pending.clear();
            return true;
        }
        let size = pending.plan();
        let mut upload = std::mem::take(&mut pending.upload);
        upload.clear();
        upload.resize(size as usize, 0);
        pending.fill(|to, bytes| upload[to..to + bytes.len()].copy_from_slice(bytes));
        if pending.staging.as_ref().is_none_or(|s| s.size() < size) {
            pending.staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gathered writes"),
                size: size.next_power_of_two().max(GATHERED as u64),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let staging = pending.staging.as_ref().expect("made above");
        queue.write_buffer(staging, 0, &upload);
        for run in &pending.runs {
            encoder.copy_buffer_to_buffer(
                staging,
                run.at,
                &buffer(run.buffer),
                run.offset,
                run.len,
            );
        }
        pending.upload = upload;
        pending.clear();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upload(writes: &[(u32, u64, &[u8])]) -> (Vec<(u32, u64, Vec<u8>)>, u64) {
        let mut gathered = Writes::default();
        for &(b, o, d) in writes {
            gathered.gather(BufferId(b), o, d);
        }
        let pending = gathered.0.get_mut().unwrap();
        let size = pending.plan();
        let mut bytes = vec![0; size as usize];
        pending.fill(|to, b| bytes[to..to + b.len()].copy_from_slice(b));
        let runs = pending
            .runs
            .iter()
            .map(|r| {
                let at = r.at as usize;
                (
                    r.buffer.0,
                    r.offset,
                    bytes[at..at + r.len as usize].to_vec(),
                )
            })
            .collect();
        (runs, size)
    }

    #[test]
    fn merges_touching_writes_into_one_copy() {
        let (runs, size) = upload(&[
            (1, 8, &[3, 3, 3, 3]),
            (0, 0, &[1, 1, 1, 1]),
            (1, 0, &[2, 2, 2, 2, 2, 2, 2, 2]),
            (0, 8, &[4, 4, 4, 4]),
        ]);
        assert_eq!(
            runs,
            [
                (0, 0, vec![1, 1, 1, 1]),
                (0, 8, vec![4, 4, 4, 4]),
                (1, 0, vec![2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3]),
            ]
        );
        assert_eq!(size, 20);
    }

    #[test]
    fn later_writes_win_where_they_overlap() {
        let (runs, _) = upload(&[
            (0, 4, &[1, 1, 1, 1, 1, 1, 1, 1]),
            (0, 0, &[2, 2, 2, 2, 2, 2, 2, 2]),
            (0, 8, &[3, 3, 3, 3]),
            (0, 4, &[4, 4, 4, 4]),
        ]);
        assert_eq!(runs, [(0, 0, vec![2, 2, 2, 2, 4, 4, 4, 4, 3, 3, 3, 3])]);
    }
}
