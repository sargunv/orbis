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
//! Natively the upload goes through Orbis's staging belt (`staging`), as
//! larger writes and texture writes do, staged when they're made.

use std::ops::Range;

use orb_hal::BufferId;

use crate::TextureEntry;
use crate::slots::Slots;

/// Writes up to this size are gathered; larger ones are staged natively
/// and go to the queue on WebGPU (on WebGL 2, gathered too).
pub(crate) const GATHERED: usize = 64 << 10;

/// What uploading the gathered writes issued: queue writes (natively,
/// the staging belt's one), and submissions made only for texture
/// writes' earlier waves.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Issued {
    pub(crate) calls: u32,
    pub(crate) submissions: u32,
}

/// The uploads WebGL 2 takes in turn: more than the submissions in flight.
#[cfg(target_arch = "wasm32")]
const RING: usize = 4;

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

/// The writes gathered since the last submission.
#[derive(Debug, Default)]
pub(crate) struct Writes {
    data: Vec<u8>,
    writes: Vec<Write>,
    /// The writes' indices by buffer and offset, and the runs they merge
    /// into: kept for their allocations.
    order: Vec<usize>,
    runs: Vec<Run>,
    /// The uploads' buffers, each reused while it's big enough: writes to
    /// one are ordered after the submissions that copied from it. WebGPU
    /// keeps one. WebGL 2 takes [`RING`] in turn, so the one written is
    /// one the GPU is done with (ANGLE's Metal back end copies a buffer
    /// the GPU still reads into a new one on every write), and as many
    /// again for index buffers: WebGL copies into an element buffer only
    /// from another.
    #[cfg(target_arch = "wasm32")]
    staging: [Vec<Option<wgpu::Buffer>>; 2],
    #[cfg(target_arch = "wasm32")]
    turn: usize,
    #[cfg(target_arch = "wasm32")]
    upload: Vec<u8>,
    #[cfg(not(target_arch = "wasm32"))]
    belt: crate::staging::Belt,
    /// Writes staged when they were made, in the order they came.
    #[cfg(not(target_arch = "wasm32"))]
    staged: Vec<crate::staging::Staged>,
}

impl Writes {
    /// Merges the writes into runs and returns the upload's size: the
    /// runs into buffers `apart` picks after the rest, in the upload too.
    fn plan(&mut self, apart: impl Fn(BufferId) -> bool) -> u64 {
        let Self {
            writes,
            order,
            runs,
            ..
        } = self;
        order.clear();
        order.extend(0..writes.len());
        order.sort_unstable_by_key(|&i| {
            let w = writes[i];
            (apart(w.buffer), w.buffer, w.offset, i)
        });
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

impl Writes {
    /// Gathers a write of `data` at `offset` in buffer `id`.
    pub(crate) fn gather(&mut self, id: BufferId, offset: u64, data: &[u8]) {
        // Copies take 4-byte offsets and sizes, as writes already do.
        debug_assert!(
            offset.is_multiple_of(4) && data.len().is_multiple_of(4),
            "unaligned write"
        );
        let at = self.data.len() as u64;
        self.data.extend_from_slice(data);
        self.writes.push(Write {
            buffer: id,
            offset,
            at,
            len: data.len() as u64,
        });
    }

    /// Whether a gathered write touches `len` bytes at `offset` of `id`: a
    /// write that goes to the queue directly would land before it.
    pub(crate) fn overlaps(&self, id: BufferId, offset: u64, len: u64) -> bool {
        self.writes
            .iter()
            .any(|w| w.buffer == id && w.offset < offset + len && offset < w.offset + w.len)
    }

    /// Drops the gathered and staged writes to a buffer being destroyed.
    pub(crate) fn forget(&mut self, id: BufferId) {
        self.writes.retain(|w| w.buffer != id);
        #[cfg(not(target_arch = "wasm32"))]
        self.staged.retain(|s| !s.writes_buffer(id));
    }

    /// Drops the staged writes to a texture being destroyed.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn forget_texture(&mut self, id: orb_hal::TextureId) {
        self.staged.retain(|s| !s.writes_texture(id));
    }

    /// Stages a write too large to gather: it lands ahead of the gathered
    /// ones, as one to the queue would. `false` when staging failed (a
    /// lost device).
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn stage(
        &mut self,
        device: &wgpu::Device,
        id: BufferId,
        offset: u64,
        data: &[u8],
    ) -> bool {
        let staged = crate::staging::Staged::buffer(&mut self.belt, device, (id, offset), data);
        let staged_ok = staged.is_some();
        self.staged.extend(staged);
        staged_ok
    }

    /// Stages a texture write, rows `bytes_per_row` apart in `data`, to
    /// land in `wave` (`texture_writes`). `false` when staging failed (a
    /// lost device).
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn stage_texture(
        &mut self,
        device: &wgpu::Device,
        (region, wave): (&orb_hal::TextureRegion, u32),
        data: &[u8],
        bytes_per_row: u32,
    ) -> bool {
        let staged = crate::staging::Staged::texture(
            &mut self.belt,
            device,
            (region, wave),
            data,
            bytes_per_row,
        );
        let staged_ok = staged.is_some();
        self.staged.extend(staged);
        staged_ok
    }

    /// Drops the gathered writes: their device is gone.
    pub(crate) fn discard(&mut self) {
        self.clear();
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.belt = Default::default();
            self.staged.clear();
        }
    }

    /// Uploads the gathered writes and copies them and the staged ones
    /// into place in `encoder`, ahead of anything else it records: the
    /// staged buffer writes, the gathered ones, then the texture writes.
    /// Texture writes in waves before the last are submitted first, a
    /// submission each (`texture_writes`). `None` when the staging belt's
    /// buffer is invalid, which only a lost device (or one out of memory)
    /// explains: the writes and the belt are dropped.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn encode(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        (buffers, textures): (&Slots<wgpu::Buffer>, &Slots<TextureEntry>),
        encoder: &mut wgpu::CommandEncoder,
    ) -> Option<Issued> {
        let mut gathered = None;
        if !self.writes.is_empty() {
            let size = self.plan(|_| false);
            let mut belt = std::mem::take(&mut self.belt);
            gathered = belt.stage(device, size, wgpu::COPY_BUFFER_ALIGNMENT, |f| {
                self.fill(|to, bytes| f.put(to, bytes));
            });
            self.belt = belt;
            if gathered.is_none() {
                self.clear();
                self.staged.clear();
                return None;
            }
        }
        if self.staged.is_empty() && gathered.is_none() {
            return Some(Issued::default());
        }
        let mut issued = Issued {
            calls: u32::from(gathered.is_some()),
            submissions: 0,
        };
        self.belt.finish(encoder);
        let staged = std::mem::take(&mut self.staged);
        let last = staged.iter().map(crate::staging::Staged::wave).max();
        for wave in 0..last.unwrap_or(0) {
            let mut early = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("texture writes"),
            });
            for s in staged.iter().filter(|s| s.wave() == wave) {
                s.encode(&mut early, buffers, textures);
            }
            queue.submit([early.finish()]);
            issued.submissions += 1;
        }
        let last = last.unwrap_or(0);
        let (to_buffers, to_textures): (Vec<_>, Vec<_>) = (staged.iter())
            .filter(|s| s.wave() == last)
            .partition(|s| matches!(s, crate::staging::Staged::Buffer { .. }));
        for s in to_buffers {
            s.encode(encoder, buffers, textures);
        }
        if let Some((staging, start)) = gathered {
            for run in &self.runs {
                encoder.copy_buffer_to_buffer(
                    &staging,
                    start + run.at,
                    &buffers[run.buffer.0],
                    run.offset,
                    run.len,
                );
            }
        }
        for s in to_textures {
            s.encode(encoder, buffers, textures);
        }
        self.clear();
        // Kept for its allocation.
        self.staged = staged;
        self.staged.clear();
        Some(issued)
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn encode(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        (buffers, _): (&Slots<wgpu::Buffer>, &Slots<TextureEntry>),
        encoder: &mut wgpu::CommandEncoder,
    ) -> Option<Issued> {
        let mut issued = Issued::default();
        if self.writes.is_empty() {
            self.clear();
            return Some(issued);
        }
        // Index buffers' runs last, from their own upload on WebGL 2.
        let index =
            |id: BufferId| cfg!(webgl) && buffers[id.0].usage().contains(wgpu::BufferUsages::INDEX);
        let size = self.plan(index);
        let mut upload = std::mem::take(&mut self.upload);
        upload.clear();
        upload.resize(size as usize, 0);
        self.fill(|to, bytes| upload[to..to + bytes.len()].copy_from_slice(bytes));
        let ring = if cfg!(webgl) { RING } else { 1 };
        let turn = self.turn % ring;
        self.turn = turn + 1;
        let split = (self.runs.iter())
            .position(|run| index(run.buffer))
            .unwrap_or(self.runs.len());
        let Self { runs, staging, .. } = self;
        for (kind, runs) in [&runs[..split], &runs[split..]].into_iter().enumerate() {
            let (Some(first), Some(last)) = (runs.first(), runs.last()) else {
                continue;
            };
            let (start, end) = (first.at, last.at + last.len);
            let ring = &mut staging[kind];
            ring.resize(ring.len().max(turn + 1), None);
            let staging = &mut ring[turn];
            if staging.as_ref().is_none_or(|s| s.size() < end - start) {
                let usage = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
                *staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gathered writes"),
                    size: (end - start).next_power_of_two().max(GATHERED as u64),
                    usage: if kind == 1 {
                        usage | wgpu::BufferUsages::INDEX
                    } else {
                        usage
                    },
                    mapped_at_creation: false,
                }));
            }
            let staging = staging.as_ref().expect("made above");
            queue.write_buffer(staging, 0, &upload[start as usize..end as usize]);
            issued.calls += 1;
            for run in runs {
                encoder.copy_buffer_to_buffer(
                    staging,
                    run.at - start,
                    &buffers[run.buffer.0],
                    run.offset,
                    run.len,
                );
            }
        }
        self.upload = upload;
        self.clear();
        Some(issued)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upload(writes: &[(u32, u64, &[u8])]) -> Vec<(u32, u64, Vec<u8>)> {
        let mut gathered = Writes::default();
        for &(b, o, d) in writes {
            gathered.gather(BufferId(b), o, d);
        }
        let size = gathered.plan(|_| false);
        let mut bytes = vec![0; size as usize];
        gathered.fill(|to, b| bytes[to..to + b.len()].copy_from_slice(b));
        gathered
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
            .collect()
    }

    #[test]
    fn later_writes_win_where_they_overlap() {
        let runs = upload(&[
            (0, 4, &[1, 1, 1, 1, 1, 1, 1, 1]),
            (0, 0, &[2, 2, 2, 2, 2, 2, 2, 2]),
            (0, 8, &[3, 3, 3, 3]),
            (0, 4, &[4, 4, 4, 4]),
        ]);
        assert_eq!(runs, [(0, 0, vec![2, 2, 2, 2, 4, 4, 4, 4, 3, 3, 3, 3])]);
    }
}
