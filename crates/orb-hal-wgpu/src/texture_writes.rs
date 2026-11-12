//! Texture writes, held until the next submission and issued in waves.
//!
//! wgpu records a submission's texture writes as copies into one Metal
//! blit encoder, with no barrier between copies into the same texture.
//! On Apple GPUs two such copies that touch one 16×16 block of a texture
//! can race: under load, a copy now and then writes the whole block back
//! with what it read before the other's texels landed, and those texels
//! are lost. The glyph atlas packs bitmaps side by side and uploads a
//! burst of them a frame, so a glyph lost a corner now and then and two
//! maps of one style drew it differently (#156).
//!
//! So writes into the same block of a texture's level and layer go in
//! separate waves, each its own submission: Metal orders a resource's
//! uses across command buffers. A write that overlaps an earlier one
//! lands in a later wave, so the later texels still win. Writes into
//! different blocks share a wave, and a burst of packed bitmaps takes a
//! handful of waves, not one each. A write sharing no block with one made
//! since the last submission would land in the first wave anyway: it is
//! issued at once, its texels not copied or held.

use orb_hal::{TextureId, TextureRegion};

/// The side, in texels, of the blocks two writes in a wave may not share:
/// twice the 16 seen on Apple GPUs.
const BLOCK: u32 = 32;

/// Where a write's texels come from.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Source {
    /// Issued when it was made, in the first wave: kept for the blocks it
    /// takes.
    Issued,
    /// `len` bytes from `at` in the held texels, rows `bytes_per_row`
    /// apart.
    Core {
        at: usize,
        len: usize,
        bytes_per_row: u32,
    },
    /// A host image ([`orb_hal::Gpu::write_texture_from_host`]).
    #[cfg(target_arch = "wasm32")]
    Host {
        image: orb_hal::HostImageId,
        premultiply: bool,
    },
    /// Host bytes from `offset`, rows `bytes_per_row` apart
    /// ([`orb_hal::Gpu::write_texture_from_host_bytes`]).
    #[cfg(target_arch = "wasm32")]
    Bytes {
        bytes: orb_hal::HostBytesId,
        offset: u64,
        bytes_per_row: u32,
    },
}

/// A write made since the last submission.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TextureWrite {
    pub(crate) region: TextureRegion,
    pub(crate) source: Source,
}

impl TextureWrite {
    /// The texture, level and layer it writes.
    fn subresource(&self) -> (TextureId, u32, u32) {
        let r = &self.region;
        (r.texture, r.level, r.layer)
    }

    /// Its texels' extent, `[x0, y0, x1, y1)`.
    fn extent(&self) -> [u32; 4] {
        let r = &self.region;
        [r.x, r.y, r.x + r.width, r.y + r.height]
    }

    /// The blocks it touches, `[x0, y0, x1, y1)`.
    fn blocks(&self) -> [u32; 4] {
        let [x0, y0, x1, y1] = self.extent();
        [
            x0 / BLOCK,
            y0 / BLOCK,
            x1.div_ceil(BLOCK),
            y1.div_ceil(BLOCK),
        ]
    }
}

fn intersect(a: [u32; 4], b: [u32; 4]) -> bool {
    a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
}

/// Bytes of held texels whose allocation outlives a submission.
const KEPT_TEXELS: usize = 1 << 20;

/// The texture writes made since the last submission.
#[derive(Debug, Default)]
pub(crate) struct TextureWrites {
    writes: Vec<TextureWrite>,
    texels: Vec<u8>,
    /// Per write, its wave; and the writes by subresource: kept for their
    /// allocations.
    waves: Vec<u32>,
    order: Vec<usize>,
}

impl TextureWrites {
    /// Takes a write of `data`, rows `bytes_per_row` apart: holds it if it
    /// shares a block with one made since the last submission, else
    /// returns `false` for the caller to issue it now.
    #[must_use]
    pub(crate) fn core(&mut self, region: &TextureRegion, data: &[u8], bytes_per_row: u32) -> bool {
        let write = TextureWrite {
            region: *region,
            source: Source::Issued,
        };
        let (subresource, blocks) = (write.subresource(), write.blocks());
        if !self
            .writes
            .iter()
            .any(|w| w.subresource() == subresource && intersect(blocks, w.blocks()))
        {
            self.writes.push(write);
            return false;
        }
        let at = self.texels.len();
        self.texels.extend_from_slice(data);
        self.writes.push(TextureWrite {
            region: *region,
            source: Source::Core {
                at,
                len: data.len(),
                bytes_per_row,
            },
        });
        true
    }

    /// Takes a write from host bytes, as [`core`](Self::core) one from
    /// core memory: held if it shares a block with one made since the last
    /// submission, else `false` for the caller to issue it now.
    #[cfg(target_arch = "wasm32")]
    #[must_use]
    pub(crate) fn bytes(
        &mut self,
        region: &TextureRegion,
        (bytes, offset): (orb_hal::HostBytesId, u64),
        bytes_per_row: u32,
    ) -> bool {
        let write = TextureWrite {
            region: *region,
            source: Source::Issued,
        };
        let (subresource, blocks) = (write.subresource(), write.blocks());
        let held = (self.writes.iter())
            .any(|w| w.subresource() == subresource && intersect(blocks, w.blocks()));
        self.writes.push(TextureWrite {
            source: match held {
                true => Source::Bytes {
                    bytes,
                    offset,
                    bytes_per_row,
                },
                false => Source::Issued,
            },
            ..write
        });
        held
    }

    /// Whether a held write reads `bytes`, which must stay registered
    /// until the writes are issued.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn reads_bytes(&self, bytes: orb_hal::HostBytesId) -> bool {
        (self.writes.iter())
            .any(|w| matches!(w.source, Source::Bytes { bytes: b, .. } if b == bytes))
    }

    /// Holds a write from a host image.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn host(
        &mut self,
        region: &TextureRegion,
        image: orb_hal::HostImageId,
        premultiply: bool,
    ) {
        self.writes.push(TextureWrite {
            region: *region,
            source: Source::Host { image, premultiply },
        });
    }

    /// Whether a held write reads `image`, which must stay registered
    /// until the writes are issued.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn reads(&self, image: orb_hal::HostImageId) -> bool {
        self.writes
            .iter()
            .any(|w| matches!(w.source, Source::Host { image: i, .. } if i == image))
    }

    /// Drops the writes into a texture being destroyed.
    pub(crate) fn forget(&mut self, texture: TextureId) {
        self.writes.retain(|w| w.region.texture != texture);
    }

    /// Assigns each write its wave: after every earlier write it overlaps,
    /// and in the first wave from there with no write in its blocks (the
    /// first, for one issued when made). Returns the number of waves.
    fn plan(&mut self) -> u32 {
        let Self {
            writes,
            waves,
            order,
            ..
        } = self;
        waves.clear();
        waves.resize(writes.len(), 0);
        order.clear();
        order.extend(0..writes.len());
        order.sort_by_key(|&i| (writes[i].subresource(), i));
        let mut count = 0;
        let mut start = 0;
        while start < order.len() {
            let subresource = writes[order[start]].subresource();
            let end = start
                + order[start..]
                    .iter()
                    .take_while(|&&i| writes[i].subresource() == subresource)
                    .count();
            for n in start..end {
                let w = &writes[order[n]];
                let (extent, blocks) = (w.extent(), w.blocks());
                let earlier = &order[start..n];
                let mut wave = earlier
                    .iter()
                    .filter(|&&j| intersect(extent, writes[j].extent()))
                    .map(|&j| waves[j] + 1)
                    .max()
                    .unwrap_or(0);
                while earlier
                    .iter()
                    .any(|&j| waves[j] == wave && intersect(blocks, writes[j].blocks()))
                {
                    wave += 1;
                }
                waves[order[n]] = wave;
                count = count.max(wave + 1);
            }
            start = end;
        }
        count
    }

    /// Issues the held writes: `put` each, wave by wave and in the order
    /// they came within a wave, and `separate` between waves. The first
    /// wave may have been issued already, when its writes were made.
    pub(crate) fn issue(
        &mut self,
        mut put: impl FnMut(&TextureWrite, &[u8]),
        mut separate: impl FnMut(),
    ) {
        if self
            .writes
            .iter()
            .all(|w| matches!(w.source, Source::Issued))
        {
            self.writes.clear();
            return;
        }
        let count = self.plan();
        for wave in 0..count {
            if wave > 0 {
                separate();
            }
            for (w, _) in self
                .writes
                .iter()
                .zip(&self.waves)
                .filter(|&(_, &v)| v == wave)
            {
                let texels = match w.source {
                    Source::Issued => continue,
                    Source::Core { at, len, .. } => &self.texels[at..at + len],
                    #[cfg(target_arch = "wasm32")]
                    Source::Host { .. } | Source::Bytes { .. } => &[][..],
                };
                put(w, texels);
            }
        }
        self.writes.clear();
        self.texels.clear();
        self.texels.shrink_to(KEPT_TEXELS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(texture: u32, layer: u32, x: u32, y: u32, width: u32, height: u32) -> TextureRegion {
        TextureRegion {
            texture: TextureId(texture),
            x,
            y,
            layer,
            width,
            height,
            level: 0,
        }
    }

    /// The waves `writes` are issued in, as each write's index.
    fn waves(writes: &[TextureRegion]) -> Vec<Vec<usize>> {
        let mut held = TextureWrites::default();
        let waves = std::cell::RefCell::new(vec![Vec::new()]);
        for (i, r) in writes.iter().enumerate() {
            let len = (r.width * r.height) as usize;
            if !held.core(r, &vec![i as u8; len], r.width) {
                waves.borrow_mut()[0].push(i);
            }
        }
        held.issue(
            |w, texels| {
                assert_eq!(texels.len(), (w.region.width * w.region.height) as usize);
                let mut waves = waves.borrow_mut();
                waves.last_mut().unwrap().push(usize::from(texels[0]));
            },
            || waves.borrow_mut().push(Vec::new()),
        );
        waves.into_inner()
    }

    #[test]
    fn writes_sharing_a_block_go_in_separate_waves() {
        // Side by side across a block edge, then one below in both's
        // blocks, then one far off.
        let writes = [
            region(1, 0, 0, 0, 40, 40),
            region(1, 0, 40, 0, 40, 40),
            region(1, 0, 0, 40, 40, 40),
            region(1, 0, 200, 200, 20, 20),
        ];
        assert_eq!(waves(&writes), [vec![0, 3], vec![1], vec![2]]);
    }

    #[test]
    fn writes_in_free_blocks_are_issued_at_once() {
        // A whole layer, another layer, then a write into the first one's
        // blocks: only the last is held, its texels alone kept.
        let mut held = TextureWrites::default();
        assert!(!held.core(&region(1, 0, 0, 0, 64, 64), &[0; 64 * 64], 64));
        assert!(!held.core(&region(1, 1, 0, 0, 64, 64), &[0; 64 * 64], 64));
        assert!(held.core(&region(1, 0, 8, 8, 4, 4), &[0; 16], 4));
        assert_eq!(held.texels.len(), 16);
        let (mut put, mut separated) = (0, 0);
        held.issue(|_, _| put += 1, || separated += 1);
        assert_eq!((put, separated), (1, 1));
    }

    #[test]
    fn other_layers_and_textures_share_a_wave() {
        let writes = [
            region(1, 0, 0, 0, 40, 40),
            region(1, 1, 0, 0, 40, 40),
            region(2, 0, 0, 0, 40, 40),
        ];
        assert_eq!(waves(&writes), [vec![0, 1, 2]]);
    }

    #[test]
    fn an_overlapping_write_lands_after_the_one_it_covers() {
        // The third covers part of the first; the second shares a block
        // with the first without overlapping it.
        let writes = [
            region(1, 0, 0, 0, 60, 60),
            region(1, 0, 62, 0, 8, 8),
            region(1, 0, 0, 0, 8, 8),
        ];
        let waves = waves(&writes);
        let wave_of = |i: usize| waves.iter().position(|w| w.contains(&i)).unwrap();
        assert!(wave_of(2) > wave_of(0));
        assert!(wave_of(1) != wave_of(0));
    }

    #[test]
    fn a_packed_burst_takes_a_handful_of_waves() {
        // A page of glyph-sized bitmaps, shelf packed as the atlas packs
        // them.
        let mut writes = Vec::new();
        let (mut x, mut y) = (0, 0);
        for i in 0..64u32 {
            let (w, h) = (44 + i * 7 % 20, 50 + i * 5 % 12);
            if x + w > 512 {
                (x, y) = (0, y + 62);
            }
            writes.push(region(1, 0, x, y, w, h));
            x += w;
        }
        assert!(waves(&writes).len() <= 6, "{}", waves(&writes).len());
    }
}
