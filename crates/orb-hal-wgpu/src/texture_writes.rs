//! Texture writes, held until the next submission and issued in waves.
//!
//! wgpu records a submission's texture writes as copies into one Metal
//! blit encoder, with no barrier between copies into the same texture.
//! On Apple GPUs two such copies into one level and layer can race: the
//! texture is stored in compressed blocks, and under load a copy now and
//! then writes a block back with what it read before the other's texels
//! landed, and those texels are lost (#156). Which blocks two copies may
//! share depends on the format and the GPU, so no spacing between them is
//! safe.
//!
//! So where copies race, a submission writes each level and layer of a
//! texture at most once: a second write into one goes in the next wave,
//! each wave its own submission, as Metal orders a resource's uses across
//! command buffers, and the later texels still win. Atlases, which add
//! many small bitmaps a frame, upload one dirty rectangle per layer a
//! frame (orb-render's `scene::texels`), so their writes never wait for a
//! wave. The first write into a level and layer is issued at once, its
//! texels not copied or held. Elsewhere every write is issued at once,
//! but the web's from host images, which all wait for the next
//! submission, in one wave, with any later write into a level and layer
//! one of them took, so the writes land in the order they were made.

use orb_hal::{TextureId, TextureRegion};

/// Where a write's texels come from.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Source {
    /// Issued when it was made, in the first wave: kept for the level and
    /// layer it took.
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
}

/// Bytes of held texels whose allocation outlives a submission.
const KEPT_TEXELS: usize = 1 << 20;

/// The texture writes made since the last submission.
#[derive(Debug, Default)]
pub(crate) struct TextureWrites {
    /// Copies into one level and layer race on this device.
    race: bool,
    writes: Vec<TextureWrite>,
    texels: Vec<u8>,
    /// Per write, its wave; and the writes by subresource: kept for their
    /// allocations.
    waves: Vec<u32>,
    order: Vec<usize>,
}

impl TextureWrites {
    pub(crate) fn new(race: bool) -> Self {
        Self {
            race,
            ..Default::default()
        }
    }

    /// Whether a write since the last submission went into `region`'s
    /// level and layer. Where copies don't race only held writes are kept
    /// (host images', and those behind one), so a later write waits its
    /// turn behind a held one and still wins.
    fn written(&self, region: &TextureRegion) -> bool {
        let subresource = (region.texture, region.level, region.layer);
        self.writes.iter().any(|w| w.subresource() == subresource)
    }

    /// Takes a write of `data`, rows `bytes_per_row` apart: holds it if
    /// one made since the last submission went into its level and layer,
    /// else returns `false` for the caller to issue it now.
    #[must_use]
    pub(crate) fn core(&mut self, region: &TextureRegion, data: &[u8], bytes_per_row: u32) -> bool {
        if !self.written(region) {
            if self.race {
                self.writes.push(TextureWrite {
                    region: *region,
                    source: Source::Issued,
                });
            }
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
    /// core memory: held if one made since the last submission went into
    /// its level and layer, else `false` for the caller to issue it now.
    #[cfg(target_arch = "wasm32")]
    #[must_use]
    pub(crate) fn bytes(
        &mut self,
        region: &TextureRegion,
        (bytes, offset): (orb_hal::HostBytesId, u64),
        bytes_per_row: u32,
    ) -> bool {
        let held = self.written(region);
        if !held && !self.race {
            return false;
        }
        self.writes.push(TextureWrite {
            region: *region,
            source: match held {
                true => Source::Bytes {
                    bytes,
                    offset,
                    bytes_per_row,
                },
                false => Source::Issued,
            },
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

    /// Assigns each write its wave: where copies race, the number of
    /// earlier writes into its level and layer. Returns the number of
    /// waves.
    fn plan(&mut self) -> u32 {
        let Self {
            race,
            writes,
            waves,
            order,
            ..
        } = self;
        waves.clear();
        waves.resize(writes.len(), 0);
        if !*race {
            return 1;
        }
        order.clear();
        order.extend(0..writes.len());
        order.sort_by_key(|&i| (writes[i].subresource(), i));
        let mut count = 0;
        for (n, &i) in order.iter().enumerate() {
            let wave = match n.checked_sub(1).map(|p| order[p]) {
                Some(p) if writes[p].subresource() == writes[i].subresource() => waves[p] + 1,
                _ => 0,
            };
            waves[i] = wave;
            count = count.max(wave + 1);
        }
        count
    }

    /// Issues the held writes: `put` each with its wave, wave by wave and
    /// in the order they came within a wave. Every wave after the first
    /// puts a write; the first may have been issued already, when its
    /// writes were made.
    pub(crate) fn issue(&mut self, mut put: impl FnMut(&TextureWrite, &[u8], u32)) {
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
                put(w, texels, wave);
            }
        }
        self.writes.clear();
        self.texels.clear();
        self.texels.shrink_to(KEPT_TEXELS);
    }
}
