//! Linked programs kept between runs: GLES 3.0 program binaries by source,
//! sealed with a checksum, for one driver (its key).

use std::collections::BTreeMap;

use crate::Info;

/// FNV-1a over byte strings.
fn fnv(parts: &[&[u8]]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for part in parts {
        for &b in *part {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        hash = (hash ^ 0xff).wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// A program's key: its two stages' sources.
pub(crate) fn key(vertex: &str, fragment: &str) -> u64 {
    fnv(&[vertex.as_bytes(), fragment.as_bytes()])
}

/// A driver's key: its vendor, renderer and version, as a file name.
pub(crate) fn driver_key(info: &Info) -> String {
    format!(
        "gl-{:016x}",
        fnv(&[
            info.vendor.as_bytes(),
            info.renderer.as_bytes(),
            info.version.as_bytes()
        ])
    )
}

const MAGIC: &[u8; 8] = b"orbgl\0\0\x01";

/// The next `n` bytes of `rest`.
fn take<'a>(rest: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    let (head, tail) = rest.split_at_checked(n)?;
    *rest = tail;
    Some(head)
}

/// The next `n`-byte little-endian number of `rest`.
fn word(rest: &mut &[u8], n: usize) -> Option<u64> {
    Some(
        take(rest, n)?
            .iter()
            .rev()
            .fold(0, |v, &b| v << 8 | u64::from(b)),
    )
}

#[derive(Debug, Default)]
pub(crate) struct ProgramCache {
    programs: BTreeMap<u64, (u32, Vec<u8>)>,
    dirty: bool,
}

impl ProgramCache {
    /// The cache in `data`: its magic, then per program its key, format,
    /// length, bytes and a checksum of them. Anything malformed starts it
    /// empty: no damaged binary reaches the driver.
    pub(crate) fn open(data: Option<&[u8]>) -> Self {
        let mut rest = match data.and_then(|d| d.strip_prefix(MAGIC.as_slice())) {
            Some(rest) => rest,
            None => return Self::default(),
        };
        let mut programs = BTreeMap::new();
        while !rest.is_empty() {
            let entry = (|| {
                let key = word(&mut rest, 8)?;
                let format = word(&mut rest, 4)? as u32;
                let len = word(&mut rest, 4)? as usize;
                let bytes = take(&mut rest, len)?;
                (word(&mut rest, 8)? == fnv(&[bytes])).then(|| (key, (format, bytes.to_vec())))
            })();
            let Some((key, program)) = entry else {
                return Self::default();
            };
            programs.insert(key, program);
        }
        Self {
            programs,
            dirty: false,
        }
    }

    pub(crate) fn get(&self, key: u64) -> Option<glow::ProgramBinary> {
        let (format, buffer) = self.programs.get(&key)?;
        Some(glow::ProgramBinary {
            buffer: buffer.clone(),
            format: *format,
        })
    }

    pub(crate) fn insert(&mut self, key: u64, binary: glow::ProgramBinary) {
        self.programs.insert(key, (binary.format, binary.buffer));
        self.dirty = true;
    }

    pub(crate) fn take_data(&mut self) -> Option<Vec<u8>> {
        if !std::mem::take(&mut self.dirty) {
            return None;
        }
        let mut data = MAGIC.to_vec();
        for (key, (format, bytes)) in &self.programs {
            data.extend(key.to_le_bytes());
            data.extend(format.to_le_bytes());
            data.extend((bytes.len() as u32).to_le_bytes());
            data.extend(bytes);
            data.extend(fnv(&[bytes]).to_le_bytes());
        }
        Some(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_refuses_damage() {
        let mut cache = ProgramCache::default();
        cache.insert(
            7,
            glow::ProgramBinary {
                buffer: vec![1, 2, 3],
                format: 9,
            },
        );
        let data = cache.take_data().unwrap();
        assert!(cache.take_data().is_none(), "nothing new");
        let opened = ProgramCache::open(Some(&data));
        let binary = opened.get(7).unwrap();
        assert_eq!((binary.buffer, binary.format), (vec![1, 2, 3], 9));
        let mut damaged = data.clone();
        let at = damaged.len() - 9;
        damaged[at] ^= 1;
        assert!(ProgramCache::open(Some(&damaged)).get(7).is_none());
        assert!(
            ProgramCache::open(Some(&data[..data.len() - 1]))
                .get(7)
                .is_none()
        );
    }
}
