//! Native pipeline caches: warm starts use native pipeline caches. Where
//! the driver supports them (Vulkan), pipelines are made
//! through one cache, whose data the host keeps between runs; browsers and
//! Metal keep their own.
//!
//! The data is sealed with a SHA-256 of wgpu's whole blob: wgpu checks its
//! header (driver, version, length) but not the payload, which goes to the
//! driver as is, so a file corrupted on disk must not get that far.

use sha2::{Digest, Sha256};

use crate::WgpuGpu;

/// What sealed cache data starts with.
const MAGIC: &[u8; 8] = b"orbpc\0\0\x01";
const SEAL: usize = MAGIC.len() + 32;

/// wgpu's cache data behind the magic and its checksum.
fn seal(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(SEAL + data.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&Sha256::digest(data));
    out.extend_from_slice(data);
    out
}

/// wgpu's cache data, if `sealed` is intact.
fn open(sealed: &[u8]) -> Option<&[u8]> {
    let (head, data) = sealed.split_at_checked(SEAL)?;
    let (magic, sum) = head.split_at(MAGIC.len());
    (magic == MAGIC && sum == Sha256::digest(data).as_slice()).then_some(data)
}

impl WgpuGpu {
    /// The name the cache's data is kept under for this driver, or `None`
    /// when the device has no pipeline cache.
    pub fn pipeline_cache_key(&self) -> Option<String> {
        let (_, adapter) = self.native.as_ref()?;
        if !self
            .device
            .features()
            .contains(wgpu::Features::PIPELINE_CACHE)
        {
            return None;
        }
        wgpu::util::pipeline_cache_key(&adapter.get_info())
    }

    /// Starts the pipeline cache, from data an earlier run's
    /// [`pipeline_cache_data`](Self::pipeline_cache_data) returned under
    /// the same key. Data that fails its checksum, or that wgpu finds is
    /// from another driver, is discarded.
    /// Pipelines made before this aren't cached. Does nothing without
    /// [`pipeline_cache_key`](Self::pipeline_cache_key).
    #[allow(unsafe_code)]
    pub fn start_pipeline_cache(&mut self, data: Option<&[u8]>) {
        if self.pipeline_cache_key().is_none() || self.pipeline_cache.is_some() {
            return;
        }
        let data = data.and_then(open);
        // SAFETY: wgpu requires the data to come from `get_data` on a
        // compatible device. It's intact `get_data` output (`open` checked
        // its SHA-256), and wgpu-core checks the header against this device
        // (magic, version, ABI, backend, adapter, validation key, length)
        // and with `fallback` starts empty when any differs.
        let cache = unsafe {
            self.device
                .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("orbis pipelines"),
                    data,
                    fallback: true,
                })
        };
        self.pipeline_cache = Some(cache);
    }

    /// The cache's data to keep for the next run, sealed.
    pub fn pipeline_cache_data(&self) -> Option<Vec<u8>> {
        Some(seal(&self.pipeline_cache.as_ref()?.get_data()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrupted_payloads_never_reach_the_driver() {
        // wgpu's own header, unchanged, then the payload.
        let blob: Vec<u8> = (0..200u8).collect();
        let sealed = seal(&blob);
        assert_eq!(open(&sealed), Some(&blob[..]));

        let mut flipped = sealed.clone();
        *flipped.last_mut().unwrap() ^= 1;
        assert_eq!(open(&flipped), None, "a flipped payload byte");
        assert_eq!(open(&sealed[..sealed.len() - 1]), None, "truncated");
        let mut extended = sealed.clone();
        extended.push(0);
        assert_eq!(open(&extended), None, "extended");
        assert_eq!(open(&blob), None, "unsealed data");
        assert_eq!(open(&[]), None);
    }
}
