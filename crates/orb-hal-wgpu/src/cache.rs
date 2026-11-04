//! Native pipeline caches (plan §8.4: warm starts use native pipeline
//! caches). Where the driver supports them (Vulkan), pipelines are made
//! through one cache, whose data the host keeps between runs; browsers and
//! Metal keep their own.

use crate::WgpuGpu;

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
    /// the same key. Data from another driver, or corrupted, is discarded.
    /// Pipelines made before this aren't cached. Does nothing without
    /// [`pipeline_cache_key`](Self::pipeline_cache_key).
    #[allow(unsafe_code)]
    pub fn start_pipeline_cache(&mut self, data: Option<&[u8]>) {
        if self.pipeline_cache_key().is_none() || self.pipeline_cache.is_some() {
            return;
        }
        // SAFETY: wgpu requires the data to come from `get_data` on a
        // compatible device. wgpu-core checks that before the driver sees
        // it: the header's magic, version and ABI, the backend, the
        // adapter's vendor, device and driver, a validation key and the
        // size, and with `fallback` starts empty when any differs.
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

    /// The cache's data to keep for the next run.
    pub fn pipeline_cache_data(&self) -> Option<Vec<u8>> {
        self.pipeline_cache.as_ref()?.get_data()
    }
}
