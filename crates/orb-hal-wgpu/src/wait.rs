//! Waiting on the GPU: the one place that knows a wait can meet a lost
//! device, and turns that into a value ([`Lost`]) rather than a panic.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::WgpuGpu;

/// The device was lost before what was waited on came: the host needs
/// another device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lost;

impl std::fmt::Display for Lost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the GPU was lost")
    }
}

impl std::error::Error for Lost {}

/// A device and its loss flag, which futures that hold no borrow of the
/// [`WgpuGpu`] wait on.
#[derive(Clone, Debug)]
pub(crate) struct Waiter {
    device: wgpu::Device,
    lost: Arc<AtomicBool>,
}

impl WgpuGpu {
    pub(crate) fn waiter(&self) -> Waiter {
        Waiter {
            device: self.device.clone(),
            lost: self.lost.clone(),
        }
    }

    /// Blocks until the GPU has finished submission `index`, or all work
    /// submitted so far (`None`). In the browser, where nothing blocks,
    /// it only polls the device and checks for a loss.
    pub(crate) fn wait(&self, index: Option<wgpu::SubmissionIndex>) -> Result<(), Lost> {
        self.waiter().wait(index)
    }
}

impl Waiter {
    fn lost(&self) -> Result<(), Lost> {
        match self.lost.load(Ordering::Relaxed) {
            true => Err(Lost),
            false => Ok(()),
        }
    }

    fn mark_lost(&self) -> Lost {
        self.lost.store(true, Ordering::Relaxed);
        Lost
    }

    /// [`WgpuGpu::wait`].
    #[cfg(not(webgl))]
    pub(crate) fn wait(&self, index: Option<wgpu::SubmissionIndex>) -> Result<(), Lost> {
        loop {
            self.lost()?;
            match self.device.poll(wgpu::PollType::Wait {
                submission_index: index.clone(),
                timeout: None,
            }) {
                Ok(_) => return self.lost(),
                // GL's driver waits a few seconds at most: wait on.
                Err(wgpu::PollError::Timeout) => {}
                // The driver reported the loss as the device waited, or
                // refused the submission waited on: a lost device takes
                // none.
                Err(wgpu::PollError::DeviceLost | wgpu::PollError::WrongSubmissionIndex(..)) => {
                    return Err(self.mark_lost());
                }
            }
        }
    }

    /// [`WgpuGpu::wait`] on WebGL 2, which can't wait for its fence.
    #[cfg(webgl)]
    pub(crate) fn wait(&self, _: Option<wgpu::SubmissionIndex>) -> Result<(), Lost> {
        self.lost()?;
        let _ = self.device.poll(wgpu::PollType::Poll);
        self.lost()
    }

    /// What `ready` gives once it gives something, the device polled before
    /// each try: WebGL 2 can't wait for its fence, so each task polls it
    /// once until the callback that `ready` watches has run, or the device
    /// is lost (a WebGL 2 context lost signals no fence).
    #[cfg(webgl)]
    async fn polled<T>(&self, mut ready: impl FnMut() -> Option<T>) -> Result<T, Lost> {
        loop {
            self.lost()?;
            let _ = self.device.poll(wgpu::PollType::Poll);
            if let Some(value) = ready() {
                return Ok(value);
            }
            let tick = js_sys::Promise::new(&mut |resolve, _| {
                let global = js_sys::global();
                let set_timeout = js_sys::Reflect::get(&global, &"setTimeout".into())
                    .ok()
                    .and_then(|f| wasm_bindgen::JsCast::dyn_into::<js_sys::Function>(f).ok())
                    .expect("setTimeout");
                let _ = set_timeout.call2(&global, &resolve, &1.into());
            });
            let _ = wasm_bindgen_futures::JsFuture::from(tick).await;
        }
    }

    /// Resolves once the GPU has finished submission `index` and the
    /// callbacks queued after it have run.
    pub(crate) async fn done(
        &self,
        index: wgpu::SubmissionIndex,
        receiver: futures_channel::oneshot::Receiver<()>,
    ) -> Result<(), Lost> {
        #[cfg(webgl)]
        {
            let _ = index;
            let mut receiver = receiver;
            let ran = self
                .polled(|| match receiver.try_recv() {
                    Ok(ran) => ran.map(|()| true),
                    Err(_) => Some(false),
                })
                .await?;
            // A dropped callback went with the device.
            if !ran {
                return Err(self.mark_lost());
            }
            Ok(())
        }
        #[cfg(not(webgl))]
        {
            self.wait(Some(index))?;
            receiver.await.map_err(|_| self.mark_lost())
        }
    }

    /// Maps `buffer` for reading once the GPU has written it. A mapping
    /// the device refuses means the device went: it is marked lost.
    pub(crate) async fn map_read(&self, buffer: &wgpu::Buffer) -> Result<wgpu::BufferView, Lost> {
        self.lost()?;
        #[cfg_attr(not(webgl), allow(unused_mut))]
        let (sender, mut receiver) = futures_channel::oneshot::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        #[cfg(webgl)]
        let mapped = self
            .polled(|| match receiver.try_recv() {
                Ok(result) => result,
                // A dropped callback went with the device.
                Err(_) => Some(Err(wgpu::BufferAsyncError)),
            })
            .await?;
        #[cfg(not(webgl))]
        let mapped = {
            self.wait(None)?;
            receiver.await.unwrap_or(Err(wgpu::BufferAsyncError))
        };
        mapped.map_err(|_| self.mark_lost())?;
        buffer
            .slice(..)
            .get_mapped_range()
            .map_err(|_| self.mark_lost())
    }
}
