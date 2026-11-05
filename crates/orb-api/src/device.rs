//! The runtime's GPU: Modern (wgpu) or Compat (a GLES 3 context through
//! EGL), chosen by the device policy (plan §4.2, §8.8): the Android
//! allowlist and the crash sentinel, with Compat where no Modern adapter
//! is found.

use std::path::{Path, PathBuf};

use orb_hal::Gpu;
use orb_hal::device::{Sentinel, sentinel_key};
use orb_hal_gl::GlGpu;
use orb_hal_gl::egl::EglContext;
use orb_hal_wgpu::WgpuGpu;

use crate::{Backend, Error, RuntimeOptions};

/// The marker file the crash sentinel keeps in a runtime's state directory.
const SENTINEL: &str = "gpu-sentinel";

/// Faults the sentinel exists for, injected by tests: `pipelines`, the
/// process aborts once the sentinel is armed (a driver crashing in its
/// compiler); `hang`, compiles never finish (a driver hung in it).
const FAULT: &str = "ORB_FAULT_INJECTION";

// One per runtime: the variants' sizes don't matter.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub(crate) enum Device {
    Modern(WgpuGpu),
    Compat(Box<Compat>),
}

#[derive(Debug)]
pub(crate) struct Compat {
    pub(crate) gpu: GlGpu,
    pub(crate) egl: EglContext,
}

impl Drop for Compat {
    /// The executor deletes its objects on its own context, which may not
    /// be current once another device was made.
    fn drop(&mut self) {
        let _ = self.egl.make_current(None);
    }
}

impl Device {
    /// The device `options` choose, with why: a sentence for diagnostics.
    /// Arms the crash sentinel for a Modern device in a state directory;
    /// its path comes back to be cleared once pipelines are made.
    pub(crate) fn choose(
        options: &RuntimeOptions,
    ) -> Result<(Self, String, Option<PathBuf>), Error> {
        let compat = |why: String| {
            make_compat(options).map(|(d, note)| (d, format!("Compat: {why}{note}"), None))
        };
        if options.backend == Backend::Compat {
            return compat("asked for".into());
        }
        let modern = match pollster::block_on(WgpuGpu::offscreen()) {
            Ok(gpu) => gpu,
            Err(e) if options.backend == Backend::Auto => return compat(e),
            Err(e) => return Err(Error::Native(e)),
        };
        let info = modern
            .adapter_info()
            .expect("offscreen devices keep their adapter");
        let platform = orb_hal_wgpu::platform(options.android_api_level);
        let adapter = orb_hal_wgpu::device_adapter(&info, platform);
        let key = sentinel_key(&adapter);
        if options.backend == Backend::Auto {
            let mut demote = None;
            if let orb_hal::device::Platform::Android { api_level } = platform
                && let Err(why) = orb_hal::device::android_vulkan_allowed(api_level, info.vendor)
            {
                demote = Some(format!("{} ({}): {why}", info.name, info.driver_info));
            }
            let marker = options.state_dir.as_deref().map(|d| d.join(SENTINEL));
            let found = marker
                .as_deref()
                .and_then(|m| std::fs::read_to_string(m).ok());
            if Sentinel::read(found.as_deref(), &key) == Sentinel::Tripped {
                demote = Some(format!(
                    "{} crashed or hung making pipelines (crash sentinel)",
                    info.name
                ));
            }
            if let Some(why) = demote {
                drop(modern);
                return match make_compat(options) {
                    Ok((device, note)) => Ok((device, format!("Compat: {why}{note}"), None)),
                    Err(e) => {
                        // Better a map on Modern than none.
                        let modern =
                            pollster::block_on(WgpuGpu::offscreen()).map_err(Error::Native)?;
                        let reason =
                            format!("Modern: {why}, but no Compat device ({})", e.message());
                        Ok((Self::Modern(modern), reason, None))
                    }
                };
            }
        }
        let armed = options
            .state_dir
            .as_deref()
            .map(|dir| arm(dir, &key))
            .transpose()?;
        let mut modern = modern;
        if std::env::var_os(FAULT).is_some_and(|f| f == "hang") {
            modern.inject_compile_hang();
            if let Some(ms) = std::env::var("ORB_COMPILE_WATCHDOG_MS")
                .ok()
                .and_then(|ms| ms.parse().ok())
            {
                modern.set_compile_watchdog(std::time::Duration::from_millis(ms));
            }
        }
        Ok((
            Self::Modern(modern),
            format!("Modern: {} ({:?})", info.name, info.backend),
            armed,
        ))
    }

    /// Makes a Compat device's context current on this thread again.
    pub(crate) fn make_current(&self) {
        if let Self::Compat(c) = self {
            let _ = c.egl.make_current(None);
        }
    }

    pub(crate) fn gpu(&mut self) -> &mut dyn Gpu {
        match self {
            Self::Modern(gpu) => gpu,
            Self::Compat(c) => &mut c.gpu,
        }
    }

    pub(crate) fn tier(&self) -> orb_hal::Tier {
        match self {
            Self::Modern(gpu) => gpu.caps().tier,
            Self::Compat(c) => c.gpu.caps().tier,
        }
    }

    pub(crate) fn pipelines_created(&self) -> u32 {
        match self {
            Self::Modern(gpu) => gpu.pipelines_created(),
            Self::Compat(c) => c.gpu.pipelines_created(),
        }
    }

    /// Calls `wake` whenever a pipeline compiles; `false` when the device
    /// can't (Compat without its compile thread), so frames poll for them.
    pub(crate) fn set_compile_wake(&self, wake: std::sync::Arc<dyn Fn() + Send + Sync>) -> bool {
        match self {
            Self::Modern(gpu) => {
                gpu.set_compile_wake(wake);
                true
            }
            Self::Compat(c) => c.gpu.set_compile_wake(wake),
        }
    }

    /// The compile the watchdog found hung, if one was: the device is
    /// lost, and the sentinel stays armed so the next device demotes.
    pub(crate) fn hung(&self) -> Option<String> {
        match self {
            Self::Modern(gpu) => gpu.hung().map(str::to_owned),
            Self::Compat(c) => c.gpu.hung().map(str::to_owned),
        }
    }

    pub(crate) fn lost(&self) -> bool {
        match self {
            Self::Modern(gpu) => gpu.lost(),
            Self::Compat(c) => c.gpu.lost(),
        }
    }

    /// Loses the device on purpose, to exercise recovery. GL has no way to
    /// lose a context on demand, so Compat devices only report it.
    pub(crate) fn lose(&mut self) {
        match self {
            Self::Modern(gpu) => gpu.lose(),
            Self::Compat(c) => c.gpu.mark_lost(),
        }
    }

    /// A texture's RGBA8 pixels, rows top to bottom.
    pub(crate) fn read_texture(&mut self, id: orb_hal::TextureId) -> Vec<u8> {
        match self {
            Self::Modern(gpu) => pollster::block_on(gpu.read_texture(id)),
            Self::Compat(c) => c.gpu.read_rgba8(id),
        }
    }
}

/// A Compat device, and a note for the selection when its programs link
/// on the drawing thread.
fn make_compat(options: &RuntimeOptions) -> Result<(Device, String), Error> {
    let (egl, gl) =
        EglContext::headless(options.egl_library.as_deref()).map_err(Error::Unsupported)?;
    let mut gpu = GlGpu::new(gl).map_err(Error::Unsupported)?;
    // Programs link on a thread of their own (plan §8.4); without one,
    // one links per frame.
    let note = match egl.compile_thread() {
        Ok(thread) => {
            gpu.set_compile_thread(thread);
            String::new()
        }
        Err(e) => format!("; programs link on the drawing thread ({e})"),
    };
    Ok((Device::Compat(Box::new(Compat { gpu, egl })), note))
}

/// Writes the sentinel's marker before any pipeline is made.
fn arm(dir: &Path, key: &str) -> Result<PathBuf, Error> {
    std::fs::create_dir_all(dir)
        .map_err(|e| Error::InvalidArgument(format!("{}: {e}", dir.display())))?;
    let marker = dir.join(SENTINEL);
    std::fs::write(&marker, key)
        .map_err(|e| Error::Native(format!("{}: {e}", marker.display())))?;
    Ok(marker)
}

/// Called just before the first pipelines are made while the sentinel is
/// armed: with `ORB_FAULT_INJECTION=pipelines`, the process dies there, as
/// it does when a driver's compiler crashes.
pub(crate) fn before_pipelines() {
    if std::env::var_os(FAULT).is_some_and(|f| f == "pipelines") {
        std::process::abort();
    }
}

/// Clears an armed sentinel: the pipelines were made, or the device was
/// lost for another reason than its compiler.
pub(crate) fn disarm(marker: &Path) {
    let _ = std::fs::remove_file(marker);
}
