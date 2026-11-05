//! Native GLES 3 contexts through EGL: Mesa on Linux (llvmpipe included),
//! Android's driver, and ANGLE elsewhere.
//!
//! libEGL is loaded at run time, so builds don't link it and hosts without
//! it fall back cleanly.

use std::ffi::c_void;

use khronos_egl as egl;

type Instance = egl::DynamicInstance<egl::EGL1_4>;

/// An EGL display, a GLES 3 context on it and the surface it draws to
/// when no window is current (a pbuffer, or the window it was made for).
pub struct EglContext {
    egl: Instance,
    /// Where libEGL was loaded from, for the compile thread's instance.
    library: String,
    display: egl::Display,
    config: egl::Config,
    context: egl::Context,
    surface: egl::Surface,
}

impl std::fmt::Debug for EglContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EglContext").finish_non_exhaustive()
    }
}

/// The platform's libEGL.
fn default_library() -> &'static str {
    if cfg!(target_os = "android") {
        "libEGL.so"
    } else if cfg!(target_os = "macos") {
        "libEGL.dylib"
    } else {
        "libEGL.so.1"
    }
}

impl EglContext {
    /// A context drawing to a 1 × 1 pbuffer, for offscreen targets: from
    /// the libEGL at `library`, or the platform's.
    pub fn headless(library: Option<&str>) -> Result<(Self, glow::Context), String> {
        Self::make(library, None)
    }

    /// A context drawing to a native window (an `ANativeWindow*` on
    /// Android, an X11 `Window` or `wl_egl_window*` on Linux).
    ///
    /// # Safety
    ///
    /// `window` must be a live native window for this platform's EGL, and
    /// outlive the context.
    pub unsafe fn window(
        library: Option<&str>,
        window: *mut c_void,
    ) -> Result<(Self, glow::Context), String> {
        Self::make(library, Some(window))
    }

    fn make(
        library: Option<&str>,
        window: Option<*mut c_void>,
    ) -> Result<(Self, glow::Context), String> {
        let path = match library {
            Some(path) => path.to_string(),
            None => default_library().to_string(),
        };
        // SAFETY: loading libEGL runs its initializers, which is what EGL
        // expects of a process.
        let instance = unsafe { Instance::load_required_from_filename(&path) }
            .map_err(|e| format!("no EGL ({path}): {e}"))?;
        // SAFETY: the default display of this process.
        let display =
            unsafe { instance.get_display(egl::DEFAULT_DISPLAY) }.ok_or("no EGL display")?;
        instance
            .initialize(display)
            .map_err(|e| format!("EGL initialize: {e}"))?;
        let surface_type = if window.is_some() {
            egl::WINDOW_BIT
        } else {
            egl::PBUFFER_BIT
        };
        let attributes = [
            egl::RENDERABLE_TYPE,
            egl::OPENGL_ES3_BIT,
            egl::SURFACE_TYPE,
            surface_type,
            egl::RED_SIZE,
            8,
            egl::GREEN_SIZE,
            8,
            egl::BLUE_SIZE,
            8,
            egl::ALPHA_SIZE,
            8,
            egl::NONE,
        ];
        let config = instance
            .choose_first_config(display, &attributes)
            .map_err(|e| format!("EGL config: {e}"))?
            .ok_or("no GLES 3 EGL config")?;
        instance
            .bind_api(egl::OPENGL_ES_API)
            .map_err(|e| format!("EGL bind API: {e}"))?;
        let context = instance
            .create_context(
                display,
                config,
                None,
                &[egl::CONTEXT_CLIENT_VERSION, 3, egl::NONE],
            )
            .map_err(|e| format!("GLES 3 context: {e}"))?;
        let surface = match window {
            // SAFETY: the caller's live native window.
            Some(window) => unsafe {
                instance.create_window_surface(
                    display,
                    config,
                    window as egl::NativeWindowType,
                    None,
                )
            },
            None => instance.create_pbuffer_surface(
                display,
                config,
                &[egl::WIDTH, 1, egl::HEIGHT, 1, egl::NONE],
            ),
        }
        .map_err(|e| format!("EGL surface: {e}"))?;
        instance
            .make_current(display, Some(surface), Some(surface), Some(context))
            .map_err(|e| format!("EGL make current: {e}"))?;
        let made = Self {
            egl: instance,
            library: path,
            display,
            config,
            context,
            surface,
        };
        let gl = made.glow();
        Ok((made, gl))
    }

    /// GL entry points of this context, for another executor on it.
    pub fn glow(&self) -> glow::Context {
        // SAFETY: entry points of the context made current above.
        unsafe {
            glow::Context::from_loader_function(|name| {
                self.egl
                    .get_proc_address(name)
                    .map_or(std::ptr::null(), |f| f as *const c_void)
            })
        }
    }

    /// A surface on another native window, drawn to by this context
    /// while [`make_current`](Self::make_current) has it current.
    ///
    /// # Safety
    ///
    /// As for [`window`](Self::window).
    pub unsafe fn create_window(&self, window: *mut c_void) -> Result<EglWindow, String> {
        // SAFETY: the caller's live native window.
        let surface = unsafe {
            self.egl.create_window_surface(
                self.display,
                self.config,
                window as egl::NativeWindowType,
                None,
            )
        }
        .map_err(|e| format!("EGL window surface: {e}"))?;
        Ok(EglWindow { surface })
    }

    /// Makes the context draw to `window`, or to its own surface.
    pub fn make_current(&self, window: Option<&EglWindow>) -> Result<(), String> {
        let surface = window.map_or(self.surface, |w| w.surface);
        self.egl
            .make_current(
                self.display,
                Some(surface),
                Some(surface),
                Some(self.context),
            )
            .map_err(|e| format!("EGL make current: {e}"))
    }

    /// A window's size in pixels.
    pub fn window_size(&self, window: &EglWindow) -> (u32, u32) {
        let query = |attribute| {
            self.egl
                .query_surface(self.display, window.surface, attribute)
                .unwrap_or(0)
                .max(1) as u32
        };
        (query(egl::WIDTH), query(egl::HEIGHT))
    }

    /// Presents a window's frame.
    pub fn swap(&self, window: &EglWindow) -> Result<(), String> {
        self.egl
            .swap_buffers(self.display, window.surface)
            .map_err(|e| format!("EGL swap: {e}"))
    }

    /// Destroys a window surface, drawing to the context's own again.
    pub fn destroy_window(&self, window: EglWindow) {
        let _ = self.make_current(None);
        let _ = self.egl.destroy_surface(self.display, window.surface);
    }
}

/// A native window's EGL surface, from [`EglContext::create_window`].
#[derive(Debug)]
pub struct EglWindow {
    surface: egl::Surface,
}

impl Drop for EglContext {
    fn drop(&mut self) {
        let _ = self.egl.make_current(self.display, None, None, None);
        let _ = self.egl.destroy_surface(self.display, self.surface);
        let _ = self.egl.destroy_context(self.display, self.context);
    }
}

/// A thread that compiles and links programs on a context of its own that
/// shares this one's objects (plan §8.4: no compile blocks the thread that
/// draws). GLES has no async compile without `KHR_parallel_shader_compile`,
/// which Android's drivers and ANGLE on SwiftShader lack; a shared context
/// works on every EGL.
pub struct CompileThread {
    shared: std::sync::Arc<crate::compile::Queue>,
    done: std::sync::mpsc::Receiver<crate::compile::Done>,
}

impl std::fmt::Debug for CompileThread {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompileThread").finish_non_exhaustive()
    }
}

/// What the thread needs of EGL: handles are plain pointers, usable from
/// any thread.
struct Handles {
    egl: Instance,
    display: egl::Display,
    context: egl::Context,
    surface: Option<egl::Surface>,
}

// SAFETY: EGL displays, contexts and surfaces may be used from any thread;
// a context is current on at most one, which the compile thread ensures by
// owning its context alone.
unsafe impl Send for Handles {}

impl EglContext {
    /// Starts a compile thread on a new context sharing this one's objects:
    /// surfaceless where EGL allows it, else on a 1 × 1 pbuffer.
    pub fn compile_thread(&self) -> Result<CompileThread, String> {
        let context = self
            .egl
            .create_context(
                self.display,
                self.config,
                Some(self.context),
                &[egl::CONTEXT_CLIENT_VERSION, 3, egl::NONE],
            )
            .map_err(|e| format!("shared GLES 3 context: {e}"))?;
        let surfaceless = self
            .egl
            .query_string(Some(self.display), egl::EXTENSIONS)
            .is_ok_and(|e| {
                e.to_string_lossy()
                    .split(' ')
                    .any(|e| e == "EGL_KHR_surfaceless_context")
            });
        let surface = if surfaceless {
            None
        } else {
            match self.egl.create_pbuffer_surface(
                self.display,
                self.config,
                &[egl::WIDTH, 1, egl::HEIGHT, 1, egl::NONE],
            ) {
                Ok(surface) => Some(surface),
                Err(e) => {
                    let _ = self.egl.destroy_context(self.display, context);
                    return Err(format!("compile thread surface: {e}"));
                }
            }
        };
        // SAFETY: the same library, loaded again: the thread owns its
        // instance as it owns its context.
        let egl = unsafe { Instance::load_required_from_filename(&self.library) }
            .map_err(|e| format!("EGL for the compile thread: {e}"))?;
        let handles = Handles {
            egl,
            display: self.display,
            context,
            surface,
        };
        let shared = std::sync::Arc::new(crate::compile::Queue::default());
        let (tx, done) = std::sync::mpsc::channel();
        let (started_tx, started) = std::sync::mpsc::channel();
        let queue = shared.clone();
        std::thread::Builder::new()
            .name("orbis programs".into())
            .spawn(move || {
                let handles = handles;
                let current = handles.egl.make_current(
                    handles.display,
                    handles.surface,
                    handles.surface,
                    Some(handles.context),
                );
                if let Err(e) = current {
                    let _ = started_tx.send(Err(format!("compile thread make current: {e}")));
                } else {
                    let _ = started_tx.send(Ok(()));
                    // SAFETY: entry points of the context current here.
                    let gl = unsafe {
                        glow::Context::from_loader_function(|name| {
                            handles
                                .egl
                                .get_proc_address(name)
                                .map_or(std::ptr::null(), |f| f as *const c_void)
                        })
                    };
                    crate::compile::work(&gl, &queue, &tx);
                    let _ = handles.egl.make_current(handles.display, None, None, None);
                }
                let _ = handles
                    .egl
                    .destroy_context(handles.display, handles.context);
                if let Some(surface) = handles.surface {
                    let _ = handles.egl.destroy_surface(handles.display, surface);
                }
            })
            .map_err(|e| format!("compile thread: {e}"))?;
        started
            .recv()
            .map_err(|_| "the compile thread exited".to_string())??;
        Ok(CompileThread { shared, done })
    }
}

impl CompileThread {
    pub(crate) fn submit(&self, job: crate::compile::Job) {
        self.shared.push(job);
    }

    pub(crate) fn prioritize(&self, slot: u32) {
        self.shared.prioritize(slot);
    }

    pub(crate) fn take(&self) -> impl Iterator<Item = crate::compile::Done> + '_ {
        self.done.try_iter()
    }

    /// Blocks for the next finished compile.
    pub(crate) fn wait(&self) -> Option<crate::compile::Done> {
        self.done.recv().ok()
    }

    pub(crate) fn hung(&self, limit: std::time::Duration) -> Option<String> {
        self.shared.hung(limit)
    }

    pub(crate) fn set_wake(&self, wake: std::sync::Arc<dyn Fn() + Send + Sync>) {
        self.shared.set_wake(wake);
    }
}

impl Drop for CompileThread {
    /// Drops queued jobs; the thread finishes its current one and exits.
    /// A thread stuck in a driver stays stuck: it is never joined.
    fn drop(&mut self) {
        self.shared.close();
    }
}
