// Frame pacers: a clock that demands frames of its sessions at the
// display's refreshes, as mln-ffi's platform pacers do.
//
// On macOS the clock is CoreVideo's display link. A display link needs an
// active display, which a Mac whose displays sleep (or a headless one)
// lacks, and a host may still draw then (offscreen, or for a capture), so
// the pacer then keeps time on a thread of its own at a nominal refresh.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
use crate::base::put;
use crate::base::{Failure, boundary, sized};
use crate::handles;
#[cfg(target_os = "macos")]
use crate::handles::Object;
use crate::session::SessionHandle;
use crate::sys::*;

/// The refresh a pacer keeps without a display.
#[cfg_attr(
    not(target_os = "macos"),
    expect(dead_code, reason = "only macOS makes a pacer")
)]
const NOMINAL_REFRESH: Duration = Duration::from_micros(16_667);

/// Demands frames of its sessions at the display's refreshes.
pub(crate) struct FramePacer {
    sessions: Mutex<Vec<Arc<SessionHandle>>>,
    /// The shortest time between two demands, from the frame rate limit.
    interval: Option<Duration>,
    /// When it last demanded frames.
    last: Mutex<Option<Instant>>,
    #[cfg(target_os = "macos")]
    clock: Mutex<Option<display_link::Link>>,
}

impl FramePacer {
    pub(crate) fn add(&self, session: &Arc<SessionHandle>) {
        self.sessions.lock().unwrap().push(session.clone());
    }

    pub(crate) fn remove(&self, session: &Arc<SessionHandle>) {
        self.sessions
            .lock()
            .unwrap()
            .retain(|s| !Arc::ptr_eq(s, session));
    }

    /// A refresh, on the clock's thread: each session draws at the next
    /// one if it has something new.
    #[cfg_attr(
        not(target_os = "macos"),
        expect(dead_code, reason = "only macOS makes a pacer")
    )]
    fn refresh(&self) {
        if let Some(interval) = self.interval {
            let now = Instant::now();
            let mut last = self.last.lock().unwrap();
            // A refresh a little early still counts: refreshes jitter.
            if last.is_some_and(|l| now + interval / 8 < l + interval) {
                return;
            }
            *last = Some(now);
        }
        for session in self.sessions.lock().unwrap().iter() {
            if !session.tick.swap(true, std::sync::atomic::Ordering::AcqRel) {
                session.wake();
            }
        }
    }
}

impl Drop for FramePacer {
    fn drop(&mut self) {
        // The display link stops, waiting out a refresh in flight, before
        // what the refresh reads goes.
        #[cfg(target_os = "macos")]
        self.clock.get_mut().unwrap().take();
    }
}

#[cfg(target_os = "macos")]
mod display_link {
    //! CoreVideo's display link: a callback at each refresh of the active
    //! displays, on a thread of its own.
    use std::ffi::c_void;

    type CVDisplayLinkRef = *mut c_void;
    type Callback = unsafe extern "C" fn(
        CVDisplayLinkRef,
        *const c_void,
        *const c_void,
        u64,
        *mut u64,
        *mut c_void,
    ) -> i32;

    #[link(name = "CoreVideo", kind = "framework")]
    unsafe extern "C" {
        fn CVDisplayLinkCreateWithActiveCGDisplays(out: *mut CVDisplayLinkRef) -> i32;
        fn CVDisplayLinkSetOutputCallback(
            link: CVDisplayLinkRef,
            callback: Callback,
            context: *mut c_void,
        ) -> i32;
        fn CVDisplayLinkStart(link: CVDisplayLinkRef) -> i32;
        fn CVDisplayLinkStop(link: CVDisplayLinkRef);
        fn CVDisplayLinkRelease(link: CVDisplayLinkRef);
    }

    pub(crate) struct Link(CVDisplayLinkRef);
    // SAFETY: a CVDisplayLink is thread-safe.
    unsafe impl Send for Link {}

    impl Link {
        /// A link that refreshes the pacer at `context`, which outlives it,
        /// or `None` when no display is active.
        pub(crate) fn start(context: *const c_void) -> Option<Self> {
            unsafe extern "C" fn output(
                _: CVDisplayLinkRef,
                _: *const c_void,
                _: *const c_void,
                _: u64,
                _: *mut u64,
                context: *mut c_void,
            ) -> i32 {
                // SAFETY: `context` is the pacer, which stops the link before
                // it goes.
                let pacer = unsafe { &*context.cast::<super::FramePacer>() };
                pacer.refresh();
                0
            }
            let mut link = std::ptr::null_mut();
            // SAFETY: CoreVideo's API, with an out pointer.
            unsafe {
                if CVDisplayLinkCreateWithActiveCGDisplays(&mut link) != 0 || link.is_null() {
                    return None;
                }
                let link = Self(link);
                CVDisplayLinkSetOutputCallback(link.0, output, context.cast_mut());
                (CVDisplayLinkStart(link.0) == 0).then_some(link)
            }
        }
    }

    impl Drop for Link {
        fn drop(&mut self) {
            // SAFETY: the link this made; stopping waits out a callback.
            unsafe {
                CVDisplayLinkStop(self.0);
                CVDisplayLinkRelease(self.0);
            }
        }
    }
}

/// Keeps time for a pacer without a display, until the pacer goes.
#[cfg(target_os = "macos")]
fn start_timer(pacer: std::sync::Weak<FramePacer>, interval: Duration) -> Result<(), Failure> {
    std::thread::Builder::new()
        .name("orbis frame pacer".into())
        .spawn(move || {
            let mut next = Instant::now() + interval;
            loop {
                std::thread::sleep(next.saturating_duration_since(Instant::now()));
                next += interval;
                match pacer.upgrade() {
                    Some(pacer) => pacer.refresh(),
                    None => return,
                }
            }
        })
        .map(drop)
        .map_err(|e| Failure::native(format!("the frame pacer's timer: {e}")))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_apple_frame_pacer_create(
    options: *const mln_frame_pacer_options,
    out_pacer: *mut mln_frame_pacer,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let mut interval = None;
            if !options.is_null() {
                let o = sized(options, "options")?;
                if o.fields & !MLN_FRAME_PACER_OPTION_MAXIMUM_FRAMES_PER_SECOND != 0 {
                    return Err(Failure::argument("options.fields sets an unknown field"));
                }
                if o.fields & MLN_FRAME_PACER_OPTION_MAXIMUM_FRAMES_PER_SECOND != 0 {
                    if o.maximum_frames_per_second == 0 {
                        return Err(Failure::argument(
                            "options.maximum_frames_per_second must be greater than zero",
                        ));
                    }
                    interval = Some(Duration::from_secs(1) / o.maximum_frames_per_second);
                }
            }
            crate::base::fresh_out(out_pacer, "out_pacer")?;
            #[cfg(target_os = "macos")]
            {
                let pacer = Arc::new(FramePacer {
                    sessions: Mutex::default(),
                    interval,
                    last: Mutex::default(),
                    clock: Mutex::new(None),
                });
                let link = display_link::Link::start(Arc::as_ptr(&pacer).cast());
                if link.is_none() {
                    let tick = interval.map_or(NOMINAL_REFRESH, |i| i.max(NOMINAL_REFRESH));
                    start_timer(Arc::downgrade(&pacer), tick)?;
                }
                *pacer.clock.lock().unwrap() = link;
                put(
                    out_pacer,
                    mln_frame_pacer(handles::insert(Object::Pacer(pacer))),
                    "out_pacer",
                )
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = interval;
                Err(Failure::unsupported(
                    "Orbis's Apple frame pacer is macOS only so far",
                ))
            }
        })
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mln_frame_pacer_release(pacer: mln_frame_pacer) {
    // Sessions keep it until they go.
    handles::remove(pacer.0);
}
