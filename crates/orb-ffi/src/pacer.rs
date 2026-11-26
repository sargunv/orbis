// A render session's frame pacing: a clock that asks the session for a
// frame at the display's refreshes, within a frame rate limit, as
// `mln_render_session_set_frame_pacing` sets it.
//
// On macOS the clock is CoreVideo's display link, on the display that
// shows the host's view, which it rechecks on the main thread a few times a
// second to follow the window to another display, or on whichever active
// display the link picks.
// A display link needs an active display, which a Mac whose displays sleep
// (or a headless one) lacks, and a host may still draw then (offscreen, or
// for a capture), so pacing then keeps time on a thread of its own at a
// nominal refresh.

#[cfg(target_os = "macos")]
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;
#[cfg(target_os = "macos")]
use std::time::Instant;

#[cfg(target_os = "macos")]
use crate::base::lock;
use crate::base::{Failure, sized};
#[cfg(target_os = "macos")]
use crate::session::SessionHandle;
use crate::sys::*;

/// What `mln_frame_pacing_options` sets, checked.
pub(crate) struct PacingOptions {
    /// The shortest time between two frames, from the frame rate limit.
    #[cfg_attr(
        not(target_os = "macos"),
        expect(dead_code, reason = "only macOS paces frames")
    )]
    interval: Option<Duration>,
    /// The host's `NSView*`, whose window's display the frames follow.
    #[cfg_attr(
        not(target_os = "macos"),
        expect(dead_code, reason = "only macOS paces frames")
    )]
    view: Option<*mut std::ffi::c_void>,
}

impl PacingOptions {
    /// Checks `options`, which says nothing of a display yet: their
    /// errors come before a platform's.
    ///
    /// # Safety
    ///
    /// `options` is null or points to a sized `mln_frame_pacing_options`.
    pub(crate) unsafe fn from_c(options: *const mln_frame_pacing_options) -> Result<Self, Failure> {
        // SAFETY: the caller's pointer.
        let o = unsafe { sized(options, "options") }?;
        let known =
            MLN_FRAME_PACING_OPTION_MAXIMUM_FRAMES_PER_SECOND | MLN_FRAME_PACING_OPTION_VIEW;
        if o.fields & !known != 0 {
            return Err(Failure::argument("options.fields sets an unknown field"));
        }
        let interval = match o.fields & MLN_FRAME_PACING_OPTION_MAXIMUM_FRAMES_PER_SECOND != 0 {
            true if o.maximum_frames_per_second == 0 => {
                return Err(Failure::argument(
                    "options.maximum_frames_per_second must be greater than zero",
                ));
            }
            true => Some(Duration::from_secs(1) / o.maximum_frames_per_second),
            false => None,
        };
        let view = match o.fields & MLN_FRAME_PACING_OPTION_VIEW != 0 {
            true if o.view.is_null() => {
                return Err(Failure::argument("options.view is null"));
            }
            true => Some(o.view),
            false => None,
        };
        Ok(Self { interval, view })
    }
}

/// Paces one session's frames until it goes: dropping it stops the clock.
#[cfg_attr(
    not(target_os = "macos"),
    expect(dead_code, reason = "only macOS paces frames")
)]
pub(crate) struct Pacing {
    /// The display link, which fields drop in order to stop, waiting out a
    /// refresh in flight, before the clock it reads goes.
    #[cfg(target_os = "macos")]
    _link: Option<display_link::Link>,
    #[cfg(target_os = "macos")]
    _clock: Arc<Clock>,
}

/// Paces `session` as `options` say, on the calling thread, which a view
/// needs to be the main one.
#[cfg(target_os = "macos")]
pub(crate) fn start(
    session: &Arc<SessionHandle>,
    options: &PacingOptions,
) -> Result<Pacing, Failure> {
    let view = match options.view {
        // SAFETY: the host's view, which `View::new` checks we're on the
        // thread of.
        Some(view) => Some(unsafe { display_link::View::new(view) }?),
        None => None,
    };
    // SAFETY: on the main thread, as `View::new` checked.
    let display = view.as_ref().and_then(|view| unsafe { view.display() });
    let clock = Arc::new(Clock {
        session: Arc::downgrade(session),
        interval: options.interval,
        last: Mutex::new(None),
        follow: OnceLock::new(),
    });
    let link = display_link::Link::start(Arc::as_ptr(&clock).cast(), display);
    match (&link, view) {
        (Some(link), Some(view)) => {
            let follow = display_link::Follow::new(view, link, display);
            let _ = clock.follow.set(follow);
        }
        // It ticks at the nominal refresh, and the limit gates the ticks as
        // it gates a display's.
        (None, _) => start_timer(Arc::downgrade(&clock))?,
        (Some(_), None) => {}
    }
    Ok(Pacing {
        _link: link,
        _clock: clock,
    })
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn start(
    _session: &std::sync::Arc<crate::session::SessionHandle>,
    options: &PacingOptions,
) -> Result<Pacing, Failure> {
    Err(Failure::unsupported(match options.view {
        Some(_) => "only Apple platforms pace frames to a view's display",
        None => "Orbis paces frames to macOS's displays alone so far",
    }))
}

/// The nominal refresh pacing keeps without a display.
#[cfg(target_os = "macos")]
const NOMINAL_REFRESH: Duration = Duration::from_micros(16_667);

/// What a refresh reads: the session it asks for frames, and the limit.
#[cfg(target_os = "macos")]
struct Clock {
    session: Weak<SessionHandle>,
    interval: Option<Duration>,
    /// When it last asked for a frame.
    last: Mutex<Option<Instant>>,
    /// The view whose window's display the link follows.
    follow: OnceLock<Arc<display_link::Follow>>,
}

#[cfg(target_os = "macos")]
impl Clock {
    /// A refresh, on the clock's thread: the session draws at the next one
    /// if it has something new.
    fn refresh(&self) {
        if let Some(follow) = self.follow.get() {
            follow.refresh();
        }
        if let Some(interval) = self.interval {
            let now = Instant::now();
            let mut last = lock(&self.last);
            // A refresh a little early still counts: refreshes jitter.
            if last.is_some_and(|l| now + interval / 8 < l + interval) {
                return;
            }
            *last = Some(now);
        }
        if let Some(session) = self.session.upgrade()
            && !session.tick.swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            session.wake();
        }
    }
}

#[cfg(target_os = "macos")]
mod display_link {
    //! CoreVideo's display link: a callback at each refresh of a display,
    //! on a thread of its own; and the display a view's window is on,
    //! which the link follows.
    use std::cell::UnsafeCell;
    use std::ffi::{c_char, c_void};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    use crate::base::Failure;

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
        fn CVDisplayLinkCreateWithCGDisplay(display: u32, out: *mut CVDisplayLinkRef) -> i32;
        fn CVDisplayLinkSetOutputCallback(
            link: CVDisplayLinkRef,
            callback: Callback,
            context: *mut c_void,
        ) -> i32;
        fn CVDisplayLinkStart(link: CVDisplayLinkRef) -> i32;
        fn CVDisplayLinkStop(link: CVDisplayLinkRef);
        fn CVDisplayLinkRetain(link: CVDisplayLinkRef) -> CVDisplayLinkRef;
        fn CVDisplayLinkRelease(link: CVDisplayLinkRef);
        fn CVDisplayLinkSetCurrentCGDisplay(link: CVDisplayLinkRef, display: u32) -> i32;
    }

    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_msgSend();
        fn objc_getClass(name: *const c_char) -> *mut c_void;
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_initWeak(location: *mut *mut c_void, object: *mut c_void) -> *mut c_void;
        fn objc_loadWeakRetained(location: *mut *mut c_void) -> *mut c_void;
        fn objc_destroyWeak(location: *mut *mut c_void);
        fn objc_release(object: *mut c_void);
    }

    unsafe extern "C" {
        fn pthread_main_np() -> i32;
        /// The main queue, `dispatch_get_main_queue()`.
        static _dispatch_main_q: u8;
        fn dispatch_async_f(
            queue: *const u8,
            context: *mut c_void,
            work: unsafe extern "C" fn(*mut c_void),
        );
    }

    /// Sends a message that takes and returns objects.
    ///
    /// # Safety
    ///
    /// `receiver` is an object or nil that answers `selector` with an
    /// object, taking `argument` when the selector has a colon.
    unsafe fn send(receiver: *mut c_void, selector: &[u8], argument: *mut c_void) -> *mut c_void {
        if receiver.is_null() {
            return receiver;
        }
        // SAFETY: objc_msgSend called through the method's own signature.
        unsafe {
            let msg: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void =
                std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            msg(
                receiver,
                sel_registerName(selector.as_ptr().cast()),
                argument,
            )
        }
    }

    /// A weak reference to the host's view, which may go before the
    /// pacing does.
    pub(super) struct View(Box<UnsafeCell<*mut c_void>>);
    // SAFETY: the runtime's weak references may be read and destroyed on
    // any thread; the view itself is touched on the main one alone.
    unsafe impl Send for View {}
    // SAFETY: as above.
    unsafe impl Sync for View {}

    impl View {
        /// # Safety
        ///
        /// `view` is an `NSView*`.
        pub(super) unsafe fn new(view: *mut c_void) -> Result<Self, Failure> {
            // SAFETY: a libc call.
            if unsafe { pthread_main_np() } == 0 {
                return Err(Failure::state(
                    "pacing frames to a view needs the main thread, as AppKit does",
                ));
            }
            let slot = Box::new(UnsafeCell::new(std::ptr::null_mut()));
            // SAFETY: a slot that stays put until `drop` destroys it.
            unsafe { objc_initWeak(slot.get(), view) };
            Ok(Self(slot))
        }

        /// The display that shows the view's window, or `None` when the
        /// view is gone, in no window, or its window on no screen.
        ///
        /// # Safety
        ///
        /// On the main thread.
        pub(super) unsafe fn display(&self) -> Option<u32> {
            // SAFETY: the weak slot `new` made; a view loaded is retained
            // until released here, on the main thread.
            unsafe {
                let view = objc_loadWeakRetained(self.0.get());
                if view.is_null() {
                    return None;
                }
                let display = view_display(view);
                objc_release(view);
                display
            }
        }
    }

    impl Drop for View {
        fn drop(&mut self) {
            // SAFETY: the slot `new` made.
            unsafe { objc_destroyWeak(self.0.get()) };
        }
    }

    /// Refreshes between two checks of the view's display.
    const RECHECK: u32 = 30;

    /// A link that follows a view's window from display to display.
    pub(super) struct Follow {
        view: View,
        /// The link, retained so that a check after it stops is harmless.
        link: CVDisplayLinkRef,
        display: AtomicU32,
        refreshes: AtomicU32,
        /// A check waits on the main queue.
        pending: AtomicBool,
    }
    // SAFETY: a CVDisplayLink is thread-safe, and `View` is.
    unsafe impl Send for Follow {}
    // SAFETY: as above.
    unsafe impl Sync for Follow {}

    impl Follow {
        pub(super) fn new(view: View, link: &Link, display: Option<u32>) -> Arc<Self> {
            Arc::new(Self {
                view,
                // SAFETY: a live link, released in `drop`.
                link: unsafe { CVDisplayLinkRetain(link.0) },
                display: AtomicU32::new(display.unwrap_or(0)),
                refreshes: AtomicU32::new(0),
                pending: AtomicBool::new(false),
            })
        }

        /// A refresh, on the link's thread: every so often, checks on the
        /// main thread which display shows the view's window, and moves the
        /// link there.
        pub(super) fn refresh(self: &Arc<Self>) {
            if !self
                .refreshes
                .fetch_add(1, Ordering::Relaxed)
                .is_multiple_of(RECHECK)
                || self.pending.swap(true, Ordering::AcqRel)
            {
                return;
            }
            unsafe extern "C" fn check(context: *mut c_void) {
                // SAFETY: the reference `refresh` gave up.
                let follow = unsafe { Arc::from_raw(context.cast::<Follow>()) };
                follow.pending.store(false, Ordering::Release);
                // SAFETY: the main queue runs on the main thread.
                let Some(display) = (unsafe { follow.view.display() }) else {
                    return;
                };
                if display != follow.display.load(Ordering::Relaxed)
                    // SAFETY: the retained link.
                    && unsafe { CVDisplayLinkSetCurrentCGDisplay(follow.link, display) } == 0
                {
                    follow.display.store(display, Ordering::Relaxed);
                }
            }
            let context = Arc::into_raw(Arc::clone(self));
            // SAFETY: libdispatch's main queue, and a context `check` takes.
            unsafe {
                dispatch_async_f(
                    &raw const _dispatch_main_q,
                    context.cast_mut().cast(),
                    check,
                );
            }
        }
    }

    impl Drop for Follow {
        fn drop(&mut self) {
            // SAFETY: the retain `new` took.
            unsafe { CVDisplayLinkRelease(self.link) };
        }
    }

    /// The display that shows `view`'s window, or `None` when it's in no
    /// window or the window on no screen, for the link to pick one.
    ///
    /// # Safety
    ///
    /// `view` is a live `NSView*`, on the main thread.
    unsafe fn view_display(view: *mut c_void) -> Option<u32> {
        // SAFETY: AppKit's API, on the main thread, as the view's docs ask:
        // [[[[view window] screen] deviceDescription]
        // objectForKey:@"NSScreenNumber"] unsignedIntValue].
        unsafe {
            let window = send(view, b"window\0", std::ptr::null_mut());
            let screen = send(window, b"screen\0", std::ptr::null_mut());
            let description = send(screen, b"deviceDescription\0", std::ptr::null_mut());
            let key = send(
                objc_getClass(c"NSString".as_ptr()),
                b"stringWithUTF8String:\0",
                c"NSScreenNumber".as_ptr().cast_mut().cast(),
            );
            let number = send(description, b"objectForKey:\0", key);
            if number.is_null() {
                return None;
            }
            let unsigned: unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32 =
                std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            Some(unsigned(
                number,
                sel_registerName(c"unsignedIntValue".as_ptr()),
            ))
        }
    }

    pub(super) struct Link(CVDisplayLinkRef);
    // SAFETY: a CVDisplayLink is thread-safe.
    unsafe impl Send for Link {}
    // SAFETY: a CVDisplayLink is thread-safe.
    unsafe impl Sync for Link {}

    impl Link {
        /// A link on `display`, or on the active displays, that refreshes
        /// the clock at `context`, which outlives it; `None` when no
        /// display is active.
        pub(super) fn start(context: *const c_void, display: Option<u32>) -> Option<Self> {
            unsafe extern "C" fn output(
                _: CVDisplayLinkRef,
                _: *const c_void,
                _: *const c_void,
                _: u64,
                _: *mut u64,
                context: *mut c_void,
            ) -> i32 {
                // SAFETY: `context` is the clock, which stops the link
                // before it goes.
                let clock = unsafe { &*context.cast::<super::Clock>() };
                clock.refresh();
                0
            }
            let mut link = std::ptr::null_mut();
            // SAFETY: CoreVideo's API, with an out pointer.
            unsafe {
                let made = match display {
                    Some(display) => CVDisplayLinkCreateWithCGDisplay(display, &mut link),
                    None => CVDisplayLinkCreateWithActiveCGDisplays(&mut link),
                };
                if made != 0 || link.is_null() {
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

/// Keeps time for a clock without a display, until it goes.
#[cfg(target_os = "macos")]
fn start_timer(clock: Weak<Clock>) -> Result<(), Failure> {
    std::thread::Builder::new()
        .name("orbis frame pacing".into())
        .spawn(move || {
            let mut next = Instant::now() + NOMINAL_REFRESH;
            loop {
                std::thread::sleep(next.saturating_duration_since(Instant::now()));
                next += NOMINAL_REFRESH;
                match clock.upgrade() {
                    Some(clock) => clock.refresh(),
                    None => return,
                }
            }
        })
        .map(drop)
        .map_err(|e| Failure::native(format!("the frame pacing timer: {e}")))
}
