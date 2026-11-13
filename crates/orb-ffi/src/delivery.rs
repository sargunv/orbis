// Callback registrations (wakes, completions) and delivery queues.
//
// A registration without a queue runs on the library thread that produces the
// call; with one, the call waits in the queue until the caller dispatches it,
// and the queue's wake says when the empty queue gains a call. On a runtime
// thread, calls wait in its outbox until the work that raised them is done,
// so a callback can call the library back. Runtime and session threads
// both have one.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::base::{Failure, Result, UserData, boundary, put, sized, view};
use crate::handles::{self, Object};
use crate::sys::*;

pub(crate) type Call = Box<dyn FnOnce() + Send + 'static>;

/// A call on its way, and the user data its registration releases after
/// it: a queue released before the call runs reports that user data as
/// abandoned instead.
pub(crate) struct Pending {
    run: Call,
    abandons: Option<UserData>,
}

thread_local! {
    /// The calls a library thread delivers once its work is done.
    static OUTBOX: RefCell<Option<Vec<(u64, Pending)>>> = const { RefCell::new(None) };
}

/// Makes this thread a library thread, whose deliveries wait for [`flush`].
pub(crate) fn start_outbox() {
    OUTBOX.with(|o| *o.borrow_mut() = Some(Vec::new()));
}

/// Hands the calls in this library thread's outbox to their queues, or
/// runs them.
pub(crate) fn flush() {
    loop {
        let calls = OUTBOX.with(|o| {
            o.borrow_mut()
                .as_mut()
                .map(std::mem::take)
                .unwrap_or_default()
        });
        if calls.is_empty() {
            return;
        }
        for (queue, call) in calls {
            send(queue, call);
        }
    }
}

fn send(queue: u64, call: Pending) {
    if queue == 0 {
        (call.run)();
        return;
    }
    // A call for a queue that went never runs.
    if let Some(Object::Queue(queue)) = handles::get(queue) {
        queue.push(call);
    }
}

/// Runs `call` on this thread (after the runtime's work) or hands it to
/// `queue`. A call for a queue that is gone never runs.
pub(crate) fn deliver(queue: u64, call: Call) {
    deliver_pending(
        queue,
        Pending {
            run: call,
            abandons: None,
        },
    );
}

fn deliver_pending(queue: u64, call: Pending) {
    let call = OUTBOX.with(|o| match o.borrow_mut().as_mut() {
        Some(outbox) => {
            outbox.push((queue, call));
            None
        }
        None => Some(call),
    });
    if let Some(call) = call {
        send(queue, call);
    }
}

/// Runs `call` on this thread now, or hands it to `queue`, even on a
/// runtime thread: for calls the library makes as it goes (log records).
pub(crate) fn deliver_now(queue: u64, call: Call) {
    send(
        queue,
        Pending {
            run: call,
            abandons: None,
        },
    );
}

/// The user data of a registration that more than one library thread
/// calls (an event handler, a custom source): each call on its way holds
/// it, and its release goes to its queue once the registration and every
/// such call are gone. A release sent when the registration went could
/// otherwise reach the host before a call that another thread still held
/// in its outbox.
pub(crate) struct Registration {
    user_data: UserData,
    release: mln_user_data_release,
    queue: u64,
}

impl Registration {
    pub(crate) fn new(
        user_data: UserData,
        release: mln_user_data_release,
        queue: u64,
    ) -> Arc<Self> {
        Arc::new(Self {
            user_data,
            release,
            queue,
        })
    }

    pub(crate) fn user_data(&self) -> *mut std::ffi::c_void {
        self.user_data.get()
    }

    /// Delivers `call` on the registration's queue; the registration lives
    /// until the call has run or will never run.
    pub(crate) fn deliver(self: &Arc<Self>, call: impl FnOnce(&Self) + Send + 'static) {
        let this = self.clone();
        deliver(self.queue, Box::new(move || call(&this)));
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        release_later(self.queue, self.release, self.user_data);
    }
}

/// Calls a registration's `release` with its user data once, on its queue.
pub(crate) fn release_later(queue: u64, release: mln_user_data_release, user_data: UserData) {
    if let Some(release) = release {
        deliver_pending(
            queue,
            Pending {
                run: Box::new(move || {
                    // SAFETY: the registration's release, called once.
                    unsafe { release(user_data.get()) }
                }),
                abandons: Some(user_data),
            },
        );
    }
}

/// Checks that `queue`, when set, names a live delivery queue.
pub(crate) fn check_queue(queue: u64, what: &str) -> Result {
    if queue == 0 {
        return Ok(());
    }
    handles::queue(queue)
        .map(|_| ())
        .map_err(|f| Failure::new(f.status, format!("{what}: {}", f.message)))
}

/// An `mln_wake`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Wake {
    callback: unsafe extern "C" fn(*mut std::ffi::c_void),
    user_data: UserData,
    release_user_data: mln_user_data_release,
    queue: u64,
}

impl Wake {
    /// A registration's wake: none when its callback is null, which then
    /// takes no release either.
    pub(crate) fn from_c(wake: &mln_wake, what: &str) -> Result<Option<Self>> {
        let Some(callback) = wake.callback else {
            if wake.release_user_data.is_some() || wake.queue.0 != 0 {
                return Err(Failure::argument(format!(
                    "{what} has no callback but a release or a queue"
                )));
            }
            return Ok(None);
        };
        check_queue(wake.queue.0, &format!("{what}.queue"))?;
        Ok(Some(Self {
            callback,
            user_data: UserData(wake.user_data),
            release_user_data: wake.release_user_data,
            queue: wake.queue.0,
        }))
    }

    pub(crate) fn signal(&self) {
        let wake = *self;
        deliver(
            self.queue,
            Box::new(move || {
                // SAFETY: the caller registered this callback and user_data.
                unsafe { (wake.callback)(wake.user_data.get()) }
            }),
        );
    }

    /// Calls the wake now, on this thread, whatever its queue.
    fn signal_here(&self) {
        // SAFETY: the caller registered this callback and user_data.
        unsafe { (self.callback)(self.user_data.get()) }
    }

    pub(crate) fn release(self) {
        release_later(self.queue, self.release_user_data, self.user_data);
    }
}

/// What a completion reports.
pub(crate) enum Value {
    None,
    /// An `mln_buffer_view` of this text, terminated.
    Text(String),
    /// An `mln_texture_readback_result` of these pixels.
    Readback {
        pixels: Vec<u8>,
        info: mln_texture_image_info,
    },
    /// Plain C structs, as their bytes in 8-byte words so doubles align,
    /// each `size` bytes, `count` of them; `keep` holds what they point at.
    Structs {
        words: Vec<u64>,
        size: usize,
        count: usize,
        _keep: Option<Box<dyn std::any::Any + Send>>,
    },
}

impl Value {
    pub(crate) fn of<T: Copy>(value: T) -> Self {
        Self::array(&[value], None)
    }

    /// An array of C structs, with what they point at.
    pub(crate) fn array<T: Copy>(
        values: &[T],
        keep: Option<Box<dyn std::any::Any + Send>>,
    ) -> Self {
        let size = std::mem::size_of::<T>();
        let mut words = vec![0u64; std::mem::size_of_val(values).div_ceil(8).max(1)];
        // SAFETY: `words` holds `size × len` bytes, 8-aligned; T is plain C
        // data.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                words.as_mut_ptr().cast::<T>(),
                values.len(),
            );
        }
        Value::Structs {
            words,
            size,
            count: values.len(),
            _keep: keep,
        }
    }
}

/// An `mln_completion`: runs once, then releases its user data.
pub(crate) struct Completion {
    callback: unsafe extern "C" fn(*mut std::ffi::c_void, *const mln_completion_result),
    user_data: UserData,
    release_user_data: mln_user_data_release,
    queue: u64,
}

impl Completion {
    /// A command's completion, which may be null.
    ///
    /// # Safety
    /// `completion` is null or a readable `mln_completion`.
    pub(crate) unsafe fn from_c(completion: *const mln_completion) -> Result<Option<Self>> {
        if completion.is_null() {
            return Ok(None);
        }
        // SAFETY: the caller's contract.
        let c = unsafe { sized(completion, "completion")? };
        let callback = c
            .callback
            .ok_or_else(|| Failure::argument("completion.callback is null"))?;
        check_queue(c.queue.0, "completion.queue")?;
        Ok(Some(Self {
            callback,
            user_data: UserData(c.user_data),
            release_user_data: c.release_user_data,
            queue: c.queue.0,
        }))
    }

    /// The completion of a query, an operation or a lifecycle call, which
    /// needs one.
    ///
    /// # Safety
    /// As [`from_c`](Self::from_c).
    pub(crate) unsafe fn required(completion: *const mln_completion) -> Result<Self> {
        // SAFETY: the caller's contract.
        unsafe { Self::from_c(completion) }?
            .ok_or_else(|| Failure::argument("the call needs a completion"))
    }

    /// Delivers the result: `generation` is the map generation that shows it.
    pub(crate) fn complete(self, result: Result<Value>, generation: u64) {
        self.complete_update(result, generation, generation);
    }

    /// Delivers the result of a map command whose change the render update
    /// `render_update` first shows.
    pub(crate) fn complete_update(
        self,
        result: Result<Value>,
        generation: u64,
        render_update: u64,
    ) {
        let Completion {
            callback,
            user_data,
            release_user_data,
            queue,
        } = self;
        deliver_pending(
            queue,
            Pending {
                run: Box::new(move || {
                    let (status, mut message, mut value) = match result {
                        Ok(value) => (MLN_STATUS_OK, String::new(), value),
                        Err(failure) => (failure.status, failure.message, Value::None),
                    };
                    let bytes_view;
                    let readback;
                    let diagnostic = crate::base::lend(&mut message);
                    let (value, value_size, value_count): (*const std::ffi::c_void, usize, usize) =
                        match &mut value {
                            Value::None => (std::ptr::null(), 0, 0),
                            Value::Text(text) => {
                                bytes_view = crate::base::lend(text);
                                (
                                    std::ptr::from_ref(&bytes_view).cast(),
                                    std::mem::size_of::<mln_buffer_view>(),
                                    1,
                                )
                            }
                            Value::Readback { pixels, info } => {
                                readback = mln_texture_readback_result {
                                    data: view(pixels),
                                    info: *info,
                                };
                                (
                                    std::ptr::from_ref(&readback).cast(),
                                    std::mem::size_of::<mln_texture_readback_result>(),
                                    1,
                                )
                            }
                            Value::Structs {
                                words, size, count, ..
                            } => {
                                if *count == 0 {
                                    (std::ptr::null(), *size, 0)
                                } else {
                                    (words.as_ptr().cast(), *size, *count)
                                }
                            }
                        };
                    let result = mln_completion_result {
                        size: std::mem::size_of::<mln_completion_result>() as u32,
                        status,
                        value_size: value_size as u32,
                        generation,
                        render_update_generation: render_update,
                        diagnostic,
                        value,
                        value_count,
                    };
                    // SAFETY: the caller registered this callback and
                    // user_data; `result` and what it points at live until
                    // it returns.
                    unsafe { callback(user_data.get(), &result) };
                    if let Some(release) = release_user_data {
                        // SAFETY: the registration's release, called once.
                        unsafe { release(user_data.get()) };
                    }
                }),
                abandons: release_user_data.map(|_| user_data),
            },
        );
    }
}

/// An `mln_delivery_queue`.
pub(crate) struct DeliveryQueue {
    calls: Mutex<VecDeque<Pending>>,
    dispatching: Mutex<()>,
    wake: Option<Wake>,
    abandon: mln_user_data_release,
}

impl DeliveryQueue {
    fn push(&self, call: Pending) {
        let was_empty = {
            let mut calls = self
                .calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            calls.push_back(call);
            calls.len() == 1
        };
        if was_empty && let Some(wake) = &self.wake {
            wake.signal_here();
        }
    }

    /// Reports the user data of the calls that will never run.
    fn abandon(&self, calls: VecDeque<Pending>) {
        for call in calls {
            if let (Some(abandon), Some(user_data)) = (self.abandon, call.abandons) {
                // SAFETY: the queue's registration, for user data its call
                // would have released.
                unsafe { abandon(user_data.get()) };
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_delivery_queue_create(
    options: *const mln_delivery_queue_options,
    out_queue: *mut mln_delivery_queue,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let (wake, abandon) = if options.is_null() {
                (None, None)
            } else {
                let options = sized(options, "options")?;
                let wake = Wake::from_c(&options.wake, "options.wake")?;
                if wake.is_some_and(|w| w.queue != 0) {
                    return Err(Failure::argument(
                        "a delivery queue's wake can't itself go through a queue",
                    ));
                }
                (wake, options.release_abandoned_user_data)
            };
            crate::base::fresh_out(out_queue, "out_queue")?;
            let queue = DeliveryQueue {
                calls: Mutex::default(),
                dispatching: Mutex::default(),
                wake,
                abandon,
            };
            let handle = handles::insert(Object::Queue(Arc::new(queue)));
            put(out_queue, mln_delivery_queue(handle), "out_queue")
        })
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mln_delivery_queue_release(queue: mln_delivery_queue) {
    if let Some(Object::Queue(queue)) = handles::remove(queue.0) {
        let calls = std::mem::take(
            &mut *queue
                .calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        queue.abandon(calls);
        if let Some(wake) = queue.wake {
            wake.release();
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_delivery_queue_dispatch(
    queue: mln_delivery_queue,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointer.
    unsafe {
        boundary(out_diagnostic, || {
            let id = queue.0;
            let queue = handles::queue(id)?;
            let Ok(_dispatching) = queue.dispatching.try_lock() else {
                return Err(Failure::new(
                    MLN_STATUS_BUSY,
                    "another dispatch of the queue is running",
                ));
            };
            loop {
                // A call that released the queue abandons the rest.
                if handles::get(id).is_none() {
                    return Ok(());
                }
                let call = queue
                    .calls
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .pop_front();
                let Some(call) = call else {
                    return Ok(());
                };
                (call.run)();
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use super::*;

    static ORDER: Mutex<Vec<&str>> = Mutex::new(Vec::new());

    unsafe extern "C" fn release(_: *mut std::ffi::c_void) {
        ORDER.lock().unwrap().push("release");
    }

    #[test]
    fn a_registration_is_released_after_the_calls_other_threads_hold() {
        let registration = Registration::new(UserData(std::ptr::null_mut()), Some(release), 0);
        let (queued, wait) = (mpsc::channel(), mpsc::channel::<()>());
        let library = std::thread::spawn({
            let registration = registration.clone();
            move || {
                start_outbox();
                registration.deliver(|_| ORDER.lock().unwrap().push("call"));
                drop(registration);
                queued.0.send(()).unwrap();
                wait.1.recv().unwrap();
                flush();
            }
        });
        queued.1.recv().unwrap();
        // The registration goes while its call waits in the other thread's
        // outbox.
        drop(registration);
        assert!(ORDER.lock().unwrap().is_empty());
        wait.0.send(()).unwrap();
        library.join().unwrap();
        assert_eq!(*ORDER.lock().unwrap(), ["call", "release"]);
    }
}
