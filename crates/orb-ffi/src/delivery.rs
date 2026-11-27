// Callback registrations and delivery queues.
//
// A registration's calls run on the library thread that makes them, or,
// with a queue, wait in it until the caller dispatches them; the queue's
// wake says when the empty queue gains a call. A library thread with an
// outbox (runtime and session threads) holds its calls until the work that
// raised them is done, so a callback can call the library back.
//
// A registration gives its user data back once: released after its last
// call, through its queue; abandoned to the queue's callback if the queue
// goes first; or left with the caller when the entry point that took it
// fails. A completion reports once whatever becomes of its work: dropped
// unrun, it reports `MLN_STATUS_CANCELLED`, or the panic that unwound it.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::base::{Failure, Result, UserData, boundary, lock, put, sized, to_host, view};
use crate::handles::{self, Object};
use crate::sys::*;

type Call = Box<dyn FnOnce() + Send + 'static>;

/// A call on its way, and what it keeps alive until it is in its queue or
/// has run: a release that comes of dropping it then follows the call.
struct Pending {
    run: Call,
    keep: Option<Box<dyn Send>>,
}

thread_local! {
    /// The calls a library thread delivers once its work is done.
    static OUTBOX: RefCell<Option<Vec<(u64, Pending)>>> = const { RefCell::new(None) };
    /// The entry points running on this thread, innermost last: more than
    /// one when a callback calls the library back.
    static CALLS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
    /// Completions dropped as a panic unwound this thread, for the
    /// [`catch`](crate::base::catch) that stops it to report with the
    /// panic's message.
    static UNWOUND: RefCell<Vec<Completion>> = const { RefCell::new(Vec::new()) };
    /// The [`catch`](crate::base::catch)es running on this thread: a
    /// completion a panic drops waits in [`UNWOUND`] only under one.
    static CATCHES: Cell<u32> = const { Cell::new(0) };
}

/// Numbers entry points, so a registration knows the call that took it.
static NEXT_CALL: AtomicU64 = AtomicU64::new(1);

/// Runs a [`catch`](crate::base::catch)'s `body`, which doesn't unwind.
pub(crate) fn catching<R>(body: impl FnOnce() -> R) -> R {
    CATCHES.with(|c| c.set(c.get() + 1));
    let out = body();
    CATCHES.with(|c| c.set(c.get() - 1));
    out
}

/// Runs an entry point's body: a registration it takes and drops before it
/// returns stays the caller's, as the call failed or never kept it.
pub(crate) fn in_call<R>(body: impl FnOnce() -> R) -> R {
    let id = NEXT_CALL.fetch_add(1, Ordering::Relaxed);
    CALLS.with(|c| c.borrow_mut().push(id));
    let out = body();
    CALLS.with(|c| c.borrow_mut().pop());
    out
}

/// The entry point running innermost on this thread, or 0.
fn current_call() -> u64 {
    CALLS
        .try_with(|c| c.borrow().last().copied())
        .ok()
        .flatten()
        .unwrap_or(0)
}

/// Reports the completions a panic dropped on this thread with its
/// `failure`.
pub(crate) fn unwound(failure: &Failure) {
    let dropped = UNWOUND
        .try_with(|u| std::mem::take(&mut *u.borrow_mut()))
        .unwrap_or_default();
    for completion in dropped {
        completion.complete(Err(failure.clone()), 0);
    }
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

/// Runs `call` or hands it to `queue`; a call for a queue that is gone
/// never runs.
fn send(queue: u64, Pending { run, keep }: Pending) {
    if queue == 0 {
        run();
    } else if let Some(Object::Queue(queue)) = handles::get(queue) {
        queue.push(run);
    }
    drop(keep);
}

/// Runs `call` on this thread (after its work, on a library thread) or
/// hands it to `queue`.
fn deliver(queue: u64, call: Pending) {
    // A call made as the thread ends, its outbox gone, goes now.
    let mut call = Some(call);
    let _ = OUTBOX.try_with(|o| {
        if let Ok(mut o) = o.try_borrow_mut()
            && let Some(outbox) = o.as_mut()
        {
            outbox.extend(call.take().map(|call| (queue, call)));
        }
    });
    if let Some(call) = call {
        send(queue, call);
    }
}

/// A registration's user data, and the release that gives it back.
struct Owned {
    user_data: UserData,
    release: mln_user_data_release,
    queue: u64,
    /// The entry point that took it.
    call: u64,
    /// Given back: released, abandoned, or left with the caller.
    done: AtomicBool,
}

impl Owned {
    /// Marks it given back, once: whether this was the time.
    fn settle(&self) -> bool {
        if self.done.swap(true, Ordering::AcqRel) {
            return false;
        }
        if self.queue != 0
            && let Some(Object::Queue(queue)) = handles::get(self.queue)
        {
            queue.untrack(self);
        }
        true
    }

    fn release(&self) {
        if self.settle() {
            self.give_back();
        }
    }

    /// Calls the release, for the caller that settled it.
    fn give_back(&self) {
        if let Some(release) = self.release {
            // SAFETY: the registration's release, called once.
            to_host(|| unsafe { release(self.user_data.get()) });
        }
    }

    /// Whether the entry point that took it is still running here.
    fn in_its_call(&self) -> bool {
        self.call != 0
            && CALLS
                .try_with(|c| c.borrow().contains(&self.call))
                .unwrap_or(false)
    }
}

/// A host's callback registration: its callback `F` (a function pointer,
/// or a few), its user data, and the queue its calls and release go
/// through. Dropped, it releases its user data after the calls already
/// on their way.
pub(crate) struct Registration<F: Copy + Send + Sync + 'static> {
    callback: F,
    queue: u64,
    owned: ManuallyDrop<Arc<Owned>>,
}

impl<F: Copy + Send + Sync + 'static> Registration<F> {
    /// `what`'s registration: none when its callback is null, which then
    /// takes no release or queue.
    pub(crate) fn optional(
        callback: Option<F>,
        user_data: *mut c_void,
        release: mln_user_data_release,
        queue: mln_delivery_queue,
        what: &str,
    ) -> Result<Option<Self>> {
        match callback {
            Some(callback) => Self::make(
                callback,
                user_data,
                release,
                queue.0,
                what,
                release.is_some(),
            )
            .map(Some),
            None if release.is_some() || queue.0 != 0 => Err(Failure::argument(format!(
                "{what} has no callback but a release or a queue"
            ))),
            None => Ok(None),
        }
    }

    /// `what`'s registration, whose callback must be set.
    pub(crate) fn required(
        callback: Option<F>,
        user_data: *mut c_void,
        release: mln_user_data_release,
        queue: mln_delivery_queue,
        what: &str,
    ) -> Result<Self> {
        if callback.is_none() {
            return Err(Failure::argument(format!("{what}.callback is null")));
        }
        Self::optional(callback, user_data, release, queue, what).map(|r| r.expect("a callback"))
    }

    /// A registration; one that `owns` its user data, which a release
    /// through its queue or its last call gives back, has it reported
    /// abandoned if the queue goes first.
    fn make(
        callback: F,
        user_data: *mut c_void,
        release: mln_user_data_release,
        queue: u64,
        what: &str,
        owns: bool,
    ) -> Result<Self> {
        let owned = Arc::new(Owned {
            user_data: UserData(user_data),
            release,
            queue,
            call: current_call(),
            done: AtomicBool::new(false),
        });
        if queue != 0 {
            let q = handles::queue(queue)
                .map_err(|f| Failure::new(f.status, format!("{what}.queue: {}", f.message)))?;
            if owns {
                q.track(&owned)?;
            }
        }
        Ok(Self {
            callback,
            queue,
            owned: ManuallyDrop::new(owned),
        })
    }

    pub(crate) fn callback(&self) -> F {
        self.callback
    }

    /// Calls the callback with its user data now, on this thread.
    pub(crate) fn call<R>(&self, f: impl FnOnce(F, *mut c_void) -> R) -> Option<R> {
        let user_data = self.owned.user_data;
        to_host(|| f(self.callback, user_data.get()))
    }

    /// Delivers a call to the callback through the registration's queue.
    pub(crate) fn deliver(self: &Arc<Self>, f: impl FnOnce(F, *mut c_void) + Send + 'static) {
        let (callback, user_data) = (self.callback, self.owned.user_data);
        deliver(
            self.queue,
            Pending {
                run: Box::new(move || {
                    to_host(|| f(callback, user_data.get()));
                }),
                keep: Some(Box::new(self.clone())),
            },
        );
    }

    /// Runs a call to the callback now, or hands it to its queue, even on a
    /// library thread: for calls the library makes as it goes (log records).
    pub(crate) fn deliver_now(&self, f: impl FnOnce(F, *mut c_void) + Send + 'static) {
        let (callback, user_data) = (self.callback, self.owned.user_data);
        let run: Call = Box::new(move || {
            to_host(|| f(callback, user_data.get()));
        });
        send(self.queue, Pending { run, keep: None });
    }

    /// Delivers the registration's last call, then its release: once the
    /// call runs, the release follows it whatever becomes of the queue.
    pub(crate) fn finish(self, f: impl FnOnce(F, *mut c_void) + Send + 'static) {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: taken once, and `this` is never dropped.
        let owned = unsafe { ManuallyDrop::take(&mut this.owned) };
        let callback = this.callback;
        let run: Call = Box::new(move || {
            // Unsettled, its user data went abandoned with its queue.
            if owned.settle() {
                to_host(|| f(callback, owned.user_data.get()));
                owned.give_back();
            }
        });
        deliver(this.queue, Pending { run, keep: None });
    }

    /// Whether the entry point that took it is still running here: one
    /// dropped now stays the caller's, as that call fails.
    pub(crate) fn in_its_call(&self) -> bool {
        self.owned.in_its_call()
    }
}

impl<F: Copy + Send + Sync + 'static> Drop for Registration<F> {
    fn drop(&mut self) {
        // SAFETY: taken once, here.
        let owned = unsafe { ManuallyDrop::take(&mut self.owned) };
        if owned.in_its_call() {
            owned.settle();
        } else if owned.release.is_some() {
            let run: Call = Box::new(move || owned.release());
            deliver(self.queue, Pending { run, keep: None });
        }
    }
}

/// An `mln_wake`: a call already waiting in its queue isn't queued again.
pub(crate) struct Wake {
    registration: Arc<Registration<unsafe extern "C" fn(*mut c_void)>>,
    queued: Arc<AtomicBool>,
}

impl Wake {
    /// A registration's wake: none when its callback is null.
    pub(crate) fn from_c(wake: &mln_wake, what: &str) -> Result<Option<Self>> {
        let registration = Registration::optional(
            wake.callback,
            wake.user_data,
            wake.release_user_data,
            wake.queue,
            what,
        )?;
        Ok(registration.map(|r| Self {
            registration: Arc::new(r),
            queued: Arc::default(),
        }))
    }

    pub(crate) fn signal(&self) {
        if self.registration.queue != 0 && self.queued.swap(true, Ordering::AcqRel) {
            return;
        }
        let queued = self.queued.clone();
        self.registration.deliver(move |callback, user_data| {
            queued.store(false, Ordering::Release);
            // SAFETY: the caller registered this callback and user_data.
            unsafe { callback(user_data) }
        });
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

type CompletionCallback = unsafe extern "C" fn(*mut c_void, *const mln_completion_result);

/// An `mln_completion`: reports once, then releases its user data. Dropped
/// unsent outside the entry point that took it, it reports
/// `MLN_STATUS_CANCELLED`, or the panic that dropped it.
pub(crate) struct Completion(Option<Registration<CompletionCallback>>);

impl Drop for Completion {
    fn drop(&mut self) {
        let Some(registration) = self.0.take() else {
            return;
        };
        // The entry point that took it is failing: it never reports.
        if registration.in_its_call() {
            return;
        }
        let panicking = std::thread::panicking();
        let mut registration = Some(registration);
        if panicking && CATCHES.try_with(Cell::get).unwrap_or(0) > 0 {
            let _ = UNWOUND.try_with(|u| {
                if let Ok(mut u) = u.try_borrow_mut() {
                    u.push(Self(registration.take()));
                }
            });
        }
        let failure = match panicking {
            true => Failure::native("the work panicked"),
            false => Failure::new(
                MLN_STATUS_CANCELLED,
                "the work was dropped before it finished",
            ),
        };
        if let Some(registration) = registration {
            send_result(registration, Err(failure), 0, 0);
        }
    }
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
        // Its callback gives its user data back, release or not.
        let registration = Registration::make(
            callback,
            c.user_data,
            c.release_user_data,
            c.queue.0,
            "completion",
            true,
        )?;
        Ok(Some(Self(Some(registration))))
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
        mut self,
        result: Result<Value>,
        generation: u64,
        render_update: u64,
    ) {
        if let Some(registration) = self.0.take() {
            send_result(registration, result, generation, render_update);
        }
    }
}

/// Delivers a completion's result, then its release.
fn send_result(
    registration: Registration<CompletionCallback>,
    result: Result<Value>,
    generation: u64,
    render_update: u64,
) {
    registration.finish(move |callback, user_data| {
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
        unsafe { callback(user_data, &result) };
    });
}

/// An `mln_delivery_queue`.
pub(crate) struct DeliveryQueue {
    calls: Mutex<VecDeque<Call>>,
    dispatching: Mutex<()>,
    wake: Mutex<Option<Arc<Registration<unsafe extern "C" fn(*mut c_void)>>>>,
    abandon: mln_user_data_release,
    /// The user data of the registrations whose release goes through the
    /// queue and hasn't run; `None` once the queue is released.
    owned: Mutex<Option<Vec<Arc<Owned>>>>,
}

impl DeliveryQueue {
    fn push(&self, call: Call) {
        let was_empty = {
            let mut calls = lock(&self.calls);
            calls.push_back(call);
            calls.len() == 1
        };
        // The wake may release the queue: it runs without the queue's locks.
        let wake = if was_empty {
            lock(&self.wake).clone()
        } else {
            None
        };
        if let Some(wake) = wake {
            // SAFETY: the caller registered this callback and user_data.
            wake.call(|callback, user_data| unsafe { callback(user_data) });
        }
    }

    fn track(&self, owned: &Arc<Owned>) -> Result {
        match &mut *lock(&self.owned) {
            Some(all) => {
                all.push(owned.clone());
                Ok(())
            }
            None => Err(Failure::state("the delivery queue was released")),
        }
    }

    fn untrack(&self, owned: &Owned) {
        if let Some(all) = &mut *lock(&self.owned) {
            all.retain(|o| !std::ptr::eq(&**o, owned));
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
            crate::base::watch_exit();
            let (wake, abandon) = if options.is_null() {
                (None, None)
            } else {
                let options = sized(options, "options")?;
                let w = &options.wake;
                if w.queue.0 != 0 {
                    return Err(Failure::argument(
                        "a delivery queue's wake can't itself go through a queue",
                    ));
                }
                let wake = Registration::optional(
                    w.callback,
                    w.user_data,
                    w.release_user_data,
                    w.queue,
                    "options.wake",
                )?;
                (wake, options.release_abandoned_user_data)
            };
            crate::base::fresh_out(out_queue, "out_queue")?;
            let queue = DeliveryQueue {
                calls: Mutex::default(),
                dispatching: Mutex::default(),
                wake: Mutex::new(wake.map(Arc::new)),
                abandon,
                owned: Mutex::new(Some(Vec::new())),
            };
            let handle = handles::insert(Object::Queue(Arc::new(queue)));
            put(out_queue, mln_delivery_queue(handle), "out_queue")
        })
    }
}

/// Releases the queue: the calls waiting in it never run, and the user
/// data of every registration whose release would go through it is
/// reported abandoned.
#[unsafe(no_mangle)]
pub extern "C" fn mln_delivery_queue_release(queue: mln_delivery_queue) {
    if let Some(Object::Queue(queue)) = handles::remove(queue.0) {
        drop(std::mem::take(&mut *lock(&queue.calls)));
        let owned = lock(&queue.owned).take().unwrap_or_default();
        for owned in owned {
            if !owned.done.swap(true, Ordering::AcqRel)
                && let Some(abandon) = queue.abandon
            {
                // SAFETY: the queue's registration, for user data a release
                // through it would have given back.
                to_host(|| unsafe { abandon(owned.user_data.get()) });
            }
        }
        drop(lock(&queue.wake).take());
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
                let call = lock(&queue.calls).pop_front();
                let Some(call) = call else {
                    return Ok(());
                };
                call();
            }
        })
    }
}

/// A completion that reports to a channel, for tests.
#[cfg(test)]
pub(crate) mod testing {
    use std::sync::mpsc::{Receiver, Sender, channel};

    use crate::sys::*;

    /// What a test completion saw: a result, then its user data's release.
    #[derive(Debug, PartialEq)]
    pub(crate) enum Report {
        Result(mln_status, String),
        Released,
    }

    unsafe extern "C" fn callback(
        user_data: *mut std::ffi::c_void,
        result: *const mln_completion_result,
    ) {
        // SAFETY: `user_data` is the sender `completion` made, and the
        // result lives for the call.
        let (tx, result) = unsafe { (&*user_data.cast::<Sender<Report>>(), &*result) };
        // SAFETY: the diagnostic the library lends for the call.
        let message = unsafe { crate::base::bytes(result.diagnostic) };
        let _ = tx.send(Report::Result(
            result.status,
            String::from_utf8_lossy(message).into(),
        ));
    }

    unsafe extern "C" fn release(user_data: *mut std::ffi::c_void) {
        // SAFETY: the sender `completion` made, released once.
        let tx = unsafe { Box::from_raw(user_data.cast::<Sender<Report>>()) };
        let _ = tx.send(Report::Released);
    }

    /// An `mln_completion` and the reports it sends. Its user data leaks
    /// unless it's released.
    pub(crate) fn completion() -> (mln_completion, Receiver<Report>) {
        let (tx, rx) = channel();
        let completion = mln_completion {
            size: std::mem::size_of::<mln_completion>() as u32,
            callback: Some(callback),
            user_data: Box::into_raw(Box::new(tx)).cast(),
            release_user_data: Some(release),
            queue: mln_delivery_queue(0),
        };
        (completion, rx)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use super::testing::{Report, completion};
    use super::*;

    #[test]
    fn a_dropped_completion_reports_cancelled_and_releases_once() {
        let (c, reports) = completion();
        // SAFETY: a valid completion.
        drop(unsafe { Completion::required(&c) }.unwrap());
        assert_eq!(
            reports.try_iter().collect::<Vec<_>>(),
            [
                Report::Result(
                    MLN_STATUS_CANCELLED,
                    "the work was dropped before it finished".into()
                ),
                Report::Released,
            ]
        );
    }

    #[test]
    fn a_completed_completion_reports_once() {
        let (c, reports) = completion();
        // SAFETY: a valid completion.
        unsafe { Completion::required(&c) }
            .unwrap()
            .complete(Ok(Value::None), 1);
        assert_eq!(
            reports.try_iter().collect::<Vec<_>>(),
            [
                Report::Result(MLN_STATUS_OK, String::new()),
                Report::Released
            ]
        );
    }

    #[test]
    fn a_failed_call_never_runs_its_completion() {
        let (c, reports) = completion();
        // SAFETY: a valid completion; no diagnostic.
        let status = unsafe {
            boundary(std::ptr::null_mut(), || {
                let _completion = Completion::required(&c)?;
                Err(Failure::argument("refused"))
            })
        };
        assert_eq!(status, MLN_STATUS_INVALID_ARGUMENT);
        assert!(reports.try_recv().is_err());
        // SAFETY: the user data the call left with the caller.
        unsafe { (c.release_user_data.unwrap())(c.user_data) };
    }

    #[test]
    fn a_completion_a_panic_drops_reports_the_panic() {
        let (c, reports) = completion();
        // SAFETY: a valid completion.
        let completion = unsafe { Completion::required(&c) }.unwrap();
        let caught = crate::base::catch(move || -> Result {
            let _completion = completion;
            panic!("the work broke");
        });
        assert!(caught.is_err());
        assert_eq!(
            reports.try_iter().collect::<Vec<_>>(),
            [
                Report::Result(MLN_STATUS_NATIVE_ERROR, "the work broke".into()),
                Report::Released,
            ]
        );
    }

    static ORDER: Mutex<Vec<&str>> = Mutex::new(Vec::new());

    unsafe extern "C" fn release(_: *mut std::ffi::c_void) {
        ORDER.lock().unwrap().push("release");
    }

    #[test]
    fn a_registration_is_released_after_the_calls_other_threads_hold() {
        unsafe extern "C" fn callback(_: *mut std::ffi::c_void) {}
        let registration = Arc::new(
            Registration::required(
                Some(callback as unsafe extern "C" fn(*mut std::ffi::c_void)),
                std::ptr::null_mut(),
                Some(release),
                mln_delivery_queue(0),
                "handler",
            )
            .unwrap(),
        );
        let (queued, wait) = (mpsc::channel(), mpsc::channel::<()>());
        let library = std::thread::spawn({
            let registration = registration.clone();
            move || {
                start_outbox();
                registration.deliver(|_, _| ORDER.lock().unwrap().push("call"));
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
