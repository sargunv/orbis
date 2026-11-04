//! Labels placed in a worker of their own (decision 0009): both ends of the
//! protocol between the render worker and the placement worker.
//!
//! The page's JS owns the transport, a `MessagePort`, and passes messages
//! through unread. The render worker's [`WorkerPlacement`] posts a bucket's
//! tables once and a frame per placement; the [`PlacementWorker`] answers
//! each frame, in order, with its commit. Payloads are
//! [`orb_runtime::pack_tables`]' bytes and the like, in transferred
//! buffers.
//!
//! Messages, render worker to placement worker: `{ingest, buffer}` (a
//! bucket id and its packed tables), `{remove}` (a bucket id) and
//! `{place: buffer}` (a packed frame). Placement worker to render worker:
//! `{committed: buffer}`, one per frame.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use js_sys::{Array, Object, Reflect, Uint8Array};
use orb_runtime::{
    InlinePlacement, PlaceDone, PlacementLane, pack_committed, pack_frame, pack_tables,
    unpack_committed, unpack_frame, unpack_tables,
};
use orb_symbol::{Frame, PlacementTables};
use wasm_bindgen::prelude::*;

thread_local! {
    /// The render worker's `post(message, transfer)` to the placement
    /// worker, and the commits it owes, oldest first. Wasm in a worker is
    /// single-threaded, so the runtime's `Send` lane reaches them here.
    static POST: RefCell<Option<js_sys::Function>> = const { RefCell::new(None) };
    static OWED: RefCell<VecDeque<PlaceDone>> = const { RefCell::new(VecDeque::new()) };
}

fn object(fields: &[(&str, JsValue)]) -> Object {
    let object = Object::new();
    for (key, value) in fields {
        let _ = Reflect::set(&object, &(*key).into(), value);
    }
    object
}

fn post(message: &Object, buffer: Option<&Uint8Array>) {
    let transfer = Array::new();
    if let Some(buffer) = buffer {
        transfer.push(&buffer.buffer());
    }
    POST.with(|post| {
        if let Some(f) = post.borrow().as_ref() {
            let _ = f.call2(&JsValue::NULL, message, &transfer);
        }
    });
}

/// The render worker's [`PlacementLane`].
#[derive(Debug)]
pub(crate) struct WorkerPlacement;

impl WorkerPlacement {
    pub(crate) fn new(post: js_sys::Function) -> Arc<Self> {
        POST.with(|p| *p.borrow_mut() = Some(post));
        Arc::new(Self)
    }

    /// Completes the oldest placement with the worker's reply.
    pub(crate) fn receive(reply: &JsValue) -> Result<(), JsError> {
        let bytes = Reflect::get(reply, &"committed".into())
            .ok()
            .and_then(|b| b.dyn_into::<Uint8Array>().ok())
            .ok_or_else(|| JsError::new("placement reply without a commit"))?;
        let committed = unpack_committed(&bytes.to_vec()).map_err(|e| JsError::new(&e))?;
        let done = OWED
            .with(|owed| owed.borrow_mut().pop_front())
            .ok_or_else(|| JsError::new("a commit nobody asked for"))?;
        done(committed);
        Ok(())
    }
}

impl PlacementLane for WorkerPlacement {
    fn ingest(&self, bucket: u64, tables: Arc<PlacementTables>) {
        let buffer = Uint8Array::from(pack_tables(&tables).as_slice());
        post(
            &object(&[
                ("ingest", (bucket as f64).into()),
                ("buffer", buffer.clone().into()),
            ]),
            Some(&buffer),
        );
    }

    fn remove(&self, bucket: u64) {
        post(&object(&[("remove", (bucket as f64).into())]), None);
    }

    fn place(&self, frame: Frame, done: PlaceDone) {
        OWED.with(|owed| owed.borrow_mut().push_back(done));
        let buffer = Uint8Array::from(pack_frame(&frame).as_slice());
        post(&object(&[("place", buffer.clone().into())]), Some(&buffer));
    }
}

/// The placement worker: `orb-symbol`'s placer behind a port.
#[wasm_bindgen]
#[derive(Debug, Default)]
pub struct PlacementWorker {
    lane: InlinePlacement,
}

#[wasm_bindgen]
impl PlacementWorker {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Handles a message from the render worker; returns the reply to post
    /// back, transferring its `committed` buffer, or nothing.
    pub fn receive(&mut self, message: &JsValue) -> Result<JsValue, JsError> {
        let get = |key: &str| Reflect::get(message, &key.into()).unwrap_or(JsValue::UNDEFINED);
        let bytes = |value: JsValue| {
            value
                .dyn_into::<Uint8Array>()
                .map(|b| b.to_vec())
                .map_err(|_| JsError::new("placement message without bytes"))
        };
        if let Some(bucket) = get("ingest").as_f64() {
            let tables = unpack_tables(&bytes(get("buffer"))?).map_err(|e| JsError::new(&e))?;
            self.lane.ingest(bucket as u64, Arc::new(tables));
            return Ok(JsValue::UNDEFINED);
        }
        if let Some(bucket) = get("remove").as_f64() {
            self.lane.remove(bucket as u64);
            return Ok(JsValue::UNDEFINED);
        }
        let frame = unpack_frame(&bytes(get("place"))?).map_err(|e| JsError::new(&e))?;
        let reply = Arc::new(Mutex::new(None));
        let sink = Arc::clone(&reply);
        self.lane.place(
            frame,
            Box::new(move |committed| {
                *sink.lock().expect("reply poisoned") = Some(pack_committed(&committed));
            }),
        );
        let packed = reply
            .lock()
            .expect("reply poisoned")
            .take()
            .expect("inline placement commits before returning");
        Ok(object(&[("committed", Uint8Array::from(packed.as_slice()).into())]).into())
    }
}
