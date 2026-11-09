//! Labels placed in a worker of their own: both ends of the
//! protocol between the map and the placement worker.
//!
//! The web package's JS owns the transport, a `MessagePort`, and passes
//! messages through unread. The map's [`WorkerPlacement`] posts a bucket's
//! tables once and a frame per placement; the [`PlacementWorker`] answers
//! each frame, in order, with its commit. Payloads are
//! [`orb_runtime::wire`]'s bytes, in transferred
//! buffers.
//!
//! Messages, map to placement worker: `{ingest, buffer}` (a
//! bucket id and its packed tables), `{remove}` (a bucket id),
//! `{ground: buffer}` (the terrain's elevation pyramid for the frames that
//! follow, empty for none; sent when the terrain changes) and
//! `{place: buffer}` (a packed frame). Placement worker to map:
//! `{committed: buffer}`, one per frame.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use js_sys::{Array, Object, Reflect, Uint8Array};
use orb_runtime::{InlinePlacement, PlaceDone, PlacementLane, TerrainElevation, wire};
use orb_symbol::{Frame, GroundRef, PlacementTables};
use wasm_bindgen::prelude::*;

thread_local! {
    /// Each map's placement worker, by map. Wasm in a page is
    /// single-threaded, so the runtime's `Send` lane reaches it here.
    static LANES: RefCell<HashMap<u32, Lane>> = RefCell::new(HashMap::new());
}

/// A map's `post(message, transfer)` to its placement worker, the commits
/// it owes, oldest first, and the terrain last sent.
struct Lane {
    post: js_sys::Function,
    owed: VecDeque<PlaceDone>,
    ground: Option<GroundRef>,
    /// The worker was replaced since the map last asked.
    restarted: bool,
}

/// Map `map`'s placement worker was lost and replaced, its `post` reaching
/// the new one: the placements it owed fail, and the map sends the new one
/// its buckets and terrain before it places again.
pub(crate) fn lost(map: u32, why: &str) {
    let owed = with_lane(map, |lane| {
        lane.restarted = true;
        lane.ground = None;
        std::mem::take(&mut lane.owed)
    });
    for done in owed.into_iter().flatten() {
        done(Err(format!("the placement worker was lost: {why}")));
    }
}

/// Forgets map `map`'s placement worker.
pub(crate) fn forget(map: u32) {
    LANES.with(|lanes| lanes.borrow_mut().remove(&map));
}

fn with_lane<T>(map: u32, f: impl FnOnce(&mut Lane) -> T) -> Option<T> {
    LANES.with(|lanes| lanes.borrow_mut().get_mut(&map).map(f))
}

fn object(fields: &[(&str, JsValue)]) -> Object {
    let object = Object::new();
    for (key, value) in fields {
        let _ = Reflect::set(&object, &(*key).into(), value);
    }
    object
}

/// A map's [`PlacementLane`]: its placement worker.
#[derive(Debug)]
pub(crate) struct WorkerPlacement(u32);

impl WorkerPlacement {
    /// Map `map`'s placement worker, reached through `post`.
    pub(crate) fn new(map: u32, post: js_sys::Function) -> Arc<Self> {
        let lane = Lane {
            post,
            owed: VecDeque::new(),
            ground: None,
            restarted: false,
        };
        LANES.with(|lanes| lanes.borrow_mut().insert(map, lane));
        Arc::new(Self(map))
    }

    fn post(&self, message: &Object, buffer: Option<&Uint8Array>) {
        let transfer = Array::new();
        if let Some(buffer) = buffer {
            transfer.push(&buffer.buffer());
        }
        // Called outside the borrow: the worker's reply may come back in.
        if let Some(post) = with_lane(self.0, |lane| lane.post.clone()) {
            let _ = post.call2(&JsValue::NULL, message, &transfer);
        }
    }

    /// Completes map `map`'s oldest placement with its worker's reply: a
    /// commit, or `{error}` when the worker failed to place.
    pub(crate) fn receive(map: u32, reply: &JsValue) -> Result<(), JsError> {
        let done = with_lane(map, |lane| lane.owed.pop_front())
            .flatten()
            .ok_or_else(|| JsError::new("a commit nobody asked for"))?;
        if let Some(error) = Reflect::get(reply, &"error".into())
            .ok()
            .and_then(|e| e.as_string())
        {
            done(Err(error));
            return Ok(());
        }
        let committed = Reflect::get(reply, &"committed".into())
            .ok()
            .and_then(|b| b.dyn_into::<Uint8Array>().ok())
            .ok_or_else(|| "placement reply without a commit".to_string())
            .and_then(|bytes| wire::decode(&bytes.to_vec()));
        let failed = committed.as_ref().err().cloned();
        done(committed);
        failed.map_or(Ok(()), |e| Err(JsError::new(&e)))
    }
}

impl PlacementLane for WorkerPlacement {
    fn ingest(&self, bucket: u64, tables: Arc<PlacementTables>) {
        let buffer = Uint8Array::from(wire::encode(&*tables).as_slice());
        self.post(
            &object(&[
                ("ingest", (bucket as f64).into()),
                ("buffer", buffer.clone().into()),
            ]),
            Some(&buffer),
        );
    }

    fn remove(&self, bucket: u64) {
        self.post(&object(&[("remove", (bucket as f64).into())]), None);
    }

    fn place(&self, frame: Frame, done: PlaceDone) {
        let changed = with_lane(self.0, |lane| {
            lane.owed.push_back(done);
            let changed = lane.ground != frame.ground;
            lane.ground.clone_from(&frame.ground);
            changed
        })
        .unwrap_or(false);
        if changed {
            let bytes = frame
                .ground
                .as_ref()
                .map_or_else(Default::default, |g| g.0.pyramid());
            let buffer = Uint8Array::from(&bytes[..]);
            self.post(&object(&[("ground", buffer.clone().into())]), Some(&buffer));
        }
        let buffer = Uint8Array::from(wire::encode(&frame).as_slice());
        self.post(&object(&[("place", buffer.clone().into())]), Some(&buffer));
    }

    fn restarted(&self) -> bool {
        with_lane(self.0, |lane| std::mem::take(&mut lane.restarted)).unwrap_or(false)
    }
}

/// The placement worker: `orb-symbol`'s placer behind a port.
#[wasm_bindgen]
#[derive(Debug, Default)]
pub struct PlacementWorker {
    lane: InlinePlacement,
    /// The terrain frames place on.
    ground: Option<GroundRef>,
}

#[wasm_bindgen]
impl PlacementWorker {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Traps, as a bug would: the page replaces the worker. To exercise
    /// that recovery.
    pub fn trap(&self) {
        panic!("a trap asked for");
    }

    /// Handles a message from the map; returns the reply to post
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
            let tables: PlacementTables =
                wire::decode(&bytes(get("buffer"))?).map_err(|e| JsError::new(&e))?;
            self.lane.ingest(bucket as u64, Arc::new(tables));
            return Ok(JsValue::UNDEFINED);
        }
        if let Some(bucket) = get("remove").as_f64() {
            self.lane.remove(bucket as u64);
            return Ok(JsValue::UNDEFINED);
        }
        let ground = get("ground");
        if !ground.is_undefined() {
            let pyramid = bytes(ground)?;
            self.ground = if pyramid.is_empty() {
                None
            } else {
                let terrain =
                    TerrainElevation::from_pyramid(&pyramid).map_err(|e| JsError::new(&e))?;
                Some(GroundRef(Arc::new(terrain)))
            };
            return Ok(JsValue::UNDEFINED);
        }
        let mut frame: Frame = wire::decode(&bytes(get("place"))?).map_err(|e| JsError::new(&e))?;
        frame.ground.clone_from(&self.ground);
        let reply = Arc::new(Mutex::new(None));
        let sink = Arc::clone(&reply);
        self.lane.place(
            frame,
            Box::new(move |committed| {
                if let Ok(committed) = committed {
                    *sink.lock().expect("reply poisoned") = Some(wire::encode(&committed));
                }
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
