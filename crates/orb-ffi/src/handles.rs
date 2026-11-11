// The handles the C API hands out: a 64-bit id with the kind in its top
// byte, a slot index and the slot's generation, so a stale, foreign or
// made-up handle is an error, never a dangling pointer, and says which.

use std::sync::{Arc, Mutex};

use crate::base::{Failure, Result};
use crate::delivery::DeliveryQueue;
use crate::expression::ExpressionHandle;
use crate::frames::FrameHandle;
use crate::map::MapHandle;
use crate::pacer::FramePacer;
use crate::projection::ProjectionHandle;
use crate::resources::{ResourceRequest, TransformRequest};
use crate::runtime::RuntimeHandle;
use crate::session::SessionHandle;
use crate::sys::mln_render_frame_result;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(crate) enum Kind {
    Runtime = 1,
    Map = 2,
    Session = 4,
    Projection = 8,
    ResourceRequest = 10,
    TransformRequest = 11,
    Expression = 12,
    Frame = 13,
    Batch = 21,
    Queue = 23,
    OffscreenDevice = 26,
    Pacer = 27,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Runtime => "mln_runtime",
            Kind::Map => "mln_map",
            Kind::Session => "mln_render_session",
            Kind::Projection => "mln_map_projection",
            Kind::ResourceRequest => "mln_resource_request",
            Kind::TransformRequest => "mln_resource_transform_request",
            Kind::Expression => "mln_style_expression",
            Kind::Frame => "mln_acquired_frame",
            Kind::Batch => "mln_render_frame_batch",
            Kind::Queue => "mln_delivery_queue",
            Kind::OffscreenDevice => "mln_offscreen_device",
            Kind::Pacer => "mln_frame_pacer",
        }
    }
}

#[derive(Clone)]
pub(crate) enum Object {
    Runtime(Arc<RuntimeHandle>),
    Map(Arc<MapHandle>),
    Session(Arc<SessionHandle>),
    Projection(Arc<ProjectionHandle>),
    ResourceRequest(Arc<ResourceRequest>),
    TransformRequest(Arc<TransformRequest>),
    Expression(Arc<ExpressionHandle>),
    Frame(Arc<FrameHandle>),
    /// Holds the results a drained batch's view points at.
    Batch(#[allow(dead_code)] Arc<Vec<mln_render_frame_result>>),
    Queue(Arc<DeliveryQueue>),
    /// Orbis owns the device of its runtime, so an offscreen device is only a
    /// token that sessions accept.
    OffscreenDevice,
    #[cfg_attr(
        not(target_os = "macos"),
        expect(dead_code, reason = "only macOS makes a pacer")
    )]
    Pacer(Arc<FramePacer>),
}

impl Object {
    fn kind(&self) -> Kind {
        match self {
            Object::Runtime(_) => Kind::Runtime,
            Object::Map(_) => Kind::Map,
            Object::Session(_) => Kind::Session,
            Object::Projection(_) => Kind::Projection,
            Object::ResourceRequest(_) => Kind::ResourceRequest,
            Object::TransformRequest(_) => Kind::TransformRequest,
            Object::Expression(_) => Kind::Expression,
            Object::Frame(_) => Kind::Frame,
            Object::Batch(_) => Kind::Batch,
            Object::Queue(_) => Kind::Queue,
            Object::OffscreenDevice => Kind::OffscreenDevice,
            Object::Pacer(_) => Kind::Pacer,
        }
    }
}

/// Every handle's slot: the generation it last issued, and its object
/// while it lives.
struct Table {
    slots: Vec<(u32, Option<Object>)>,
    free: Vec<u32>,
}

static TABLE: Mutex<Table> = Mutex::new(Table {
    slots: Vec::new(),
    free: Vec::new(),
});

fn table() -> std::sync::MutexGuard<'static, Table> {
    TABLE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A handle's parts: kind byte, slot index, generation.
fn split(handle: u64) -> (u8, u32, u32) {
    (
        (handle >> 56) as u8,
        ((handle >> 32) & 0xFF_FFFF) as u32,
        handle as u32,
    )
}

/// Issues a handle for `object`: a freed slot again with its next
/// generation, or a new one.
pub(crate) fn insert(object: Object) -> u64 {
    let kind = object.kind() as u64;
    let mut table = table();
    let index = match table.free.pop() {
        Some(index) => index,
        None => {
            table.slots.push((0, None));
            (table.slots.len() - 1) as u32
        }
    };
    let slot = &mut table.slots[index as usize];
    slot.0 += 1;
    slot.1 = Some(object);
    (kind << 56) | (u64::from(index) << 32) | u64::from(slot.0)
}

/// The object a handle names, if it lives.
pub(crate) fn get(handle: u64) -> Option<Object> {
    let (kind, index, generation) = split(handle);
    let table = table();
    match table.slots.get(index as usize) {
        Some((issued, Some(object))) if *issued == generation && object.kind() as u8 == kind => {
            Some(object.clone())
        }
        _ => None,
    }
}

/// Retires a handle, freeing its slot for the next one.
pub(crate) fn remove(handle: u64) -> Option<Object> {
    let (kind, index, generation) = split(handle);
    let mut table = table();
    let slot = table.slots.get_mut(index as usize)?;
    if slot.0 != generation || slot.1.as_ref().is_none_or(|o| o.kind() as u8 != kind) {
        return None;
    }
    let object = slot.1.take();
    table.free.push(index);
    object
}

/// The kind a handle's top byte names.
fn kind_of(byte: u8) -> Option<Kind> {
    [
        Kind::Runtime,
        Kind::Map,
        Kind::Session,
        Kind::Projection,
        Kind::ResourceRequest,
        Kind::TransformRequest,
        Kind::Expression,
        Kind::Frame,
        Kind::Batch,
        Kind::Queue,
        Kind::OffscreenDevice,
        Kind::Pacer,
    ]
    .into_iter()
    .find(|k| *k as u8 == byte)
}

pub(crate) fn lookup(handle: u64, kind: Kind) -> Result<Object> {
    let name = kind.name();
    if handle == 0 {
        return Err(Failure::argument(format!("the {name} handle is null")));
    }
    let (byte, index, generation) = split(handle);
    let Some(actual) = kind_of(byte) else {
        return Err(Failure::argument(format!(
            "the {name} handle is malformed: no handle kind has its top byte"
        )));
    };
    if actual != kind {
        return Err(Failure::argument(format!(
            "an {} handle was passed where an {name} handle is expected",
            actual.name()
        )));
    }
    let table = table();
    match table.slots.get(index as usize) {
        None => Err(Failure::argument(format!(
            "the {name} handle was never created"
        ))),
        Some((issued, _)) if generation > *issued || generation == 0 => Err(Failure::argument(
            format!("the {name} handle was never created"),
        )),
        Some((issued, Some(object))) if *issued == generation && object.kind() == kind => {
            Ok(object.clone())
        }
        Some(_) => Err(Failure::state(format!(
            "the {name} handle is stale: its object was released"
        ))),
    }
}

pub(crate) fn runtime(handle: u64) -> Result<Arc<RuntimeHandle>> {
    match lookup(handle, Kind::Runtime)? {
        Object::Runtime(r) => Ok(r),
        _ => unreachable!(),
    }
}

pub(crate) fn map(handle: u64) -> Result<Arc<MapHandle>> {
    match lookup(handle, Kind::Map)? {
        Object::Map(m) => Ok(m),
        _ => unreachable!(),
    }
}

pub(crate) fn session(handle: u64) -> Result<Arc<SessionHandle>> {
    match lookup(handle, Kind::Session)? {
        Object::Session(s) => Ok(s),
        _ => unreachable!(),
    }
}

pub(crate) fn frame(handle: u64) -> Result<Arc<FrameHandle>> {
    match lookup(handle, Kind::Frame)? {
        Object::Frame(f) => Ok(f),
        _ => unreachable!(),
    }
}

pub(crate) fn queue(handle: u64) -> Result<Arc<DeliveryQueue>> {
    match lookup(handle, Kind::Queue)? {
        Object::Queue(q) => Ok(q),
        _ => unreachable!(),
    }
}

pub(crate) fn pacer(handle: u64) -> Result<Arc<FramePacer>> {
    match lookup(handle, Kind::Pacer)? {
        Object::Pacer(p) => Ok(p),
        _ => unreachable!(),
    }
}

pub(crate) fn offscreen_device(handle: u64) -> Result<()> {
    lookup(handle, Kind::OffscreenDevice).map(|_| ())
}
