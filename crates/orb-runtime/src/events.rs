//! A map's events (plan §7.5, GL JS's map events as reference): style and
//! source data, load, idle and errors. Camera events come from the camera
//! store (`orb_camera::Camera`), which the host owns.

use std::collections::BTreeSet;

use orb_camera::CameraState;

/// Something the map did, drained with [`Map::take_events`](crate::Map::take_events).
#[derive(Clone, Debug, PartialEq)]
pub enum MapEvent {
    /// A style was installed (GL JS `style.load`). `camera` is the style's
    /// own camera when the map took it, because the host hadn't moved the
    /// camera: hosts with a camera store adopt it.
    StyleLoad { camera: Option<CameraState> },
    /// The style changed: loaded or edited (GL JS `styledata`). Frames show
    /// the change in full from `generation` on.
    StyleData { generation: u64 },
    /// A source's metadata or tiles arrived this frame (GL JS `sourcedata`).
    /// `loaded`: none of its tiles are loading.
    SourceData { source: String, loaded: bool },
    /// The first frame with the style and every visible tile in (GL JS
    /// `load`); once per map.
    Load,
    /// The frame drawn last before the map has nothing to do: no loads,
    /// fades or placement in flight. Hosts draw no more frames until
    /// something changes (GL JS `idle`).
    Idle,
    /// A style, source, tile or sprite failed to load (GL JS `error`).
    Error { message: String },
    /// The map renders at `tier`: on its first frame, and on its first
    /// frame on a new GPU (after device loss, perhaps on another tier).
    /// What the tier gives up arrives as warnings (plan §8.7).
    Capability { tier: orb_hal::Tier },
}

#[derive(Debug, Default)]
pub(crate) struct EventQueue {
    pub(crate) queue: Vec<MapEvent>,
    loaded: bool,
    idle: bool,
    /// Sources with data this frame, by plan index.
    pub(crate) sources: BTreeSet<usize>,
}

impl EventQueue {
    pub(crate) fn push(&mut self, event: MapEvent) {
        self.queue.push(event);
    }

    /// Records a frame's end state: `load` once, `idle` on entering it.
    pub(crate) fn frame(&mut self, loaded: bool, idle: bool) {
        if loaded && !self.loaded {
            self.loaded = true;
            self.queue.push(MapEvent::Load);
        }
        if idle && !self.idle {
            self.queue.push(MapEvent::Idle);
        }
        self.idle = idle;
    }

    /// Something will need frames again: the next idle frame reports.
    pub(crate) fn wake(&mut self) {
        self.idle = false;
    }
}
