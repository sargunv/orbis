//! A map's events (plan §7.5, GL JS's map events as reference): style and
//! source data, load, idle, errors and warnings, the camera's moves and
//! clicks.

use std::collections::BTreeSet;

use glam::DVec2;
use orb_camera::CameraEvent;
use orb_interact::Button;

/// Something the map did, drained with [`Map::take_events`](crate::Map::take_events).
#[derive(Clone, Debug, PartialEq)]
pub enum MapEvent {
    /// A style was installed (GL JS `style.load`). Until the app or the
    /// user moved the camera, it took the style's.
    StyleLoad,
    /// The style changed: loaded or edited (GL JS `styledata`). Frames show
    /// the change in full from `generation` on.
    StyleData {
        generation: u64,
    },
    /// A source's metadata or tiles arrived this frame (GL JS `sourcedata`).
    /// `loaded`: none of its tiles are loading.
    SourceData {
        source: String,
        loaded: bool,
    },
    /// The first frame with the style and every visible tile in (GL JS
    /// `load`); once per map.
    Load,
    /// The frame drawn last before the map has nothing to do: no loads,
    /// fades or placement in flight, and the camera at rest. Hosts draw no
    /// more frames until something changes (GL JS `idle`). Once per rest.
    Idle,
    /// A tile's layers name an image the style doesn't have (GL JS
    /// `styleimagemissing`); once per name until it's added. A host may add
    /// it with [`Map::add_image`](crate::Map::add_image).
    StyleImageMissing {
        name: String,
    },
    /// A style, source, tile or sprite failed to load (GL JS `error`).
    Error {
        message: String,
    },
    /// Something in the style or its data the map can't draw as written.
    Warning {
        message: String,
    },
    /// The map renders at `tier`: on its first frame, and on its first
    /// frame on a new GPU (after device loss, perhaps on another tier).
    /// What the tier gives up arrives as warnings (plan §8.7).
    Capability {
        tier: orb_hal::Tier,
    },
    /// What the host's player of a `video` source should do, on a host
    /// that plays video ([`Map::set_plays_video`](crate::Map::set_plays_video)).
    Video {
        source: String,
        request: crate::VideoRequest,
    },
    /// The camera started, went on or stopped moving, zooming, rotating or
    /// pitching, or an animation ended.
    Camera(CameraEvent),
    /// A press and release that didn't move the map, in logical pixels.
    Click {
        position: DVec2,
        button: Button,
    },
    DoubleClick {
        position: DVec2,
    },
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
