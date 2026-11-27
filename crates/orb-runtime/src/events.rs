//! A map's events (GL JS's map events as reference): style and
//! source data, load, idle, errors and warnings, the camera's moves,
//! clicks and notches.

use std::collections::BTreeSet;

use glam::DVec2;
use orb_camera::CameraEvent;

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
    /// The map draws on a new GPU after a device loss: its first frame
    /// there.
    Restored,
    /// What the renderer's device draws otherwise than WebGPU, whenever a
    /// style revision or renderer changes it: each [`Gap`](crate::Gap) the
    /// style uses and the layers it affects; empty once nothing is.
    Degraded {
        gaps: Vec<(crate::Gap, Vec<String>)>,
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
    /// A primary press and release that didn't move the map, in logical
    /// pixels.
    Click {
        position: DVec2,
    },
    DoubleClick {
        position: DVec2,
    },
    /// A touch or pen held still, or a right click (Compose's
    /// `longClick`, GL JS's `contextmenu`).
    LongClick {
        position: DVec2,
    },
    /// Input moved the camera to or across one of the gestures' notches
    /// ([`Movement::notches`](orb_camera::input::Movement::notches)), for
    /// the host to play haptic feedback.
    Notch {
        axis: orb_camera::input::Axis,
        /// In the part's unit; a bearing from 0 up to 360 degrees.
        value: f64,
        /// The notch's kind, from the options.
        kind: u32,
    },
    /// A source's description arrived: its TileJSON (MapLibre Native's
    /// `onSourceChanged`).
    SourceChanged {
        source: String,
    },
    /// A tile of a source arrived from its load: laid out, failed or
    /// given up (MapLibre Native's `onTileAction`).
    Tile {
        source: String,
        tile: orb_crs::TileIndex,
        progress: Progress,
    },
    /// A sprite of the style was asked for, loaded or failed (MapLibre
    /// Native's sprite events). A style without a sprite reports an empty
    /// one loaded, as Native does.
    Sprite {
        id: String,
        url: String,
        progress: Progress,
    },
    /// The map needs a tile of a source the app feeds
    /// ([`HOST_TILES`](crate::HOST_TILES)): answer with
    /// [`Map::host_tile`](crate::Map::host_tile). Once per tile until
    /// answered.
    TileRequest {
        source: String,
        tile: orb_crs::TileIndex,
    },
    /// A batch of heights the app gave
    /// ([`Map::set_host_elevations`](crate::Map::set_host_elevations))
    /// shows, whole, from this frame on; or it had nothing the map draws.
    ElevationsShown {
        source: String,
        batch: u64,
    },
}

/// How a load went, for [`MapEvent::Tile`] and [`MapEvent::Sprite`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Progress {
    Requested,
    Loaded,
    Failed(String),
    /// Given up before it loaded.
    Cancelled,
}

/// A condition frames find while it holds, warned of once when it
/// starts ([`DrapeRefused`](Self::DrapeRefused) once a style revision,
/// as it clears between refusals).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Warn {
    TerrainPipelines,
    TerrainTms { dem: String, map: String },
    PlacementsFull,
    BackgroundPlacementsFull,
    DrapeRoom,
    DrapeRefused { mib: u64, fewer: bool },
    SceneFull,
}

impl std::fmt::Display for Warn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Warn::TerrainPipelines => {
                f.write_str("terrain: the GPU failed to make its pipelines; the map is flat")
            }
            Warn::TerrainTms { dem, map } => write!(
                f,
                "terrain: the DEM source's tiles ({dem}) are not the map's ({map}); the map is flat"
            ),
            Warn::PlacementsFull => f.write_str("out of tile placements; some tiles are not drawn"),
            Warn::BackgroundPlacementsFull => {
                f.write_str("out of tile placements; the background is not drawn")
            }
            Warn::DrapeRoom => {
                f.write_str("too many drape stacks for the drape atlas; terrain is flat")
            }
            Warn::DrapeRefused { mib, fewer } => write!(
                f,
                "the GPU refused a drape texture's {mib} MiB; terrain {} until the map settles",
                if *fewer {
                    "covers the view in fewer tiles"
                } else {
                    "is flat"
                }
            ),
            Warn::SceneFull => f.write_str("out of scene memory; an image layer misses tiles"),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct EventQueue {
    pub(crate) queue: Vec<MapEvent>,
    /// The conditions found since the last drawn frame, and those it
    /// found.
    held: Vec<Warn>,
    shown: Vec<Warn>,
    loaded: bool,
    idle: bool,
    /// The map has been idle once (GL JS `_idleTriggered`).
    pub(crate) idled: bool,
    /// Sources with data since the last drawn frame, by key: an edit
    /// renumbers the plan's sources, and a new style drops the set.
    pub(crate) sources: BTreeSet<u32>,
}

impl EventQueue {
    pub(crate) fn push(&mut self, event: MapEvent) {
        self.queue.push(event);
    }

    /// Notes that `warn` holds for the frame being drawn.
    pub(crate) fn hold(&mut self, warn: Warn) {
        if !self.held.contains(&warn) {
            self.held.push(warn);
        }
    }

    /// Records a frame's end state: `load` once, `idle` on entering it,
    /// and a warning for each condition that started.
    pub(crate) fn frame(&mut self, loaded: bool, idle: bool) {
        for warn in &self.held {
            if !self.shown.contains(warn) {
                let message = warn.to_string();
                self.queue.push(MapEvent::Warning { message });
            }
        }
        self.shown = std::mem::take(&mut self.held);
        if loaded && !self.loaded {
            self.loaded = true;
            self.queue.push(MapEvent::Load);
        }
        if idle && !self.idle {
            self.queue.push(MapEvent::Idle);
            self.idled = true;
        }
        self.idle = idle;
    }

    /// Something will need frames again: the next idle frame reports.
    pub(crate) fn wake(&mut self) {
        self.idle = false;
    }
}
