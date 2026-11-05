//! The runtime (plan §5.4, §7): a [`Map`] owns a style, the camera and its
//! gestures, and the tiles they need, runs tile jobs on the host's executor, and draws frames
//! with `orb-render`.

#![forbid(unsafe_code)]

pub mod batteries;
mod camera;
mod edit;
mod events;
mod features;
mod globe;
mod glyphs;
mod image;
mod jobs;
mod labels;
mod lights;
mod line;
mod loader;
mod map;
pub mod offline;
mod placement;
mod placement_wire;
mod point;
mod queries;
mod query;
mod sprite;
mod state;
mod style;
mod style_image;
mod surface;
mod symbol;
mod terrain;
#[cfg(test)]
mod testing;
mod tile;
mod transitions;
mod video;
pub mod wire;

pub use events::MapEvent;
pub use orb_a11y::Snapshot as SemanticSnapshot;
pub use video::{NO_VIDEO, VideoRequest};

/// The warning a style with terrain gets on Compat (plan §8.7), whose
/// maps draw flat.
pub const COMPAT_NO_TERRAIN: &str =
    "terrain is not drawn on the Compat tier (WebGL2, GLES 3.0); the map is flat";
/// The warning heatmap layers get on devices without float render
/// targets (Compat without EXT_color_buffer_float).
pub const COMPAT_NO_HEATMAP: &str = "heatmap requires filterable, blendable rgba16float render attachments (Compat: EXT_color_buffer_float); heatmap layers are skipped";
pub use features::{FeatureRequest, FeatureSelect, FeaturesDone, TileFeature};
pub use glyphs::{FontFiles, GlyphBitmap, GlyphRaster, LocalGlyphs, RasterDone};
pub use jobs::{
    Built, DemUnpack, ImagePixels, JobReply, JobRequest, Jobs, PoolJobs, ResultMesh, Stream,
    StyleUpdate, TileDone, TileImage, TileRequest, TileResult, TileWorker, pack, packed_header,
    unpack,
};
pub use loader::{LoadDone, TileLoader, TileLocation};
pub use map::{FrameReport, GridTile, Images, JobCounts, Map, Memory, RenderMode, Services};
pub use orb_camera::{
    AnimationId, Bounds, CameraEvent, CameraState, CameraTarget, Constraints, EaseOptions,
    FitOptions, FlyOptions, View, Viewport,
};
pub use orb_interact::{Button, Feel, Input};
pub use orb_layout::fill::Bend;
pub use orb_render::Renderer;
pub use orb_render::{
    ARGS_WORDS, DESKTOP_BUDGET, ImageMetadata, MOBILE_BUDGET, PoolMemory, SceneCapacity,
    SceneMemory, TargetFormat,
};
pub use orb_v8::{FeatureTarget, Transaction};
pub use placement::{InlinePlacement, PlaceDone, PlacementLane, ThreadPlacement};
pub use placement_wire::{
    pack_committed, pack_frame, pack_tables, unpack_committed, unpack_frame, unpack_tables,
};
pub use queries::QueryDone;
pub use query::{Hit, HitFeature, MeshHits};
pub use style::ImageKind;
pub use style_image::StyleImage;
pub use surface::{
    SurfaceTiles, projection_from_json, projection_to_json, surface_from_json, surface_to_json,
    tms_from_json, tms_to_json,
};
pub use symbol::{ResultSymbols, SYMBOL_TABLES, SpriteIcons, SymbolMeta, SymbolTable};
pub use terrain::TerrainElevation;
pub use tile::{ImageRef, TileBuild};
