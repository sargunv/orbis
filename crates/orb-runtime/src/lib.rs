//! The runtime (plan §5.4, §7): a [`Map`] owns a style, a camera and the
//! tiles they need, runs tile jobs on the host's executor, and draws frames
//! with `orb-render`.

#![forbid(unsafe_code)]

mod edit;
mod events;
mod features;
mod globe;
mod glyphs;
mod gpu;
mod image;
mod jobs;
mod labels;
mod lights;
mod line;
mod loader;
mod map;
mod placement;
mod placement_wire;
mod point;
mod queries;
mod query;
mod source;
mod sprite;
mod state;
mod style;
mod surface;
mod symbol;
mod terrain;
#[cfg(test)]
mod testing;
mod tile;

pub use events::MapEvent;

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
    Built, DemUnpack, ImagePixels, Jobs, PoolJobs, ResultMesh, Stream, StyleUpdate, TileDone,
    TileImage, TileRequest, TileResult, TileWorker, pack, unpack,
};
pub use loader::{LoadDone, TileLoader, TileLocation};
pub use map::{FrameReport, Images, JobCounts, Map, Services};
pub use orb_camera::{CameraState, View, Viewport};
pub use orb_layout::fill::Bend;
pub use orb_render::{ARGS_WORDS, SceneCapacity, TargetFormat};
pub use orb_style::{CostModel, FeatureTarget, Transaction};
pub use placement::{InlinePlacement, PlaceDone, PlacementLane, ThreadPlacement};
pub use placement_wire::{
    pack_committed, pack_frame, pack_tables, unpack_committed, unpack_frame, unpack_tables,
};
pub use queries::QueryDone;
pub use query::{Hit, HitFeature, HitIndex, HitPublication, MeshHits};
pub use style::ImageKind;
pub use surface::{
    SurfaceTiles, projection_from_json, projection_to_json, surface_from_json, surface_to_json,
    tms_from_json, tms_to_json,
};
pub use symbol::{ResultSymbols, SYMBOL_TABLES, SpriteIcons, SymbolMeta, SymbolTable};
pub use terrain::TerrainElevation;
pub use tile::{ImageRef, TileBuild};
