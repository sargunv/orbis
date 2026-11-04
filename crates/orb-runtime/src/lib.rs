//! The runtime (plan §5.4, §7): a [`Map`] owns a style, a camera and the
//! tiles they need, runs tile jobs on the host's executor, and draws frames
//! with `orb-render`.

#![forbid(unsafe_code)]

mod edit;
mod events;
mod features;
mod glyphs;
mod gpu;
mod image;
mod jobs;
mod labels;
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
pub use features::{FeatureRequest, FeatureSelect, FeaturesDone, TileFeature};
pub use glyphs::{FontFiles, GlyphBitmap, GlyphRaster, LocalGlyphs, RasterDone};
pub use jobs::{
    Built, DemUnpack, ImagePixels, Jobs, PoolJobs, ResultMesh, Stream, StyleUpdate, TileDone,
    TileImage, TileRequest, TileResult, TileWorker, pack, unpack,
};
pub use loader::{LoadDone, TileLoader, TileLocation};
pub use map::{FrameReport, Images, JobCounts, Map, Services};
pub use orb_camera::{CameraState, View, Viewport};
pub use orb_render::{ARGS_WORDS, SceneCapacity, TargetFormat};
pub use orb_style::{CostModel, FeatureTarget, Transaction};
pub use placement::{InlinePlacement, PlaceDone, PlacementLane, ThreadPlacement};
pub use placement_wire::{
    pack_committed, pack_frame, pack_tables, unpack_committed, unpack_frame, unpack_tables,
};
pub use queries::QueryDone;
pub use query::{Hit, HitFeature, HitIndex};
pub use style::ImageKind;
pub use surface::{projection_from_json, projection_to_json};
pub use symbol::{ResultSymbols, SYMBOL_TABLES, SpriteIcons, SymbolMeta, SymbolTable};
pub use terrain::TerrainElevation;
pub use tile::{ImageRef, TileBuild};
