//! The runtime (plan §5.4, §7): a [`Map`] owns a style, the camera and its
//! gestures, and the tiles they need, runs tile jobs on the host's executor, and draws frames
//! with `orb-render`.

#![forbid(unsafe_code)]

mod arrival;
pub mod batteries;
mod camera;
mod edit;
mod events;
mod features;
mod frame;
mod geojson;
mod globe;
mod glyphs;
mod host_tiles;
mod image;
mod job_host;
mod jobs;
mod labels;
mod lights;
mod line;
mod loader;
mod map;
mod meshes;
mod model;
pub mod offline;
mod paint;
mod placement;
mod point;
mod queries;
mod query;
mod resident;
mod restore;
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
mod worker;

pub use events::MapEvent;
pub use orb_a11y::Snapshot as SemanticSnapshot;
pub use video::{NO_VIDEO, VideoRequest};

/// The warning a style with terrain gets on Compat (plan §8.7), whose
/// maps draw flat.
pub const COMPAT_NO_TERRAIN: &str =
    "terrain is not drawn on the Compat tier (WebGL2, GLES 3.0); the map is flat";
/// The warning scenes whose lights cast shadows get on the Compat tier.
pub const COMPAT_NO_SHADOWS: &str = "shadows are not drawn on the Compat tier (WebGL2, GLES 3.0)";
/// The warning they get while the view shows a globe.
pub const GLOBE_NO_SHADOWS: &str =
    "shadows are drawn on a plane only, not while the view shows a globe";
/// The warning heatmap layers get on devices without float render
/// targets (Compat without EXT_color_buffer_float).
pub const COMPAT_NO_HEATMAP: &str = "heatmap requires filterable, blendable rgba16float render attachments (Compat: EXT_color_buffer_float); heatmap layers are skipped";
/// Compat's warning for a style with models (plan §8.7).
pub const COMPAT_NO_MODELS: &str = "model layers draw on WebGPU and native GPUs only (Compat: WebGL2 and GLES); model layers are skipped";
/// Compat's warning for a scene's lights over the whole map (decision 0048).
pub const COMPAT_UNLIT: &str = "a scene's lights shade the whole map on WebGPU and native GPUs only (Compat: WebGL2 and GLES); its layers draw unlit";
pub use features::{FeatureRequest, FeatureSelect, FeaturesDone, TileFeature};
pub use geojson::{ClusterQuery, ClusterRequest};
pub use glyphs::{FontFiles, GlyphBitmap, GlyphRaster, LocalGlyphs, RasterDone};
pub use job_host::{JobHost, Loaded, LoadedDone, ModelDone};
pub use jobs::{
    ImagePixels, JobReply, JobRequest, Jobs, PoolJobs, ResultMesh, Stream, TileDone, TileImage,
    TileResult, pack, packed_header, unpack,
};
pub use loader::{LoadDone, TileLoader, TileLocation};
pub use map::{FrameReport, GridTile, Images, JobCounts, Map, Memory, RenderMode, Services};
pub use orb_camera::{
    AnimationId, Bounds, CameraEvent, CameraState, CameraTarget, Constraints, EaseOptions,
    FitOptions, FlyOptions, View, Viewport,
};
pub use orb_expr::json;
pub use orb_interact::{Button, Feel, Input};
pub use orb_layout::fill::Bend;
pub use orb_render::Renderer;
pub use orb_render::{
    ARGS_WORDS, DESKTOP_BUDGET, ImageMetadata, MOBILE_BUDGET, PoolMemory, SceneCapacity,
    SceneMemory, TargetFormat,
};
pub use orb_source::endpoint::{HOST_TILES, HostTile};
pub use orb_v8::{FeatureTarget, Op, StyleDoc, Transaction};
pub use placement::{InlinePlacement, PlaceDone, PlacementLane, ThreadPlacement};
pub use queries::QueryDone;
pub use query::{Hit, HitGeometry, MeshHits};
pub use restore::AppState;
pub use style::ImageKind;
pub use style_image::StyleImage;
pub use surface::SurfaceTiles;
pub use symbol::{ResultSymbols, SYMBOL_TABLES, SpriteIcons, SymbolMeta, SymbolTable};
pub use terrain::TerrainElevation;
pub use tile::{ImageRef, TileBuild};
pub use worker::{Built, DemUnpack, StyleUpdate, TileRequest, TileWorker};
