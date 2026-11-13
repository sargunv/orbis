//! The runtime: a [`Map`] owns a style, the camera and its
//! gestures, and the tiles they need, runs tile jobs on the host's executor, and draws frames
//! with `orb-render`.

#![forbid(unsafe_code)]

mod arrival;
mod camera;
mod edit;
mod elevation;
mod events;
mod features;
mod frame;
mod geojson;
mod globe;
mod glyphs;
mod host_heights;
mod host_tiles;
mod image;
mod job_host;
mod jobs;
mod labels;
mod light;
mod line;
mod loader;
mod map;
mod meshes;
pub mod modules;
#[cfg(feature = "sqlite")]
pub mod native_db;
#[cfg(feature = "sqlite")]
pub use orb_sqlite;
pub mod offline;
mod options;
mod paint;
mod pipelines;
mod placement;
mod point;
mod queries;
mod query;
mod resident;
mod root;
mod schedule;
mod settle;
mod sprite;
pub mod stages;
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

pub use events::{MapEvent, Progress};
pub use video::{NO_VIDEO, VideoRequest};

pub use features::{FeatureRequest, FeaturesDone, TileFeature};
pub use geojson::{ClusterQuery, ClusterRequest};
pub use glyphs::{FontFiles, GlyphBitmap, GlyphRaster, LocalGlyphs, RasterDone};
pub use host_heights::HostElevations;
pub use job_host::{JobHost, Loaded};
pub use jobs::{
    ImagePixels, JobReply, JobRequest, Jobs, PoolJobs, TileDone, TileImage, TileOutcome,
    TileResult, pack, packed_header, unpack, unpremultiply,
};
pub use loader::TileLocation;
pub use map::{
    FrameBudget, FrameReport, FrameTimes, GridTile, Images, JobCounts, Map, Memory, MissingTiles,
    RenderMode, Services,
};
pub use options::MapOptions;
pub use orb_camera::input::{Button, Input};
pub use orb_camera::{
    AnimationId, Bounds, CameraEvent, CameraState, CameraTarget, Constraints, EaseOptions,
    FitOptions, FlyOptions, Skew, View, Viewport,
};
pub use orb_expr::json;
pub use orb_layout::fill::Bend;
pub use orb_render::{
    ARGS_WORDS, ImageMetadata, PoolMemory, Resized, SceneCapacity, SceneMemory, TargetFormat,
};
pub use orb_render::{Gap, Renderer};
pub use orb_source::endpoint::{HOST_TILES, HostTile};
pub use orb_style::v8::{FeatureTarget, Op, StyleDoc, Transaction};
pub use orb_style::{Preview, Previews};
pub use placement::{InlinePlacement, PlaceDone, PlacementLane};
pub use queries::QueryDone;
pub use query::Hit;
pub use schedule::Budget;
pub use settle::Settle;
pub use style::ImageKind;
pub use style_image::StyleImage;
pub use symbol::SpriteIcons;
pub use terrain::{TerrainElevation, apply_pyramid_delta, pyramid_delta};
pub use tile::TileBuild;
pub use worker::{Built, DemUnpack, StyleUpdate, TileRequest, TileWorker};
