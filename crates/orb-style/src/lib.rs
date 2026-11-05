//! The native scene document: the [`Scene`] that a v8 style
//! and a scene document parse into, with table-driven layers. A layer holds
//! a [`Slot`] per property of its type, indexed by [`PropertyId`] in one
//! table of every layer property's type, default and section, and read
//! typed through a [`Key`]. `orb-v8` is the front end that builds it.

#![forbid(unsafe_code)]

mod atmosphere;
mod change;
mod layer;
mod lights;
mod property;
mod resources;
mod scene;
mod source;
mod spec;
mod transition;

pub use atmosphere::Atmosphere;
pub use change::{
    Binding, Change, Class, FeatureStateOp, LayerKey, LayoutChange, Patch, SourceKey,
};
pub use layer::Layer;
pub use lights::{AmbientLight, Direction, DirectionalLight, LayerMaterial, Level, Lights};
pub use orb_expr::{Color, Kind, PropertyExpression, Value};
pub use property::*;
pub use resources::{FontFace, FontFaces, Sprite, Terrain};
pub use scene::{Dialect, Scene};
pub use source::{
    Cluster, ClusterProperty, GeoJsonData, GeoJsonSource, ImageContent, ImageSource, PromoteId,
    RasterDemSource, RasterSource, Source, SourceKind, TileSet, VectorSource,
};
pub use spec::*;
pub use transition::{Transition, TransitionOverride};
