//! The style model: the [`Style`] that a v8 style parses into, with
//! table-driven layers. A layer holds
//! a [`Slot`] per property of its type, indexed by [`PropertyId`] in one
//! table of every layer property's type, default and section, and read
//! typed through a [`Key`]. The [`v8`] module is the front end that
//! builds it.

#![forbid(unsafe_code)]

mod atmosphere;
mod change;
mod layer;
mod light;
mod property;
mod resources;
mod source;
mod spec;
mod style;
mod transition;
pub mod v8;

pub use atmosphere::Atmosphere;
pub use change::{
    Binding, Change, Class, FeatureStateOp, LayerKey, LayoutChange, Patch, SourceKey,
};
pub use layer::Layer;
pub use light::Light;
pub use orb_expr::{Color, Kind, PropertyExpression, Value};
pub use property::*;
pub use resources::{FontFace, FontFaces, Sprite, Terrain};
pub use source::{
    Cluster, ClusterProperty, GeoJsonData, GeoJsonSource, ImageContent, ImageSource, PromoteId,
    RasterDemSource, RasterSource, Source, SourceKind, TileSet, VectorSource,
};
pub use spec::*;
pub use style::Style;
pub use transition::{Transition, TransitionOverride};
