//! Edits as the renderer sees them: a transaction's changes, keyed by
//! layer and source identity and by [`PropertyId`], each in a class of
//! what it costs.

use orb_expr::{Diagnostic, Inputs, PropertyExpression, Value};

use crate::{PropertyId, Slot};

/// A layer's identity across edits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerKey(pub u32);

/// A source's identity across edits; a source whose definition changes is
/// a new source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceKey(pub u32);

/// A feature-state edit, validated against the document's sources.
#[derive(Clone, Debug, PartialEq)]
pub enum FeatureStateOp {
    Set {
        source: SourceKey,
        source_layer: Option<String>,
        id: String,
        state: Vec<(String, Value<'static>)>,
    },
    Remove {
        source: SourceKey,
        source_layer: Option<String>,
        id: Option<String>,
        key: Option<String>,
    },
}

/// How a paint value binds to the renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Binding {
    /// The same everywhere: a node uniform.
    Constant,
    /// A zoom curve: a node uniform rewritten as the camera zooms.
    Camera,
    /// Per feature: property rows.
    Source,
    /// Per feature and zoom: property rows at two zooms.
    Composite,
    /// Per feature state only, not feature data: the state table
    /// (`StateLookup`). Values that read both are rows, which the state
    /// table overrides for features with state.
    State,
}

impl Binding {
    /// How a parsed paint value binds.
    pub fn of(value: &PropertyExpression) -> Self {
        if value.reads().contains(Inputs::STATE) && !value.reads().contains(Inputs::DATA) {
            return Self::State;
        }
        match value.kind() {
            orb_expr::Kind::Constant => Self::Constant,
            orb_expr::Kind::Camera => Self::Camera,
            orb_expr::Kind::Source => Self::Source,
            orb_expr::Kind::Composite => Self::Composite,
        }
    }

    /// How a property's slot binds: a list by its first value.
    pub(crate) fn of_slot(slot: &Slot) -> Self {
        slot.items()
            .first()
            .and_then(Slot::as_expression)
            .map_or(Self::Constant, Self::of)
    }

    /// Whether the value needs per-feature property rows.
    pub(crate) fn is_rows(self) -> bool {
        matches!(self, Self::Source | Self::Composite)
    }
}

/// What changed in a layer's layout inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutChange {
    Property(PropertyId),
    Filter,
    /// `minzoom` or `maxzoom`: GL JS lays out no tiles outside them.
    ZoomRange,
}

/// One change of a patch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Change {
    /// The document needs a full reload (a sprite change, say).
    Restyle,
    /// The root `sprite` changed: its images load again in place of the
    /// old ones, and symbol layers lay out again with them.
    Sprite,
    /// A root value frames read from the style (sky, light, terrain,
    /// transition, camera defaults) changed.
    Root,
    /// The root `projection` changed: every tile lays out again for the
    /// new surface, from cached data.
    Projection,
    SourceAdded(SourceKey),
    SourceRemoved(SourceKey),
    /// A GeoJSON source's new data: its layers lay out again from it.
    SourceData(SourceKey),
    LayerAdded(LayerKey),
    LayerRemoved(LayerKey),
    /// The order of the layers that stayed changed.
    LayersReordered,
    Paint {
        layer: LayerKey,
        id: PropertyId,
        from: Binding,
        to: Binding,
    },
    /// `visibility`: shows or hides the layer's nodes.
    Visibility(LayerKey),
    /// The render targets a layer draws into or reads, or an image-process
    /// layer's operations.
    Targets(LayerKey),
    /// A paint property's `*-transition`: how its next changes ease.
    Transitions(LayerKey),
    Layout {
        layer: LayerKey,
        change: LayoutChange,
    },
}

/// What a change costs the renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Class {
    /// Node uniforms (the frame block).
    Block,
    /// Feature-state lookups: the state table's layout and rows.
    State,
    /// Property rows re-evaluated from cached tiles, without layout.
    Column,
    /// One layer laid out again from cached tiles.
    Relayout,
    /// Render nodes added, removed or reordered.
    Nodes,
    /// A source's tiles loaded again.
    Reload,
    /// Everything loaded again.
    Restyle,
}

impl Change {
    pub fn class(&self) -> Class {
        match self {
            Self::Restyle => Class::Restyle,
            Self::Root => Class::Block,
            Self::Projection | Self::Sprite | Self::SourceData(_) => Class::Relayout,
            Self::SourceAdded(_) | Self::SourceRemoved(_) => Class::Reload,
            Self::LayerAdded(_)
            | Self::LayerRemoved(_)
            | Self::LayersReordered
            | Self::Targets(_) => Class::Nodes,
            Self::Visibility(_) | Self::Transitions(_) => Class::Block,
            Self::Layout { .. } => Class::Relayout,
            Self::Paint { from, to, .. } => {
                if to.is_rows() {
                    Class::Column
                } else if *from == Binding::State || *to == Binding::State {
                    Class::State
                } else {
                    Class::Block
                }
            }
        }
    }

    /// The layer the change is to, if one.
    pub fn layer(&self) -> Option<LayerKey> {
        match self {
            Self::LayerAdded(l)
            | Self::LayerRemoved(l)
            | Self::Visibility(l)
            | Self::Targets(l)
            | Self::Transitions(l)
            | Self::Paint { layer: l, .. }
            | Self::Layout { layer: l, .. } => Some(*l),
            _ => None,
        }
    }
}

/// What a transaction changed.
#[derive(Clone, Debug, Default)]
pub struct Patch {
    /// The document version this patch produced; a feature-state-only
    /// transaction keeps the version it was applied to.
    pub version: u64,
    pub changes: Vec<Change>,
    pub feature_state: Vec<FeatureStateOp>,
    /// Warnings of the re-parsed layers and sources.
    pub warnings: Vec<Diagnostic>,
    /// GeoJSON sources whose data changed by diffs alone (GL JS
    /// `updateData`), with the diffs in order: job hosts holding the
    /// source's data apply them rather than receive it whole.
    pub source_diffs: Vec<(SourceKey, Vec<std::sync::Arc<orb_expr::json::Json>>)>,
}

impl Patch {
    /// The costliest class among the changes.
    pub fn class(&self) -> Option<Class> {
        self.changes.iter().map(Change::class).max()
    }

    /// The patches of transactions applied in turn, as one: each change
    /// once, and each source's diffs in order, but none for a source an
    /// edit gave whole data, whose JSON holds them already. Feature state
    /// and warnings stay with the edits that made them.
    pub fn merge(edits: impl IntoIterator<Item = Patch>) -> Patch {
        let mut merged = Patch::default();
        let mut whole = Vec::new();
        for edit in edits {
            merged.version = edit.version;
            for change in edit.changes {
                if let Change::SourceAdded(key) | Change::SourceData(key) = change
                    && !edit.source_diffs.iter().any(|(k, _)| *k == key)
                {
                    whole.push(key);
                }
                if !merged.changes.contains(&change) {
                    merged.changes.push(change);
                }
            }
            for (key, diffs) in edit.source_diffs {
                match merged.source_diffs.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, held)) => held.extend(diffs),
                    None => merged.source_diffs.push((key, diffs)),
                }
            }
        }
        merged.source_diffs.retain(|(key, _)| !whole.contains(key));
        merged
    }
}
