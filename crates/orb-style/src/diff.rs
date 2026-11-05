//! Diff of a transaction's draft against the document, into classified
//! changes.

use std::collections::{HashMap, HashSet};

use orb_expr::PropertyExpression;
use orb_expr::json::{Json, JsonValue, Member};
use orb_v8::{Diagnostic, Diagnostics, Layer, Source, Style, Value};

use crate::CostModel;
use crate::doc::{Draft, LayerKey, SourceKey, StyleDoc, member, synthetic};
use crate::tx::FeatureStateOp;
use crate::{same, same_opt};

/// How a paint value binds to the renderer (plan §11.3's binding plans).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Binding {
    /// The same everywhere: a node uniform.
    Constant,
    /// A zoom curve: a node uniform rewritten as the camera zooms.
    Camera,
    /// Per feature: property rows (`TableLookup`).
    Source,
    /// Per feature and zoom: property rows at two zooms (`TableCurve`).
    Composite,
    /// Per feature state only, not feature data: the state table
    /// (`StateLookup`).
    State,
    /// Per feature (and zoom and state), interpreted on the GPU from
    /// input columns beside the property rows (`GpuBytecode`).
    Bytecode,
}

impl Binding {
    /// Whether the value needs per-feature property rows or input columns.
    pub fn is_rows(self) -> bool {
        matches!(self, Self::Source | Self::Composite | Self::Bytecode)
    }

    /// Whether the value is a uniform alone.
    pub fn is_uniform(self) -> bool {
        matches!(self, Self::Constant | Self::Camera)
    }
}

/// What changed in a layer's layout inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutChange {
    Property(String),
    Filter,
    /// `minzoom` or `maxzoom`: GL JS lays out no tiles outside them.
    ZoomRange,
}

/// One change of a patch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Change {
    /// The document needs a full reload (a sprite change, say).
    Restyle,
    /// The root `projection` changed: every tile lays out again for the
    /// new surface, from cached data.
    Projection,
    SourceAdded(SourceKey),
    SourceRemoved(SourceKey),
    LayerAdded(LayerKey),
    LayerRemoved(LayerKey),
    /// The order of the layers that stayed changed.
    LayersReordered,
    Paint {
        layer: LayerKey,
        name: String,
        from: Binding,
        to: Binding,
    },
    /// `visibility`: shows or hides the layer's nodes.
    Visibility(LayerKey),
    Layout {
        layer: LayerKey,
        change: LayoutChange,
    },
}

/// Plan §11.2's S6 classes: what a change costs the renderer.
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
            Self::Projection => Class::Relayout,
            Self::SourceAdded(_) | Self::SourceRemoved(_) => Class::Reload,
            Self::LayerAdded(_) | Self::LayerRemoved(_) | Self::LayersReordered => Class::Nodes,
            Self::Visibility(_) => Class::Block,
            Self::Layout { .. } => Class::Relayout,
            // A program edit keeps its columns when they pack the same;
            // the renderer, which packs them, lowers that to `Block`.
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
}

impl Patch {
    /// The costliest class among the changes.
    pub fn class(&self) -> Option<Class> {
        self.changes.iter().map(Change::class).max()
    }
}

/// Whether `json` reads any of `keys` through `global-state`.
fn reads_global(json: &Json, keys: &HashSet<&str>) -> bool {
    match &json.value {
        JsonValue::Array(items) => {
            matches!(items.as_slice(), [op, key] if op.as_str() == Some("global-state")
                && key.as_str().is_some_and(|k| keys.contains(k)))
                || items.iter().any(|i| reads_global(i, keys))
        }
        JsonValue::Object(members) => members.iter().any(|m| reads_global(&m.value, keys)),
        _ => false,
    }
}

/// The binding of a paint value of a `layer_type` layer.
fn binding(
    cost: &CostModel,
    layer_type: &str,
    name: &str,
    json: Option<&Json>,
    state: &[(String, Value<'static>)],
) -> Binding {
    json.zip(orb_v8::property_spec(layer_type, true, name))
        .and_then(|(json, spec)| PropertyExpression::parse_with_state(json, &spec, state).ok())
        .map_or(Binding::Constant, |parsed| cost.binding(name, &parsed))
}

fn section<'a>(json: &'a Json, key: &str) -> &'a [Member] {
    json.get(key).and_then(Json::as_object).unwrap_or_default()
}

/// The names in either object, in order of first appearance.
fn names<'a>(a: &'a [Member], b: &'a [Member]) -> Vec<&'a str> {
    let mut names: Vec<&str> = Vec::new();
    for m in a.iter().chain(b) {
        if !names.contains(&m.key.as_str()) {
            names.push(&m.key);
        }
    }
    names
}

/// The changes between a layer's JSON before and after, given the global
/// state keys whose values changed.
#[allow(clippy::too_many_arguments)]
fn layer_changes(
    cost: &CostModel,
    key: LayerKey,
    old: &Json,
    new: &Json,
    old_state: &[(String, Value<'static>)],
    new_state: &[(String, Value<'static>)],
    changed: &HashSet<&str>,
    out: &mut Vec<Change>,
) {
    let ty = new.get("type").and_then(Json::as_str).unwrap_or_default();
    let differs = |a: Option<&Json>, b: Option<&Json>| {
        !same_opt(a, b) || b.is_some_and(|b| reads_global(b, changed))
    };
    let (old_paint, new_paint) = (section(old, "paint"), section(new, "paint"));
    for name in names(old_paint, new_paint) {
        if name.ends_with("-transition") {
            continue;
        }
        let (a, b) = (member(old_paint, name), member(new_paint, name));
        if differs(a, b) {
            out.push(Change::Paint {
                layer: key,
                name: name.into(),
                from: binding(cost, ty, name, a, old_state),
                to: binding(cost, ty, name, b, new_state),
            });
        }
    }
    let (old_layout, new_layout) = (section(old, "layout"), section(new, "layout"));
    for name in names(old_layout, new_layout) {
        if differs(member(old_layout, name), member(new_layout, name)) {
            out.push(if name == "visibility" {
                Change::Visibility(key)
            } else {
                Change::Layout {
                    layer: key,
                    change: LayoutChange::Property(name.into()),
                }
            });
        }
    }
    if differs(old.get("filter"), new.get("filter")) {
        out.push(Change::Layout {
            layer: key,
            change: LayoutChange::Filter,
        });
    }
    if ["minzoom", "maxzoom"]
        .iter()
        .any(|k| !same_opt(old.get(k), new.get(k)))
    {
        out.push(Change::Layout {
            layer: key,
            change: LayoutChange::ZoomRange,
        });
    }
}

/// The root keys whose change needs a full reload.
const RESTYLE_ROOT: &[&str] = &["sprite", "glyphs", "font-faces", "light", "sky"];

/// The new snapshot and the patch from `doc` to `draft`.
pub(crate) fn diff(doc: &StyleDoc, draft: &Draft) -> Result<(Style, Patch), Diagnostics> {
    let old = doc.style();
    let mut changes = Vec::new();
    let mut warnings = Vec::new();

    // The root: camera and state defaults, sprites.
    let mut style = Style {
        name: old.name.clone(),
        center: old.center,
        zoom: old.zoom,
        bearing: old.bearing,
        pitch: old.pitch,
        roll: old.roll,
        sprites: old.sprites.clone(),
        glyphs: old.glyphs.clone(),
        font_faces: old.font_faces.clone(),
        lights: old.lights.clone(),
        atmosphere: old.atmosphere.clone(),
        terrain: None,
        globe: old.globe,
        state: old.state.clone(),
        sources: Vec::new(),
        layers: Vec::new(),
        warnings: Vec::new(),
    };
    if draft.new_root {
        // Terrain names a source: it is parsed with the sources below.
        let mut members: Vec<Member> = draft
            .root
            .iter()
            .filter(|m| m.key != "terrain")
            .cloned()
            .collect();
        for (key, value) in [
            ("sources", JsonValue::Object(Vec::new())),
            ("layers", JsonValue::Array(Vec::new())),
        ] {
            members.push(Member {
                key: key.into(),
                key_span: Default::default(),
                value: synthetic(value),
            });
        }
        let root = orb_v8::parse(&synthetic(JsonValue::Object(members)).to_string())?;
        if RESTYLE_ROOT
            .iter()
            .any(|k| !same_opt(member(&doc.root, k), member(&draft.root, k)))
        {
            changes.push(Change::Restyle);
        }
        if !same_opt(
            member(&doc.root, "projection"),
            member(&draft.root, "projection"),
        ) {
            changes.push(Change::Projection);
        }
        warnings.extend(root.warnings.iter().cloned());
        style = Style {
            warnings: Vec::new(),
            ..root
        };
    }

    let old_state = doc.global_state();
    let new_state = orb_v8::resolve_state(&style.state, &draft.overrides);
    let changed: HashSet<&str> = old_state
        .iter()
        .chain(&new_state)
        .map(|(k, _)| k.as_str())
        .filter(|k| {
            let value = |state: &[(String, Value<'static>)]| {
                state.iter().find(|(s, _)| s == k).map(|(_, v)| v.clone())
            };
            value(&old_state) != value(&new_state)
        })
        .collect();

    if let Some(json) = member(&draft.root, "sky")
        && reads_global(json, &changed)
    {
        let (sky, w) = orb_v8::parse_sky(json, &new_state)?;
        style.atmosphere = Some(sky);
        warnings.extend(w);
        if !changes.contains(&Change::Restyle) {
            changes.push(Change::Restyle);
        }
    }

    // Sources.
    let old_sources: HashMap<SourceKey, &Source> = doc.source_keys().zip(&old.sources).collect();
    for entry in &draft.sources {
        match old_sources.get(&entry.key) {
            Some(source) => style.sources.push((*source).clone()),
            None => {
                let (source, w) = orb_v8::parse_source(&entry.id, &entry.json)?;
                warnings.extend(w);
                style.sources.push(source);
                changes.push(Change::SourceAdded(entry.key));
            }
        }
    }
    for key in doc.source_keys() {
        if !draft.sources.iter().any(|s| s.key == key) {
            changes.push(Change::SourceRemoved(key));
        }
    }

    // Terrain reads the new sources; switching it needs no reload.
    if let Some(json) = member(&draft.root, "terrain") {
        let (terrain, w) = orb_v8::parse_terrain(json, &style.sources)?;
        warnings.extend(w);
        style.terrain = Some(terrain);
    }

    // Layers.
    let old_layers: HashMap<LayerKey, (usize, &Layer)> = doc
        .layer_keys()
        .zip(&old.layers)
        .enumerate()
        .map(|(i, (k, l))| (k, (i, l)))
        .collect();
    let mut ids = HashSet::new();
    for (index, entry) in draft.layers.iter().enumerate() {
        let id = entry
            .json
            .get("id")
            .and_then(Json::as_str)
            .unwrap_or_default();
        if !ids.insert(id) {
            return Err(Diagnostics {
                errors: vec![Diagnostic {
                    path: format!("layers[{index}].id"),
                    span: Default::default(),
                    message: format!("duplicate layer id \"{id}\""),
                }],
                warnings: Vec::new(),
            });
        }
        let old_entry = old_layers.get(&entry.key);
        let global = reads_global(&entry.json, &changed);
        let reparse = old_entry.is_none() || draft.dirty.contains(&entry.key) || global;
        if !reparse {
            style.layers.push(old_entry.expect("kept").1.clone());
            continue;
        }
        let (layer, w) = orb_v8::parse_layer(&entry.json, index, &style.sources, &new_state)?;
        warnings.extend(w);
        style.layers.push(layer);
        match old_entry {
            None => changes.push(Change::LayerAdded(entry.key)),
            Some(&(old_index, _)) => layer_changes(
                &doc.cost,
                entry.key,
                doc.layer_json(old_index),
                &entry.json,
                &old_state,
                &new_state,
                &changed,
                &mut changes,
            ),
        }
    }
    for key in doc.layer_keys() {
        if !draft.layers.iter().any(|l| l.key == key) {
            changes.push(Change::LayerRemoved(key));
        }
    }
    let kept: Vec<LayerKey> = draft
        .layers
        .iter()
        .map(|l| l.key)
        .filter(|k| old_layers.contains_key(k))
        .collect();
    let before: Vec<LayerKey> = doc
        .layer_keys()
        .filter(|k| draft.layers.iter().any(|l| l.key == *k))
        .collect();
    if kept != before {
        changes.push(Change::LayersReordered);
    }
    // Unchanged layers' JSON is shared; only edited ones were compared.
    debug_assert!(draft.layers.iter().all(|l| {
        old_layers
            .get(&l.key)
            .is_none_or(|&(i, _)| draft.dirty.contains(&l.key) || same(doc.layer_json(i), &l.json))
    }));
    style.warnings = warnings.clone();
    Ok((
        style,
        Patch {
            version: 0,
            changes,
            feature_state: Vec::new(),
            warnings,
        },
    ))
}
