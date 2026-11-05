//! The persistent document.

use orb_style::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::{Diagnostic, Diagnostics};
use orb_expr::json::{self, Json, JsonValue, Member, Span};

use crate::diff;
use crate::tx::{FeatureTarget, Op, Transaction};

#[derive(Clone, Debug)]
pub(crate) struct LayerEntry {
    pub(crate) key: LayerKey,
    pub(crate) json: Arc<Json>,
}

#[derive(Clone, Debug)]
pub(crate) struct SourceEntry {
    pub(crate) key: SourceKey,
    pub(crate) id: String,
    pub(crate) json: Arc<Json>,
}

/// A v8 style as JSON and as its typed IR, edited by transactions.
#[derive(Clone, Debug)]
pub struct StyleDoc {
    version: u64,
    /// Root members in document order; `sources` and `layers` are kept
    /// apart and written back in place.
    pub(crate) root: Arc<Vec<Member>>,
    pub(crate) sources: Vec<SourceEntry>,
    pub(crate) layers: Vec<LayerEntry>,
    /// Global state set by transactions, over the root `state` defaults.
    pub(crate) overrides: Vec<(String, Value<'static>)>,
    style: Arc<Scene>,
    next_key: u32,
    /// What the scene document sets over the style.
    decoration: Arc<Decoration>,
}

/// What a scene document sets over the style it extends (plan §11.1, S0):
/// it decorates the style's v8 meaning, never edits it, so every snapshot
/// of the style gets it again.
#[derive(Clone, Debug, Default)]
struct Decoration {
    projection: Option<orb_crs::Projection>,
    globe: Option<orb_crs::Globe>,
    tile_matrix_sets: Vec<(String, orb_crs::TileMatrixSet)>,
    lights: Option<Lights>,
    atmosphere: Option<Atmosphere>,
    symbol_occlusion: bool,
}

impl Decoration {
    fn apply(&self, mut scene: Scene) -> Scene {
        scene.projection = self.projection;
        scene.symbol_occlusion = self.symbol_occlusion;
        scene.globe = self.globe.or(scene.globe);
        scene.tile_matrix_sets = self.tile_matrix_sets.clone();
        if let Some(lights) = &self.lights {
            scene.lights = lights.clone();
        }
        if let Some(atmosphere) = &self.atmosphere {
            scene.atmosphere = Some(atmosphere.clone());
        }
        scene
    }
}

/// The working copy a transaction edits.
#[derive(Clone, Debug)]
pub(crate) struct Draft {
    pub(crate) root: Arc<Vec<Member>>,
    pub(crate) sources: Vec<SourceEntry>,
    pub(crate) layers: Vec<LayerEntry>,
    pub(crate) overrides: Vec<(String, Value<'static>)>,
    /// Layers whose JSON changed.
    pub(crate) dirty: Vec<LayerKey>,
    /// Sources given new data.
    pub(crate) new_data: Vec<SourceKey>,
    /// Sources whose data changed by diffs alone, with the diffs.
    pub(crate) diffs: Vec<(SourceKey, Vec<Arc<Json>>)>,
    /// The root (other than sources and layers) was replaced.
    pub(crate) new_root: bool,
    pub(crate) feature_state: Vec<FeatureStateOp>,
    next_key: u32,
}

pub(crate) fn error(message: impl Into<String>) -> Diagnostics {
    Diagnostics {
        errors: vec![Diagnostic {
            path: String::new(),
            span: Span::default(),
            message: message.into(),
        }],
        warnings: Vec::new(),
    }
}

/// A JSON value without a location.
pub(crate) fn synthetic(value: JsonValue) -> Json {
    Json::new(value, Span::default())
}

/// The value of `key` in an object's members.
pub(crate) fn member<'a>(members: &'a [Member], key: &str) -> Option<&'a Json> {
    members
        .iter()
        .rev()
        .find(|m| m.key == key)
        .map(|m| &m.value)
}

/// Sets or (with `None`) removes `key` among `members`.
pub(crate) fn set_member(members: &mut Vec<Member>, key: &str, value: Option<Json>) {
    members.retain(|m| m.key != key);
    if let Some(value) = value {
        members.push(Member {
            key: key.into(),
            key_span: Span::default(),
            value,
        });
    }
}

fn layer_id(json: &Json) -> Option<&str> {
    json.get("id").and_then(Json::as_str)
}

impl StyleDoc {
    /// Parses a v8 document.
    pub fn parse(text: &str) -> Result<Self, Diagnostics> {
        let style = crate::parse(text)?;
        let root = json::parse(text).map_err(|e| error(e.to_string()))?;
        Ok(Self::from_parts(style, &root))
    }

    /// A v8 document as a scene document extends it (plan §11.1, S0):
    /// with the scene's `models` (resolved URLs), the paint it sets on
    /// the style's layers and its added layers, in
    /// the scene dialect, under the scene's projection, tile matrix sets,
    /// low-zoom surface, lights and atmosphere.
    pub fn parse_extended(text: &str, scene: &crate::SceneDocument) -> Result<Self, Diagnostics> {
        let (models, added) = (&scene.models, &scene.layers);
        let root = json::parse(text).map_err(|e| error(e.to_string()))?;
        let mut members = root
            .as_object()
            .ok_or_else(|| error("a style must be an object"))?
            .to_vec();
        let mut layers: Vec<Json> = member(&members, "layers")
            .and_then(Json::as_array)
            .unwrap_or_default()
            .to_vec();
        for layer in added {
            let at = |id: &str| {
                layers
                    .iter()
                    .position(|l| layer_id(l) == Some(id))
                    .ok_or_else(|| error(format!("layers+: the layer \"{id}\" does not exist")))
            };
            let index = match (&layer.before, &layer.after) {
                (Some(before), _) => at(before)?,
                (_, Some(after)) => at(after)? + 1,
                _ => layers.len(),
            };
            layers.insert(index, layer.json.clone());
        }
        for (id, paint) in &scene.paint {
            let layer = layers
                .iter_mut()
                .find(|l| layer_id(l) == Some(id))
                .ok_or_else(|| error(format!("layers.{id}: the layer \"{id}\" does not exist")))?;
            let mut members = layer.as_object().unwrap_or_default().to_vec();
            let mut own = member(&members, "paint")
                .and_then(Json::as_object)
                .unwrap_or_default()
                .to_vec();
            for m in paint.as_object().unwrap_or_default() {
                set_member(&mut own, &m.key, Some(m.value.clone()));
            }
            set_member(
                &mut members,
                "paint",
                Some(synthetic(JsonValue::Object(own))),
            );
            *layer = synthetic(JsonValue::Object(members));
        }
        set_member(
            &mut members,
            "layers",
            Some(synthetic(JsonValue::Array(layers))),
        );
        if !models.is_empty() {
            let object = models
                .iter()
                .map(|(id, url)| Member {
                    key: id.clone(),
                    key_span: Span::default(),
                    value: synthetic(JsonValue::String(url.clone())),
                })
                .collect();
            set_member(
                &mut members,
                "models",
                Some(synthetic(JsonValue::Object(object))),
            );
        }
        let merged = synthetic(JsonValue::Object(members));
        let style = crate::parse_dialect(&merged.to_string(), &[], Dialect::Scene)?;
        let mut doc = Self::from_parts(style, &merged);
        doc.decoration = Arc::new(Decoration {
            projection: scene.projection,
            globe: scene.globe,
            tile_matrix_sets: scene.sources.clone(),
            lights: scene.lights.clone(),
            atmosphere: scene.atmosphere.clone(),
            symbol_occlusion: scene.symbol_occlusion,
        });
        doc.style = Arc::new(doc.decoration.apply((*doc.style).clone()));
        Ok(doc)
    }

    fn from_parts(style: Scene, root: &Json) -> Self {
        let mut next_key = 0;
        let mut key = || {
            next_key += 1;
            next_key
        };
        let members = root.as_object().expect("parsed styles are objects");
        let sources = member(members, "sources")
            .and_then(Json::as_object)
            .unwrap_or_default()
            .iter()
            .map(|m| SourceEntry {
                key: SourceKey(key()),
                id: m.key.clone(),
                json: Arc::new(m.value.clone()),
            })
            .collect();
        let layers = member(members, "layers")
            .and_then(Json::as_array)
            .unwrap_or_default()
            .iter()
            .map(|l| LayerEntry {
                key: LayerKey(key()),
                json: Arc::new(l.clone()),
            })
            .collect();
        let root = members
            .iter()
            .filter(|m| m.key != "sources" && m.key != "layers")
            .cloned()
            .collect();
        Self {
            version: 1,
            root: Arc::new(root),
            sources,
            layers,
            overrides: Vec::new(),
            style: Arc::new(style),
            next_key,
            decoration: Arc::default(),
        }
    }

    /// Counts applied transactions; the first document is version 1.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// The current snapshot.
    pub fn style(&self) -> &Arc<Scene> {
        &self.style
    }

    /// Keys of [`style`](Self::style)'s layers, in order.
    pub fn layer_keys(&self) -> impl ExactSizeIterator<Item = LayerKey> + '_ {
        self.layers.iter().map(|l| l.key)
    }

    /// Keys of the style's sources, in order.
    pub fn source_keys(&self) -> impl ExactSizeIterator<Item = SourceKey> + '_ {
        self.sources.iter().map(|s| s.key)
    }

    pub fn source_index(&self, key: SourceKey) -> Option<usize> {
        self.sources.iter().position(|s| s.key == key)
    }

    /// The key of the source named `id`.
    pub fn source_key(&self, id: &str) -> Option<SourceKey> {
        self.sources.iter().find(|s| s.id == id).map(|s| s.key)
    }

    /// The JSON of the layer at `index`.
    pub fn layer_json(&self, index: usize) -> &Json {
        &self.layers[index].json
    }

    pub fn source_json(&self, index: usize) -> &Json {
        &self.sources[index].json
    }

    /// The layers' ids, in drawing order.
    pub fn layer_ids(&self) -> impl Iterator<Item = &str> {
        self.layers.iter().filter_map(|l| layer_id(&l.json))
    }

    pub fn source_ids(&self) -> impl Iterator<Item = &str> {
        self.sources.iter().map(|s| s.id.as_str())
    }

    /// Layer `id` as the document has it, its paint and layout properties
    /// included (GL JS `getLayer`).
    pub fn layer(&self, id: &str) -> Option<&Json> {
        self.layers
            .iter()
            .map(|l| &*l.json)
            .find(|json| layer_id(json) == Some(id))
    }

    /// Source `id` as the document has it (GL JS `getSource`).
    pub fn source(&self, id: &str) -> Option<&Json> {
        self.sources.iter().find(|s| s.id == id).map(|s| &*s.json)
    }

    /// The global state: root defaults with transactions' values over them.
    pub fn global_state(&self) -> Vec<(String, Value<'static>)> {
        crate::resolve_state(&self.style.state, &self.overrides)
    }

    /// The document as v8 JSON, with the global state written as its
    /// defaults, so that loading it fresh gives this snapshot.
    pub fn to_json(&self) -> String {
        let mut members: Vec<Member> = self.root.as_ref().clone();
        let state = self.global_state();
        if !state.is_empty() {
            let value = synthetic(JsonValue::Object(
                state
                    .iter()
                    .map(|(k, v)| Member {
                        key: k.clone(),
                        key_span: Span::default(),
                        value: synthetic(JsonValue::Object(vec![Member {
                            key: "default".into(),
                            key_span: Span::default(),
                            value: value_json(v),
                        }])),
                    })
                    .collect(),
            ));
            set_member(&mut members, "state", Some(value));
        }
        members.push(Member {
            key: "sources".into(),
            key_span: Span::default(),
            value: synthetic(JsonValue::Object(
                self.sources
                    .iter()
                    .map(|s| Member {
                        key: s.id.clone(),
                        key_span: Span::default(),
                        value: (*s.json).clone(),
                    })
                    .collect(),
            )),
        });
        members.push(Member {
            key: "layers".into(),
            key_span: Span::default(),
            value: synthetic(JsonValue::Array(
                self.layers.iter().map(|l| (*l.json).clone()).collect(),
            )),
        });
        synthetic(JsonValue::Object(members)).to_string()
    }

    /// Applies a transaction: every edit or, on the first error, none.
    /// Returns what changed, classified for the renderer.
    pub fn apply(&mut self, tx: &Transaction) -> Result<Patch, Diagnostics> {
        let mut draft = Draft {
            root: Arc::clone(&self.root),
            sources: self.sources.clone(),
            layers: self.layers.clone(),
            overrides: self.overrides.clone(),
            dirty: Vec::new(),
            new_data: Vec::new(),
            diffs: Vec::new(),
            new_root: false,
            feature_state: Vec::new(),
            next_key: self.next_key,
        };
        for op in &tx.ops {
            draft.apply(op).map_err(error)?;
        }
        if tx.only_feature_state() {
            return Ok(Patch {
                version: self.version,
                feature_state: draft.feature_state,
                ..Patch::default()
            });
        }
        let (style, mut patch) = diff::diff(self, &draft)?;
        for op in &tx.ops {
            if let Op::SetPlane(projection) = op {
                Arc::make_mut(&mut self.decoration).projection = Some(**projection);
                patch.changes.push(Change::Restyle);
            }
        }
        self.version += 1;
        patch.version = self.version;
        patch.feature_state = std::mem::take(&mut draft.feature_state);
        self.root = draft.root;
        self.sources = draft.sources;
        self.layers = draft.layers;
        self.overrides = draft.overrides;
        self.next_key = draft.next_key;
        self.style = Arc::new(self.decoration.apply(style));
        Ok(patch)
    }
}

/// A value as JSON.
fn value_json(value: &Value<'_>) -> Json {
    synthetic(match value {
        Value::Null => JsonValue::Null,
        Value::Bool(b) => JsonValue::Bool(*b),
        Value::Number(n) => JsonValue::Number(*n),
        Value::String(s) => JsonValue::String(s.to_string()),
        Value::Array(items) => JsonValue::Array(items.iter().map(value_json).collect()),
        Value::Object(members) => JsonValue::Object(
            members
                .iter()
                .map(|(k, v)| Member {
                    key: k.clone(),
                    key_span: Span::default(),
                    value: value_json(v),
                })
                .collect(),
        ),
        // Colours and other typed values don't come from JSON state.
        other => JsonValue::String(other.to_string()),
    })
}

impl Draft {
    fn key(&mut self) -> u32 {
        self.next_key += 1;
        self.next_key
    }

    fn layer(&self, id: &str) -> Result<usize, String> {
        self.layers
            .iter()
            .position(|l| layer_id(&l.json) == Some(id))
            .ok_or_else(|| format!("the layer \"{id}\" does not exist"))
    }

    fn source(&self, id: &str) -> Result<&SourceEntry, String> {
        self.sources
            .iter()
            .find(|s| s.id == id)
            .ok_or_else(|| format!("the source \"{id}\" does not exist"))
    }

    /// Edits the members of layer `id`.
    fn edit_layer(&mut self, id: &str, edit: impl FnOnce(&mut Vec<Member>)) -> Result<(), String> {
        let index = self.layer(id)?;
        let entry = &mut self.layers[index];
        let mut members = entry.json.as_object().expect("layers are objects").to_vec();
        edit(&mut members);
        entry.json = Arc::new(synthetic(JsonValue::Object(members)));
        if !self.dirty.contains(&entry.key) {
            self.dirty.push(entry.key);
        }
        Ok(())
    }

    /// Edits one property of a layer's `paint` or `layout` object.
    fn set_property(
        &mut self,
        layer: &str,
        section: &str,
        name: &str,
        value: Option<Json>,
    ) -> Result<(), String> {
        self.edit_layer(layer, |members| {
            let mut properties = member(members, section)
                .and_then(Json::as_object)
                .unwrap_or_default()
                .to_vec();
            set_member(&mut properties, name, value);
            let object = (!properties.is_empty()).then(|| synthetic(JsonValue::Object(properties)));
            set_member(members, section, object);
        })
    }

    fn feature_target(
        &self,
        target: &FeatureTarget,
    ) -> Result<(crate::SourceKey, Option<String>), String> {
        let source = self.source(&target.source)?;
        let vector = source.json.get("type").and_then(Json::as_str) == Some("vector");
        if vector && target.source_layer.is_none() {
            return Err(
                "the sourceLayer parameter must be provided for vector source types".into(),
            );
        }
        Ok((source.key, target.source_layer.clone()))
    }

    pub(crate) fn apply(&mut self, op: &Op) -> Result<(), String> {
        match op {
            // The document applies it, over the style.
            Op::SetPlane(_) => Ok(()),
            Op::SetPaint { layer, name, value } => {
                self.set_property(layer, "paint", name, value.clone())
            }
            Op::SetLayout { layer, name, value } => {
                self.set_property(layer, "layout", name, value.clone())
            }
            Op::SetFilter { layer, filter } => {
                self.edit_layer(layer, |m| set_member(m, "filter", filter.clone()))
            }
            Op::SetZoomRange { layer, min, max } => self.edit_layer(layer, |m| {
                let number = |z: &Option<f64>| z.map(|z| synthetic(JsonValue::Number(z)));
                set_member(m, "minzoom", number(min));
                set_member(m, "maxzoom", number(max));
            }),
            Op::AddLayer { layer, before } => {
                let id = layer_id(layer).ok_or("a layer needs an id")?;
                if self.layer(id).is_ok() {
                    return Err(format!("the layer \"{id}\" already exists"));
                }
                let index = match before {
                    Some(before) => self.layer(before)?,
                    None => self.layers.len(),
                };
                let key = LayerKey(self.key());
                self.layers.insert(
                    index,
                    LayerEntry {
                        key,
                        json: Arc::new(layer.clone()),
                    },
                );
                Ok(())
            }
            Op::RemoveLayer { id } => {
                let index = self.layer(id)?;
                self.layers.remove(index);
                Ok(())
            }
            Op::MoveLayer { id, before } => {
                let entry = self.layers.remove(self.layer(id)?);
                let index = match before {
                    Some(before) => self.layer(before)?,
                    None => self.layers.len(),
                };
                self.layers.insert(index, entry);
                Ok(())
            }
            Op::AddSource { id, source } => {
                if self.source(id).is_ok() {
                    return Err(format!("the source \"{id}\" already exists"));
                }
                let key = SourceKey(self.key());
                self.sources.push(SourceEntry {
                    key,
                    id: id.clone(),
                    json: Arc::new(source.clone()),
                });
                Ok(())
            }
            Op::RemoveSource { id } => {
                self.source(id)?;
                if let Some(layer) = self
                    .layers
                    .iter()
                    .find(|l| l.json.get("source").and_then(Json::as_str) == Some(id))
                {
                    return Err(format!(
                        "the source \"{id}\" cannot be removed while the layer \"{}\" uses it",
                        layer_id(&layer.json).unwrap_or_default()
                    ));
                }
                self.sources.retain(|s| s.id != *id);
                Ok(())
            }
            Op::SetSourceData { id, data } => {
                let entry = self
                    .sources
                    .iter_mut()
                    .find(|s| s.id == *id)
                    .ok_or_else(|| format!("the source \"{id}\" does not exist"))?;
                if entry.json.get("type").and_then(Json::as_str) != Some("geojson") {
                    return Err(format!("the source \"{id}\" is not a GeoJSON source"));
                }
                let mut members = entry
                    .json
                    .as_object()
                    .expect("sources are objects")
                    .to_vec();
                set_member(&mut members, "data", Some(data.clone()));
                entry.json = Arc::new(synthetic(JsonValue::Object(members)));
                if !self.new_data.contains(&entry.key) {
                    self.new_data.push(entry.key);
                }
                Ok(())
            }
            Op::UpdateSourceData { id, diff } => {
                let entry = self
                    .sources
                    .iter_mut()
                    .find(|s| s.id == *id)
                    .ok_or_else(|| format!("the source \"{id}\" does not exist"))?;
                if entry.json.get("type").and_then(Json::as_str) != Some("geojson") {
                    return Err(format!("the source \"{id}\" is not a GeoJSON source"));
                }
                let data = entry
                    .json
                    .get("data")
                    .filter(|d| d.as_object().is_some())
                    .ok_or_else(|| format!("the source \"{id}\" has no inline data to update"))?;
                let promote_id = match entry.json.get("promoteId") {
                    Some(p) => p
                        .as_str()
                        .or_else(|| p.get(orb_tile::geojson::LAYER_NAME).and_then(Json::as_str)),
                    None => None,
                };
                let parsed = crate::geojson::Diff::parse(diff)?;
                let data = parsed.apply_json(data, promote_id)?;
                let mut members = entry
                    .json
                    .as_object()
                    .expect("sources are objects")
                    .to_vec();
                set_member(&mut members, "data", Some(data));
                entry.json = Arc::new(synthetic(JsonValue::Object(members)));
                let key = entry.key;
                match self.diffs.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, diffs)) => diffs.push(Arc::new(diff.clone())),
                    None => self.diffs.push((key, vec![Arc::new(diff.clone())])),
                }
                Ok(())
            }
            Op::SetGlobalState { key, value } => {
                self.overrides.retain(|(k, _)| k != key);
                if let Some(value) = value.as_ref().filter(|v| **v != Value::Null) {
                    self.overrides.push((key.clone(), value.clone()));
                }
                Ok(())
            }
            Op::SetFeatureState { target, state } => {
                let (source, source_layer) = self.feature_target(target)?;
                self.feature_state.push(FeatureStateOp::Set {
                    source,
                    source_layer,
                    id: target.id.clone().ok_or("the feature id is required")?,
                    state: state.clone(),
                });
                Ok(())
            }
            Op::RemoveFeatureState { target, key } => {
                let (source, source_layer) = self.feature_target(target)?;
                if key.is_some() && target.id.is_none() {
                    return Err("a state key can only be removed from one feature".into());
                }
                self.feature_state.push(FeatureStateOp::Remove {
                    source,
                    source_layer,
                    id: target.id.clone(),
                    key: key.clone(),
                });
                Ok(())
            }
            Op::SetStyle { style } => self.replace(style),
            Op::SetRoot { key, value } => {
                let mut root: Vec<Member> = self.root.as_ref().clone();
                root.retain(|m| m.key != *key);
                if let Some(value) = value {
                    root.push(Member {
                        key: key.clone(),
                        key_span: Default::default(),
                        value: value.clone(),
                    });
                }
                self.root = Arc::new(root);
                self.new_root = true;
                Ok(())
            }
        }
    }

    /// GL JS's `setStyle` with `diff: true`: the new document, keeping the
    /// keys of sources whose definitions are unchanged and of layers of the
    /// same type drawing the same source data.
    fn replace(&mut self, style: &Json) -> Result<(), String> {
        let members = style.as_object().ok_or("a style must be an object")?;
        let mut old_sources: HashMap<String, (SourceKey, Arc<Json>)> = self
            .sources
            .iter()
            .map(|s| (s.id.clone(), (s.key, Arc::clone(&s.json))))
            .collect();
        let new_sources: Vec<(String, Json)> = member(members, "sources")
            .and_then(Json::as_object)
            .unwrap_or_default()
            .iter()
            .map(|m| (m.key.clone(), m.value.clone()))
            .collect();
        let mut sources = Vec::with_capacity(new_sources.len());
        for (id, json) in new_sources {
            let kept = old_sources
                .remove(&id)
                .filter(|(_, old)| same(old, &json))
                .map(|(key, _)| key);
            let key = match kept {
                Some(key) => key,
                None => SourceKey(self.key()),
            };
            sources.push(SourceEntry {
                key,
                id,
                json: Arc::new(json),
            });
        }
        let source_key = |sources: &[SourceEntry], json: &Json| {
            let id = json.get("source").and_then(Json::as_str)?;
            sources.iter().find(|s| s.id == id).map(|s| s.key)
        };
        let identity = |json: &Json| {
            ["type", "source", "source-layer"].map(|k| json.get(k).map(ToString::to_string))
        };
        let mut layers = Vec::new();
        for json in member(members, "layers")
            .and_then(Json::as_array)
            .unwrap_or_default()
        {
            let old = layer_id(json).and_then(|id| self.layer(id).ok());
            let kept = old.map(|i| &self.layers[i]).filter(|old| {
                identity(&old.json) == identity(json)
                    && source_key(&self.sources, &old.json) == source_key(&sources, json)
            });
            let key = match kept {
                Some(old) => {
                    if !same(&old.json, json) {
                        self.dirty.push(old.key);
                    }
                    old.key
                }
                None => LayerKey(self.key()),
            };
            layers.push(LayerEntry {
                key,
                json: Arc::new(json.clone()),
            });
        }
        let root: Vec<Member> = members
            .iter()
            .filter(|m| m.key != "sources" && m.key != "layers")
            .cloned()
            .collect();
        self.new_root = true;
        self.root = Arc::new(root);
        self.sources = sources;
        self.layers = layers;
        self.overrides.clear();
        Ok(())
    }
}

/// JSON equality ignoring spans: two documents mean the same value.
pub(crate) fn same(a: &Json, b: &Json) -> bool {
    match (&a.value, &b.value) {
        (JsonValue::Array(a), JsonValue::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same(a, b))
        }
        (JsonValue::Object(a), JsonValue::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(a, b)| a.key == b.key && same(&a.value, &b.value))
        }
        (a, b) => a == b,
    }
}

/// [`same`] for optional members, an absent value matching only another.
pub(crate) fn same_opt(a: Option<&Json>, b: Option<&Json>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => same(a, b),
        (None, None) => true,
        _ => false,
    }
}
