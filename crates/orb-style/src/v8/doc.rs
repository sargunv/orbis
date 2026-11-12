//! The persistent document.

use crate::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::v8::{Diagnostic, Diagnostics};
use orb_expr::json::{Json, JsonValue, Member, Span};

use crate::v8::diff;
use crate::v8::tx::{Op, Transaction};

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
    style: Arc<Style>,
    next_key: u32,
}

/// The working copy a transaction edits.
#[derive(Clone, Debug)]
pub(crate) struct Draft {
    pub(crate) root: Arc<Vec<Member>>,
    pub(crate) sources: Vec<SourceEntry>,
    pub(crate) layers: Vec<LayerEntry>,
    pub(crate) overrides: Vec<(String, Value<'static>)>,
    /// Layers whose JSON changed, each once or more.
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

/// A feature-state op resolved against `sources`.
fn feature_state_op(sources: &[SourceEntry], op: &Op) -> Result<FeatureStateOp, String> {
    let (Op::SetFeatureState { target, .. } | Op::RemoveFeatureState { target, .. }) = op else {
        unreachable!("not a feature-state op: {op:?}");
    };
    let source = (sources.iter())
        .find(|s| s.id == target.source)
        .ok_or_else(|| format!("the source \"{}\" does not exist", target.source))?;
    let vector = source.json.get("type").and_then(Json::as_str) == Some("vector");
    if vector && target.source_layer.is_none() {
        return Err("the sourceLayer parameter must be provided for vector source types".into());
    }
    let (source, source_layer) = (source.key, target.source_layer.clone());
    Ok(match op {
        Op::SetFeatureState { state, .. } => FeatureStateOp::Set {
            source,
            source_layer,
            id: target.id.clone().ok_or("the feature id is required")?,
            state: state.clone(),
        },
        Op::RemoveFeatureState { key, .. } => {
            if key.is_some() && target.id.is_none() {
                return Err("a state key can only be removed from one feature".into());
            }
            FeatureStateOp::Remove {
                source,
                source_layer,
                id: target.id.clone(),
                key: key.clone(),
            }
        }
        _ => unreachable!(),
    })
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
    /// Parses a v8 document with no previews.
    pub fn parse(text: &str) -> Result<Self, Diagnostics> {
        Self::parse_with(text, Previews::NONE)
    }

    /// Parses a v8 document with `previews` enabled; its edits keep them.
    pub fn parse_with(text: &str, previews: Previews) -> Result<Self, Diagnostics> {
        Self::from_json(crate::v8::read(text)?, previews)
    }

    /// [`parse_with`](Self::parse_with) of a document already read
    /// ([`read`](crate::v8::read)).
    pub fn from_json(mut root: Json, previews: Previews) -> Result<Self, Diagnostics> {
        crate::v8::deref_layers(&mut root);
        let style = crate::v8::parse_json(&root, previews, &[])?;
        let JsonValue::Object(members) = root.value else {
            unreachable!("parsed styles are objects");
        };
        let mut next_key = 0;
        let mut key = || {
            next_key += 1;
            next_key
        };
        // The last of a repeated key is the one that counts.
        let (mut sources, mut layers, mut root) = (Vec::new(), Vec::new(), Vec::new());
        for m in members {
            match m.key.as_str() {
                "sources" => match m.value.value {
                    JsonValue::Object(members) => sources = members,
                    _ => sources.clear(),
                },
                "layers" => match m.value.value {
                    JsonValue::Array(items) => layers = items,
                    _ => layers.clear(),
                },
                _ => root.push(m),
            }
        }
        let sources = sources
            .into_iter()
            .map(|m| SourceEntry {
                key: SourceKey(key()),
                id: m.key,
                json: Arc::new(m.value),
            })
            .collect();
        let layers = layers
            .into_iter()
            .map(|json| LayerEntry {
                key: LayerKey(key()),
                json: Arc::new(json),
            })
            .collect();
        Ok(Self {
            version: 1,
            root: Arc::new(root),
            sources,
            layers,
            overrides: Vec::new(),
            style: Arc::new(style),
            next_key,
        })
    }

    /// Counts applied transactions; the first document is version 1.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// The current snapshot.
    pub fn style(&self) -> &Arc<Style> {
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

    /// Root member `key` as the document has it now, but `sources` and
    /// `layers`, which [`source`](Self::source) and [`layer`](Self::layer)
    /// read (GL JS `getLight`, `getSky`, `getProjection`, …).
    pub fn root(&self, key: &str) -> Option<&Json> {
        member(&self.root, key)
    }

    /// The global state: root defaults with transactions' values over them.
    pub fn global_state(&self) -> Vec<(String, Value<'static>)> {
        crate::v8::resolve_state(&self.style.state, &self.overrides)
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
        // State edits change no snapshot: resolve them without a draft,
        // whose copy of every source and layer entry a burst of per-feature
        // calls would pay each time.
        if tx.only_feature_state() {
            let feature_state = (tx.ops.iter())
                .map(|op| feature_state_op(&self.sources, op).map_err(error))
                .collect::<Result<_, _>>()?;
            return Ok(Patch {
                version: self.version,
                feature_state,
                ..Patch::default()
            });
        }
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
        let (style, mut patch) = diff::diff(self, &draft)?;
        self.version += 1;
        patch.version = self.version;
        patch.feature_state = std::mem::take(&mut draft.feature_state);
        self.root = draft.root;
        self.sources = draft.sources;
        self.layers = draft.layers;
        self.overrides = draft.overrides;
        self.next_key = draft.next_key;
        self.style = Arc::new(style);
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
        self.dirty.push(entry.key);
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

    pub(crate) fn apply(&mut self, op: &Op) -> Result<(), String> {
        match op {
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
                let parsed = crate::v8::geojson::Diff::parse(diff)?;
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
            Op::SetFeatureState { .. } | Op::RemoveFeatureState { .. } => {
                let op = feature_state_op(&self.sources, op)?;
                self.feature_state.push(op);
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
        let old_layers = std::mem::take(&mut self.layers);
        let mut by_id: HashMap<&str, &LayerEntry> = HashMap::new();
        for layer in &old_layers {
            if let Some(id) = layer_id(&layer.json) {
                by_id.entry(id).or_insert(layer);
            }
        }
        let mut layers = Vec::new();
        for json in member(members, "layers")
            .and_then(Json::as_array)
            .unwrap_or_default()
        {
            let old = layer_id(json).and_then(|id| by_id.get(id));
            let kept = old.filter(|old| {
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
