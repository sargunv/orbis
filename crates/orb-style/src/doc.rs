//! The persistent document.

use std::collections::HashMap;
use std::sync::Arc;

use orb_expr::json::{self, Json, JsonValue, Member, Span};
use orb_v8::{Diagnostic, Diagnostics, Style, Value};

use crate::diff::{self, Patch};
use crate::tx::{FeatureStateOp, FeatureTarget, Op, Transaction};

/// A layer's identity across edits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerKey(pub u32);

/// A source's identity across edits; a source whose definition changes is
/// a new source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceKey(pub u32);

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
    /// Layers whose JSON changed.
    pub(crate) dirty: Vec<LayerKey>,
    /// The root (other than sources and layers) was replaced.
    pub(crate) new_root: bool,
    pub(crate) feature_state: Vec<FeatureStateOp>,
    next_key: u32,
}

fn error(message: impl Into<String>) -> Diagnostics {
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
fn set_member(members: &mut Vec<Member>, key: &str, value: Option<Json>) {
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
        let style = orb_v8::parse(text)?;
        let root = json::parse(text).map_err(|e| error(e.to_string()))?;
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

    /// The layer with `key`: its index in the style.
    pub fn layer_index(&self, key: LayerKey) -> Option<usize> {
        self.layers.iter().position(|l| l.key == key)
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

    /// The global state: root defaults with transactions' values over them.
    pub fn global_state(&self) -> Vec<(String, Value<'static>)> {
        orb_v8::resolve_state(&self.style.state, &self.overrides)
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
                .filter(|(_, old)| crate::same(old, &json))
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
                    if !crate::same(&old.json, json) {
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
