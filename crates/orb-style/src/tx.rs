//! Transactions: edits to apply to a [`StyleDoc`](crate::StyleDoc) as one.

use orb_expr::json::{self, Json, JsonValue};
use orb_v8::Value;

/// One edit. Layer and source ids are as in the style.
#[derive(Clone, Debug)]
pub enum Op {
    /// `None` unsets the property, back to its default.
    SetPaint {
        layer: String,
        name: String,
        value: Option<Json>,
    },
    SetLayout {
        layer: String,
        name: String,
        value: Option<Json>,
    },
    SetFilter {
        layer: String,
        filter: Option<Json>,
    },
    SetZoomRange {
        layer: String,
        min: Option<f64>,
        max: Option<f64>,
    },
    /// Inserts a layer before `before`, or on top.
    AddLayer {
        layer: Json,
        before: Option<String>,
    },
    RemoveLayer {
        id: String,
    },
    MoveLayer {
        id: String,
        before: Option<String>,
    },
    AddSource {
        id: String,
        source: Json,
    },
    RemoveSource {
        id: String,
    },
    /// Replaces a GeoJSON source's `data` (GL JS's `setData`): a new data
    /// revision of the same source, whose drawn tiles stay until the new
    /// data's replace them.
    SetSourceData {
        id: String,
        data: Json,
    },
    /// Sets a global state property; `None` (or `null`) resets it to the
    /// style's default.
    SetGlobalState {
        key: String,
        value: Option<Value<'static>>,
    },
    SetFeatureState {
        target: FeatureTarget,
        state: Vec<(String, Value<'static>)>,
    },
    /// Removes `key` of a feature's state, its whole state, or (without an
    /// id) the state of every feature of the source (layer).
    RemoveFeatureState {
        target: FeatureTarget,
        key: Option<String>,
    },
    /// Replaces the whole document, applied as the diff to the new one.
    SetStyle {
        style: Json,
    },
    /// Sets or removes a root property other than sources and layers:
    /// GL JS's `setTerrain`, `setSky`, `setLight` and `setTransition`.
    SetRoot {
        key: String,
        value: Option<Json>,
    },
}

/// The feature, or features, a feature-state edit addresses.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FeatureTarget {
    pub source: String,
    /// Required for vector sources.
    pub source_layer: Option<String>,
    /// As GL JS keys it: `String(id)`.
    pub id: Option<String>,
}

/// A feature-state edit, validated against the document's sources.
#[derive(Clone, Debug, PartialEq)]
pub enum FeatureStateOp {
    Set {
        source: crate::SourceKey,
        source_layer: Option<String>,
        id: String,
        state: Vec<(String, Value<'static>)>,
    },
    Remove {
        source: crate::SourceKey,
        source_layer: Option<String>,
        id: Option<String>,
        key: Option<String>,
    },
}

/// Edits applied together by [`StyleDoc::apply`](crate::StyleDoc::apply):
/// all of them or, if any fails, none.
#[derive(Clone, Debug, Default)]
pub struct Transaction {
    pub(crate) ops: Vec<Op>,
}

impl Transaction {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, op: Op) -> &mut Self {
        self.ops.push(op);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Whether every edit is to feature state, which needs no new snapshot.
    pub fn only_feature_state(&self) -> bool {
        self.ops.iter().all(|op| {
            matches!(
                op,
                Op::SetFeatureState { .. } | Op::RemoveFeatureState { .. }
            )
        })
    }

    /// Parses GL JS map calls, `[["setPaintProperty", "water", "fill-color",
    /// "blue"], …]`, as the render tests' `operations` and the web
    /// package's `edit` write them. An unknown or unsupported call is an
    /// error naming it.
    pub fn from_commands(text: &str) -> Result<Self, String> {
        let doc = json::parse(text).map_err(|e| e.to_string())?;
        let calls = doc.as_array().ok_or("commands must be an array")?;
        let mut tx = Self::new();
        for call in calls {
            tx.push(Op::from_command(call)?);
        }
        Ok(tx)
    }
}

impl Op {
    /// One GL JS map call as JSON: `[name, ...args]`.
    pub fn from_command(call: &Json) -> Result<Self, String> {
        let items = call.as_array().ok_or("a command must be an array")?;
        let name = items
            .first()
            .and_then(Json::as_str)
            .ok_or("a command starts with its name")?;
        let arg = |i: usize| items.get(i).filter(|j| !j.is_null());
        let string = |i: usize, what: &str| {
            arg(i)
                .and_then(Json::as_str)
                .map(str::to_string)
                .ok_or_else(|| format!("{name}: {what} must be a string"))
        };
        let number = |i: usize| arg(i).and_then(Json::as_f64);
        Ok(match name {
            "setPaintProperty" => Op::SetPaint {
                layer: string(1, "the layer")?,
                name: string(2, "the property")?,
                value: arg(3).cloned(),
            },
            "setLayoutProperty" => Op::SetLayout {
                layer: string(1, "the layer")?,
                name: string(2, "the property")?,
                value: arg(3).cloned(),
            },
            "setFilter" => Op::SetFilter {
                layer: string(1, "the layer")?,
                filter: arg(2).cloned(),
            },
            "setLayerZoomRange" => Op::SetZoomRange {
                layer: string(1, "the layer")?,
                min: number(2),
                max: number(3),
            },
            "addLayer" => {
                let layer = arg(1).cloned().ok_or("addLayer: no layer")?;
                if layer.get("source").is_some_and(|s| s.as_object().is_some()) {
                    return Err("addLayer: inline sources are not supported".into());
                }
                Op::AddLayer {
                    layer,
                    before: arg(2).and_then(Json::as_str).map(str::to_string),
                }
            }
            "removeLayer" => Op::RemoveLayer {
                id: string(1, "the layer")?,
            },
            "moveLayer" => Op::MoveLayer {
                id: string(1, "the layer")?,
                before: arg(2).and_then(Json::as_str).map(str::to_string),
            },
            "addSource" => Op::AddSource {
                id: string(1, "the source id")?,
                source: arg(2).cloned().ok_or("addSource: no source")?,
            },
            "removeSource" => Op::RemoveSource {
                id: string(1, "the source id")?,
            },
            "setSourceData" => Op::SetSourceData {
                id: string(1, "the source id")?,
                data: arg(2).cloned().ok_or("setSourceData: no data")?,
            },
            "setGlobalStateProperty" => Op::SetGlobalState {
                key: string(1, "the property")?,
                value: arg(2).map(Value::from_json),
            },
            "setFeatureState" => {
                let target = target(arg(1), name)?;
                if target.id.is_none() {
                    return Err(format!("{name}: the feature id is required"));
                }
                let state = arg(2)
                    .and_then(Json::as_object)
                    .ok_or_else(|| format!("{name}: the state must be an object"))?;
                Op::SetFeatureState {
                    target,
                    state: state
                        .iter()
                        .map(|m| (m.key.clone(), Value::from_json(&m.value)))
                        .collect(),
                }
            }
            "removeFeatureState" => Op::RemoveFeatureState {
                target: target(arg(1), name)?,
                key: arg(2).and_then(Json::as_str).map(str::to_string),
            },
            "setStyle" => match arg(1) {
                Some(style) if style.as_object().is_some() => Op::SetStyle {
                    style: style.clone(),
                },
                _ => return Err("setStyle: only style objects are supported".into()),
            },
            "setTerrain" => Op::SetRoot {
                key: "terrain".into(),
                value: arg(1).cloned(),
            },
            "setSky" => Op::SetRoot {
                key: "sky".into(),
                value: arg(1).cloned(),
            },
            "setProjection" => Op::SetRoot {
                key: "projection".into(),
                value: arg(1).cloned(),
            },
            "setLight" => Op::SetRoot {
                key: "light".into(),
                value: arg(1).cloned(),
            },
            "setTransition" => Op::SetRoot {
                key: "transition".into(),
                value: arg(1).cloned(),
            },
            other => return Err(format!("unsupported operation {other}")),
        })
    }
}

/// A feature-state target object: `{source, sourceLayer?, id?}`.
fn target(json: Option<&Json>, name: &str) -> Result<FeatureTarget, String> {
    let json = json.ok_or_else(|| format!("{name}: no feature"))?;
    let source = json
        .get("source")
        .and_then(Json::as_str)
        .ok_or_else(|| format!("{name}: the feature needs a source"))?;
    let id = match json.get("id").map(|j| &j.value) {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::String(s)) => Some(s.clone()),
        Some(JsonValue::Number(n)) => Some(orb_expr::js_number_string(*n)),
        Some(_) => return Err(format!("{name}: the id must be a number or string")),
    };
    Ok(FeatureTarget {
        source: source.into(),
        source_layer: json
            .get("sourceLayer")
            .and_then(Json::as_str)
            .map(str::to_string),
        id,
    })
}
