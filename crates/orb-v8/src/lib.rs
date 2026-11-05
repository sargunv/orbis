//! The front end of the native scene (plan §11.1, S0–S1): parses and
//! validates v8 styles and the scene documents that extend them into an
//! `orb_style::Scene`, with errors and warnings located by JSON path
//! (`layers[3].paint.line-width`) and byte span; keeps the documents of
//! record ([`StyleDoc`]) and applies GL JS-style edits to them as
//! transactions, diffed into classified changes. Defaults and coercions
//! follow GL JS.

#![forbid(unsafe_code)]

mod diff;
mod doc;
mod font_faces;
mod geojson;
mod layer;
mod light;
mod lights;
mod paint;
mod scene;
mod sky;
mod source;
mod sprite;
mod terrain;
mod tx;

use std::collections::HashSet;
use std::fmt;

pub use doc::StyleDoc;
pub use geojson::parse_geojson;
pub use light::default_lights;
pub use lights::parse_time;
pub use orb_expr::json::Span;
pub use orb_expr::{Diagnostic, Filter};
pub use scene::{AddedLayer, SceneDocument, StyleRef, is_scene, parse_scene, tms_definition};
pub use tx::{FeatureTarget, Op, Transaction};

// The native model this front end builds.
use orb_style::*;

use orb_expr::json::{self, Json, JsonValue, Member};

/// Everything a failed parse found: any error rejects the style, as GL JS's
/// validation does.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Diagnostics {
    pub errors: Vec<Diagnostic>,
    pub warnings: Vec<Diagnostic>,
}

impl fmt::Display for Diagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, e) in self.errors.iter().enumerate() {
            if i > 0 {
                f.write_str("\n")?;
            }
            write!(f, "{e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Diagnostics {}

/// Parses and validates a v8 style document, its `global-state` reads
/// resolved to the root `state` defaults.
pub fn parse(text: &str) -> Result<Scene, Diagnostics> {
    parse_with_state(text, &[])
}

/// Parses a style whose global state is its root `state` defaults
/// overridden by `overrides` (plan §11.2, S2): `global-state` reads fold to
/// those values, so they classify as constants where they are the only
/// input.
pub fn parse_with_state(
    text: &str,
    overrides: &[(String, Value<'static>)],
) -> Result<Scene, Diagnostics> {
    parse_dialect(text, overrides, Dialect::V8)
}

/// [`parse_with_state`] in `dialect`.
pub fn parse_dialect(
    text: &str,
    overrides: &[(String, Value<'static>)],
    dialect: Dialect,
) -> Result<Scene, Diagnostics> {
    let mut cx = Cx {
        scene: dialect == Dialect::Scene,
        ..Cx::default()
    };
    let doc = match json::parse(text) {
        Ok(doc) => doc,
        Err(e) => {
            cx.error(
                "",
                Span {
                    start: e.offset,
                    end: e.offset,
                },
                e.message,
            );
            return Err(cx.diagnostics);
        }
    };
    let style = parse_root(&mut cx, &doc, overrides);
    match style {
        Some(mut style) if cx.diagnostics.errors.is_empty() => {
            style.warnings = cx.diagnostics.warnings;
            Ok(style)
        }
        _ => Err(cx.diagnostics),
    }
}

/// Root keys v8 defines that Orbis does not model yet.
const UNMODELED_ROOT: &[&str] = &["centerAltitude"];

fn parse_root(cx: &mut Cx, doc: &Json, overrides: &[(String, Value<'static>)]) -> Option<Scene> {
    let root = cx.object(doc, "")?;
    for m in root {
        match m.key.as_str() {
            "version" | "name" | "metadata" | "center" | "zoom" | "bearing" | "pitch" | "roll"
            | "sources" | "layers" | "sprite" | "glyphs" | "font-faces" | "light" | "state"
            | "sky" | "terrain" | "projection" => {}
            "models" if cx.scene => {}
            "transition" => {}
            key if UNMODELED_ROOT.contains(&key) => {
                cx.warn(
                    key,
                    m.key_span,
                    format!("\"{key}\" is not supported yet and is ignored"),
                );
            }
            key => cx.warn(key, m.key_span, format!("unknown property \"{key}\"")),
        }
    }
    match doc.get("version") {
        None => cx.error("", doc.span, "missing required property \"version\""),
        Some(v) if v.as_f64() == Some(8.0) => {}
        Some(v) => cx.error("version", v.span, format!("expected one of [8], {v} found")),
    }
    let name = doc.get("name").and_then(|n| cx.string(n, "name"));
    let center = doc.get("center").and_then(|c| cx.numbers::<2>(c, "center"));
    let zoom = doc.get("zoom").and_then(|z| cx.number(z, "zoom"));
    let bearing = doc
        .get("bearing")
        .and_then(|b| cx.number(b, "bearing"))
        .unwrap_or(0.0);
    let pitch = doc
        .get("pitch")
        .and_then(|p| cx.number(p, "pitch"))
        .unwrap_or(0.0);
    let roll = doc
        .get("roll")
        .and_then(|r| cx.number(r, "roll"))
        .unwrap_or(0.0);
    let transition = match doc.get("transition") {
        None => Transition::default(),
        Some(json) => paint::parse_transition(json)
            .map(|t| t.over(Transition::default()))
            .unwrap_or_else(|| {
                cx.warn(
                    "transition",
                    json.span,
                    "expected {\"duration\", \"delay\"} in milliseconds",
                );
                Transition::default()
            }),
    };

    let sprites = sprite::parse(cx, doc.get("sprite"));
    let glyphs = doc.get("glyphs").and_then(|g| cx.string(g, "glyphs"));
    if let Some(url) = &glyphs {
        for token in ["{fontstack}", "{range}"] {
            if !url.contains(token) {
                let span = doc.get("glyphs").expect("present").span;
                cx.error(
                    "glyphs",
                    span,
                    format!("\"glyphs\" url must include a \"{token}\" token"),
                );
            }
        }
    }
    let state = doc
        .get("state")
        .map(|s| parse_state(cx, s))
        .unwrap_or_default();
    cx.state = resolve_state(&state, overrides);
    let sources = match doc.get("sources") {
        None => {
            cx.error("", doc.span, "missing required property \"sources\"");
            Vec::new()
        }
        Some(sources) => source::parse_sources(cx, sources),
    };
    let layers = match doc.get("layers") {
        None => {
            cx.error("", doc.span, "missing required property \"layers\"");
            Vec::new()
        }
        Some(layers) => match cx.array(layers, "layers") {
            None => Vec::new(),
            Some(items) => {
                let mut ids = HashSet::new();
                items
                    .iter()
                    .enumerate()
                    .filter_map(|(i, l)| layer::parse_layer(cx, l, i, &sources, &mut ids))
                    .collect()
            }
        },
    };
    let models = match doc.get("models").filter(|_| cx.scene) {
        None => Vec::new(),
        Some(models) => cx
            .object(models, "models")
            .unwrap_or_default()
            .iter()
            .filter_map(|m| {
                let url = cx.string(&m.value, &format!("models.{}", m.key))?;
                Some((m.key.clone(), url))
            })
            .collect(),
    };
    Some(Scene {
        dialect: if cx.scene {
            Dialect::Scene
        } else {
            Dialect::V8
        },
        models,
        name,
        center,
        zoom,
        bearing,
        pitch,
        roll,
        transition,
        sprites,
        glyphs,
        font_faces: font_faces::parse(cx, doc.get("font-faces")),
        lights: light::parse(cx, doc.get("light")).lower(),
        globe: doc.get("projection").and_then(|p| parse_globe(cx, p)),
        projection: None,
        tile_matrix_sets: Vec::new(),
        state,
        atmosphere: doc
            .get("sky")
            .map(|json| sky::parse(cx, json, sky::Declared::Sky)),
        terrain: doc
            .get("terrain")
            .and_then(|json| terrain::parse(cx, json, &sources)),
        sources,
        layers,
        warnings: Vec::new(),
    })
}

/// The root `state`: `{"name": {"default": value}}`.
fn parse_state(cx: &mut Cx, json: &Json) -> Vec<(String, Value<'static>)> {
    let Some(members) = cx.object(json, "state") else {
        return Vec::new();
    };
    let mut state = Vec::new();
    for m in members {
        let path = format!("state.{}", m.key);
        if cx.object(&m.value, &path).is_none() {
            continue;
        }
        let Some(default) = m.value.get("default") else {
            cx.error(&path, m.value.span, "missing required property \"default\"");
            continue;
        };
        state.push((m.key.clone(), Value::from_json(default)));
    }
    state
}

/// Global state: the defaults, each replaced by an override of the same
/// name; overrides without a default are added.
pub fn resolve_state(
    defaults: &[(String, Value<'static>)],
    overrides: &[(String, Value<'static>)],
) -> Vec<(String, Value<'static>)> {
    let mut state = defaults.to_vec();
    for (key, value) in overrides {
        match state.iter_mut().find(|(k, _)| k == key) {
            Some((_, v)) => *v = value.clone(),
            None => state.push((key.clone(), value.clone())),
        }
    }
    state
}

/// Parses one layer of a style whose sources are `sources`, with
/// `global-state` reads resolved to `state`. `index` locates diagnostics
/// (`layers[index]`). Returns the layer and its warnings.
pub fn parse_layer(
    json: &Json,
    index: usize,
    sources: &[Source],
    state: &[(String, Value<'static>)],
    dialect: Dialect,
) -> Result<(Layer, Vec<Diagnostic>), Diagnostics> {
    let mut cx = Cx {
        state: state.to_vec(),
        scene: dialect == Dialect::Scene,
        ..Cx::default()
    };
    let layer = layer::parse_layer(&mut cx, json, index, sources, &mut HashSet::new());
    match layer {
        Some(layer) if cx.diagnostics.errors.is_empty() => Ok((layer, cx.diagnostics.warnings)),
        _ => Err(cx.diagnostics),
    }
}

/// The v8 spec of a modelled paint (`paint: true`) or layout property of a
/// `layer_type` layer.
pub fn property_spec(layer_type: &str, paint: bool, name: &str) -> Option<orb_expr::PropertySpec> {
    PropertyId::find(LayerType::from_name(layer_type)?, paint, name).map(PropertyId::spec)
}

/// Parses a root sky with the current global state, returning its warnings.
pub fn parse_sky(
    json: &Json,
    state: &[(String, Value<'static>)],
) -> Result<(Atmosphere, Vec<Diagnostic>), Diagnostics> {
    let mut cx = Cx {
        state: state.to_vec(),
        ..Cx::default()
    };
    let sky = sky::parse(&mut cx, json, sky::Declared::Sky);
    if cx.diagnostics.errors.is_empty() {
        Ok((sky, cx.diagnostics.warnings))
    } else {
        Err(cx.diagnostics)
    }
}

/// Parses a root `terrain` against the style's sources, returning its
/// warnings.
pub fn parse_terrain(
    json: &Json,
    sources: &[Source],
) -> Result<(Terrain, Vec<Diagnostic>), Diagnostics> {
    let mut cx = Cx::default();
    match terrain::parse(&mut cx, json, sources) {
        Some(terrain) if cx.diagnostics.errors.is_empty() => Ok((terrain, cx.diagnostics.warnings)),
        _ => Err(cx.diagnostics),
    }
}

/// Parses a root `projection`, returning its globe (`None` for Web
/// Mercator) and its warnings.
pub fn parse_projection(
    json: &Json,
    state: &[(String, Value<'static>)],
) -> (Option<orb_crs::Globe>, Vec<Diagnostic>) {
    let mut cx = Cx {
        state: state.to_vec(),
        ..Cx::default()
    };
    let globe = parse_globe(&mut cx, json);
    (globe, cx.diagnostics.warnings)
}

/// v8 `projection`: a `type` of `mercator`, `globe` (GL JS's
/// interpolation from a vertical perspective through zoom 11 to Mercator
/// from zoom 12), `vertical-perspective` or a low-zoom plane
/// (style-spec#1865: `equal-earth`, `natural-earth`, `equirectangular`,
/// handing over to Mercator between zooms 6 and 7), or GL JS's zoom
/// interpolation from one of them to `mercator`; and style-spec#1866's
/// `transition`, `[start, end]` or `false`, which replaces the zooms it
/// hands over between. Others warn and stay Web Mercator.
fn parse_globe(cx: &mut Cx, json: &Json) -> Option<orb_crs::Globe> {
    let path = "projection.type";
    let ty = json.get("type")?;
    let unsupported = |cx: &mut Cx| {
        cx.warn(
            path,
            ty.span,
            "only \"mercator\", \"globe\", \"vertical-perspective\", \"equal-earth\", \"natural-earth\", \"equirectangular\" and zoom interpolations from the last four to \"mercator\" are supported; drawing Web Mercator",
        );
        None
    };
    // A type from the global state: `["global-state", name]`.
    let from_state = ty.as_array().and_then(|items| match items {
        [op, name] if op.as_str() == Some("global-state") => {
            let name = name.as_str()?;
            match cx.state.iter().find(|(key, _)| key == name) {
                Some((_, Value::String(value))) => Some(value.to_string()),
                _ => None,
            }
        }
        _ => None,
    });
    let (low, handover) = if let Some(name) = from_state.as_deref().or(ty.as_str()) {
        if name == "mercator" {
            return None;
        }
        match low_surface(name) {
            Some((low, handover)) => (low, handover),
            None => return unsupported(cx),
        }
    } else {
        // ["interpolate", ["linear"], ["zoom"], z0, low, z1, "mercator"]
        let items = ty.as_array()?;
        let text = |k: usize| items.get(k).and_then(Json::as_str);
        let number = |k: usize| items.get(k).and_then(Json::as_f64);
        let linear = items
            .get(1)
            .and_then(Json::as_array)
            .is_some_and(|a| a.first().and_then(Json::as_str) == Some("linear"));
        let zoom = items
            .get(2)
            .and_then(Json::as_array)
            .is_some_and(|a| a.len() == 1 && a[0].as_str() == Some("zoom"));
        let low = text(4).filter(|&t| t != "globe").and_then(low_surface);
        match (text(0), linear, zoom, number(3), low, number(5), text(6)) {
            (
                Some("interpolate"),
                true,
                true,
                Some(from),
                Some((low, _)),
                Some(to),
                Some("mercator"),
            ) if items.len() == 7 && from < to => (low, Some([from, to])),
            _ => return unsupported(cx),
        }
    };
    let handover = match json.get("transition") {
        None => handover,
        Some(t) => match parse_transition(t) {
            Some(transition) => transition,
            None => {
                cx.warn(
                    "projection.transition",
                    t.span,
                    "expected [start, end] zooms with start < end, or false",
                );
                handover
            }
        },
    };
    Some(orb_crs::Globe {
        handover,
        plane: low,
    })
}

/// A low-zoom surface v8 names: the sphere (`None`) or a plane, and the
/// zooms it hands over to Mercator between by default.
fn low_surface(name: &str) -> Option<(Option<orb_crs::Projection>, Option<[f64; 2]>)> {
    match name {
        "globe" => Some((None, orb_crs::Globe::V8.handover)),
        "vertical-perspective" => Some((None, None)),
        "equal-earth" | "natural-earth" | "equirectangular" => Some((
            orb_crs::Projection::from_descriptor(name).ok(),
            Some(orb_crs::Globe::PLANE_HANDOVER),
        )),
        _ => None,
    }
}

/// style-spec#1866's `transition`: `[start, end]` zooms, or `false` for no
/// handover. `None` if it is neither.
pub(crate) fn parse_transition(json: &Json) -> Option<Option<[f64; 2]>> {
    match &json.value {
        JsonValue::Bool(false) => Some(None),
        JsonValue::Array(zooms) => match zooms.as_slice() {
            [a, b] => {
                let (a, b) = (a.as_f64()?, b.as_f64()?);
                (a < b).then_some(Some([a, b]))
            }
            _ => None,
        },
        _ => None,
    }
}

/// Parses one source, `id`, of a style. Returns it and its warnings.
pub fn parse_source(id: &str, json: &Json) -> Result<(Source, Vec<Diagnostic>), Diagnostics> {
    let mut cx = Cx::default();
    let source = source::parse_source(&mut cx, id, json, &format!("sources.{id}"));
    match source {
        Some(source) if cx.diagnostics.errors.is_empty() => Ok((source, cx.diagnostics.warnings)),
        _ => Err(cx.diagnostics),
    }
}

/// The parse context: collects diagnostics with absolute paths.
#[derive(Debug, Default)]
pub(crate) struct Cx {
    diagnostics: Diagnostics,
    /// Parsing a style a scene document extends: its layers may be the
    /// scene's types (`model`) and its root may have `models`.
    pub(crate) scene: bool,
    /// The global state `global-state` reads resolve to.
    state: Vec<(String, Value<'static>)>,
}

impl Cx {
    pub(crate) fn error(
        &mut self,
        path: impl Into<String>,
        span: Span,
        message: impl Into<String>,
    ) {
        self.diagnostics.errors.push(Diagnostic {
            path: path.into(),
            span,
            message: message.into(),
        });
    }

    pub(crate) fn warn(&mut self, path: impl Into<String>, span: Span, message: impl Into<String>) {
        self.diagnostics.warnings.push(Diagnostic {
            path: path.into(),
            span,
            message: message.into(),
        });
    }

    /// Adds `orb_expr` diagnostics, whose paths are relative to `prefix`.
    pub(crate) fn errors_at(&mut self, prefix: &str, diagnostics: Vec<Diagnostic>) {
        for d in diagnostics {
            self.error(format!("{prefix}{}", d.path), d.span, d.message);
        }
    }

    pub(crate) fn warnings_at(&mut self, prefix: &str, diagnostics: Vec<Diagnostic>) {
        for d in diagnostics {
            self.warn(format!("{prefix}{}", d.path), d.span, d.message);
        }
    }

    fn expected(&mut self, json: &Json, path: &str, what: &str) {
        self.error(
            path,
            json.span,
            format!("{what} expected, {} found", json.type_name()),
        );
    }

    pub(crate) fn object<'a>(&mut self, json: &'a Json, path: &str) -> Option<&'a [Member]> {
        let object = json.as_object();
        if object.is_none() {
            self.expected(json, path, "object");
        }
        object
    }

    pub(crate) fn array<'a>(&mut self, json: &'a Json, path: &str) -> Option<&'a [Json]> {
        let array = json.as_array();
        if array.is_none() {
            self.expected(json, path, "array");
        }
        array
    }

    pub(crate) fn string(&mut self, json: &Json, path: &str) -> Option<String> {
        let s = json.as_str().map(str::to_string);
        if s.is_none() {
            self.expected(json, path, "string");
        }
        s
    }

    pub(crate) fn number(&mut self, json: &Json, path: &str) -> Option<f64> {
        let n = json.as_f64();
        if n.is_none() {
            self.expected(json, path, "number");
        }
        n
    }

    /// A number within `minimum..=maximum`.
    pub(crate) fn number_in(
        &mut self,
        json: &Json,
        path: &str,
        minimum: f64,
        maximum: f64,
    ) -> Option<f64> {
        let n = self.number(json, path)?;
        if n < minimum {
            self.error(
                path,
                json.span,
                format!("{n} is less than the minimum value {minimum}"),
            );
            None
        } else if n > maximum {
            self.error(
                path,
                json.span,
                format!("{n} is greater than the maximum value {maximum}"),
            );
            None
        } else {
            Some(n)
        }
    }

    pub(crate) fn numbers<const N: usize>(&mut self, json: &Json, path: &str) -> Option<[f64; N]> {
        let items = self.array(json, path)?;
        if items.len() != N {
            self.error(
                path,
                json.span,
                format!("array length {N} expected, length {} found", items.len()),
            );
            return None;
        }
        let mut out = [0.0; N];
        for (i, (item, slot)) in items.iter().zip(&mut out).enumerate() {
            *slot = self.number(item, &format!("{path}[{i}]"))?;
        }
        Some(out)
    }

    pub(crate) fn enumeration<E: Enum>(&mut self, json: &Json, path: &str) -> Option<E> {
        let value = json.as_str().and_then(E::from_name);
        if value.is_none() {
            self.error(
                path,
                json.span,
                format!("expected one of [{}], {json} found", E::VALUES.join(", ")),
            );
        }
        value
    }
}

#[cfg(test)]
mod doc_tests;
#[cfg(test)]
mod tests;
