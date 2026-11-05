//! MapLibre v8 style documents (plan §11.1, S1): parse and validate a v8
//! style into typed sources and layers, with errors and warnings located by
//! JSON path (`layers[3].paint.line-width`) and byte span.
//!
//! `background`, `fill`, `line`, `circle`, `heatmap`, `raster`, `hillshade` and `color-relief`
//! layers are fully typed: every paint and layout property is a
//! [`Property`] backed by an `orb_expr` expression and its classification,
//! with the spec default. Other layer and source types
//! are kept as `Unsupported` with a warning, so callers can skip them.
//! Defaults and coercions follow GL JS.

#![forbid(unsafe_code)]

mod font_faces;
mod geojson;
mod layer;
mod light;
pub mod lights;
mod paint;
mod property;
mod scene;
mod sky;
mod source;
mod spec;
mod sprite;
mod terrain;

use std::collections::HashSet;
use std::fmt;

pub use font_faces::{FontFace, FontFaces};
pub use geojson::parse_geojson;
pub use layer::{
    Background, BackgroundPaint, Circle, CircleLayout, CirclePaint, ColorRelief, ColorReliefPaint,
    Fill, FillExtrusion, FillExtrusionPaint, FillLayout, FillPaint, Heatmap, HeatmapPaint,
    Hillshade, HillshadePaint, Layer, LayerKind, Line, LineLayout, LinePaint, Raster, RasterPaint,
    Selection, Symbol, SymbolLayout, SymbolPaint, SymbolPartPaint,
};
pub use lights::{AmbientLight, Direction, DirectionalLight, Level, Lights, parse_time};
pub use orb_expr::json::Span;
pub use orb_expr::{Color, Diagnostic, Filter, Kind, PropertyExpression, Value};
pub use paint::{PaintMut, PaintRef, PaintValue, Transition, TransitionOverride};
pub use property::{
    Alignment, Anchor, AnchorOffsets, BoundProperty, DemEncoding, Encoding, Enum, HeightAnchor,
    HillshadeMethod, IconTextFit, IlluminationAnchor, LayerBlend, LineCap, LineJoin, Overlap,
    Padding, Property, PropertyValue, Resampling, Scheme, SymbolPlacement, SymbolZOrder,
    TextJustify, TextRotationAlignment, TextTransform, TranslateAnchor, Visibility, WritingMode,
};
pub use scene::{Scene, StyleRef, is_scene, parse_scene, tms_definition};
pub use sky::Atmosphere;
pub use source::{
    GeoJsonData, GeoJsonSource, ImageSource, PromoteId, RasterDemSource, RasterSource, Source,
    SourceKind, TileSet, VectorSource,
};
pub use sprite::Sprite;
pub use terrain::Terrain;

use orb_expr::json::{self, Json, Member};

/// A parsed style.
#[derive(Clone, Debug)]
pub struct Style {
    pub name: Option<String>,
    /// The default camera: `[longitude, latitude]`.
    pub center: Option<[f64; 2]>,
    pub zoom: Option<f64>,
    pub bearing: f64,
    pub pitch: f64,
    /// The default camera's roll about the view axis, degrees.
    pub roll: f64,
    /// How paint changes animate unless a property says otherwise.
    pub transition: Transition,
    pub sprites: Vec<Sprite>,
    /// The glyph PBF URL template, with `{fontstack}` and `{range}`.
    pub glyphs: Option<String>,
    /// Font files for `text-font` names (the root `font-faces`).
    pub font_faces: FontFaces,
    /// The lights: v8's `light` lowered (decision 0026).
    pub lights: Lights,
    /// The root `state`: each global state property's default.
    pub state: Vec<(String, Value<'static>)>,
    /// The atmosphere: v8's `sky` lowered (decision 0025).
    pub atmosphere: Option<Atmosphere>,
    pub terrain: Option<Terrain>,
    /// The root `projection`: a globe (`globe`, `vertical-perspective` or
    /// a zoom interpolation from one to `mercator`), or `None` for Web
    /// Mercator.
    pub globe: Option<orb_crs::Globe>,
    pub sources: Vec<Source>,
    pub layers: Vec<Layer>,
    /// What parsed but isn't supported or rendered yet.
    pub warnings: Vec<Diagnostic>,
}

impl Style {
    pub fn source(&self, id: &str) -> Option<&Source> {
        self.sources.iter().find(|s| s.id == id)
    }
}

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
pub fn parse(text: &str) -> Result<Style, Diagnostics> {
    parse_with_state(text, &[])
}

/// Parses a style whose global state is its root `state` defaults
/// overridden by `overrides` (plan §11.2, S2): `global-state` reads fold to
/// those values, so they classify as constants where they are the only
/// input.
pub fn parse_with_state(
    text: &str,
    overrides: &[(String, Value<'static>)],
) -> Result<Style, Diagnostics> {
    let mut cx = Cx::default();
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

fn parse_root(cx: &mut Cx, doc: &Json, overrides: &[(String, Value<'static>)]) -> Option<Style> {
    let root = cx.object(doc, "")?;
    for m in root {
        match m.key.as_str() {
            "version" | "name" | "metadata" | "center" | "zoom" | "bearing" | "pitch" | "roll"
            | "sources" | "layers" | "sprite" | "glyphs" | "font-faces" | "light" | "state"
            | "sky" | "terrain" | "projection" => {}
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
    Some(Style {
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
        state,
        atmosphere: doc.get("sky").map(|json| sky::parse(cx, json)),
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
) -> Result<(Layer, Vec<Diagnostic>), Diagnostics> {
    let mut cx = Cx {
        state: state.to_vec(),
        ..Cx::default()
    };
    let layer = layer::parse_layer(&mut cx, json, index, sources, &mut HashSet::new());
    match layer {
        Some(layer) if cx.diagnostics.errors.is_empty() => Ok((layer, cx.diagnostics.warnings)),
        _ => Err(cx.diagnostics),
    }
}

/// One modelled paint property of a `layer_type` layer from its value in
/// the layer's `paint` (`None`: unset), with `global-state` reads resolved
/// to `state`, as [`parse_layer`] gives it: an invalid value gives the
/// default. `None` for a property Orbis doesn't model.
pub fn parse_paint<T: PropertyValue>(
    layer_type: &str,
    name: &str,
    json: Option<&Json>,
    state: &[(String, Value<'static>)],
) -> Option<Property<T>> {
    let prop = spec::find_prop(layer_type, true, name)?;
    let mut cx = Cx {
        state: state.to_vec(),
        ..Cx::default()
    };
    Some(layer::property_at(&mut cx, json, name, prop))
}

/// The v8 spec of a modelled paint (`paint: true`) or layout property of a
/// `layer_type` layer.
pub fn property_spec(layer_type: &str, paint: bool, name: &str) -> Option<orb_expr::PropertySpec> {
    spec::find(layer_type, paint, name)
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
    let sky = sky::parse(&mut cx, json);
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

/// v8 `projection.type`: `mercator`, `globe` (GL JS's interpolation from
/// a vertical perspective through zoom 11 to Mercator from zoom 12) or
/// `vertical-perspective`, or that interpolation with other zooms. Others
/// warn and stay Web Mercator.
fn parse_globe(cx: &mut Cx, json: &Json) -> Option<orb_crs::Globe> {
    let path = "projection.type";
    let ty = json.get("type")?;
    let unsupported = |cx: &mut Cx| {
        cx.warn(
            path,
            ty.span,
            "only \"mercator\", \"globe\", \"vertical-perspective\" and zoom interpolations between the last and \"mercator\" are supported; drawing Web Mercator",
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
    if let Some(name) = from_state.as_deref().or(ty.as_str()) {
        return match name {
            "mercator" => None,
            "globe" => Some(orb_crs::Globe::V8),
            "vertical-perspective" => Some(orb_crs::Globe { handover: None }),
            _ => unsupported(cx),
        };
    }
    // ["interpolate", ["linear"], ["zoom"], z0, "vertical-perspective", z1, "mercator"]
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
    match (
        text(0),
        linear,
        zoom,
        number(3),
        text(4),
        number(5),
        text(6),
    ) {
        (
            Some("interpolate"),
            true,
            true,
            Some(from),
            Some("vertical-perspective"),
            Some(to),
            Some("mercator"),
        ) if items.len() == 7 && from < to => Some(orb_crs::Globe {
            handover: Some([from, to]),
        }),
        _ => unsupported(cx),
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
mod tests;
