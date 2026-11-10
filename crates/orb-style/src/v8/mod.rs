//! The v8 front end: parses and
//! validates v8 styles into a [`Style`], with errors and
//! warnings located by JSON path
//! (`layers[3].paint.line-width`) and byte span; keeps the documents of
//! record ([`StyleDoc`]) and applies GL JS-style edits to them as
//! transactions, diffed into classified changes. Defaults and coercions
//! follow GL JS.

mod diff;
mod doc;
mod font_faces;
mod geojson;
mod layer;
mod light;
mod paint;
mod sky;
mod source;
mod sprite;
mod terrain;
mod tx;

use std::fmt;

pub use doc::StyleDoc;
pub use geojson::{apply_diffs, geojson_features, parse_geojson};
pub use light::default_light;
pub use orb_expr::json::Span;
pub use orb_expr::{Diagnostic, Filter};
pub use source::{EMPTY_SOURCE, tms_definition};
pub use tx::{FeatureTarget, Op, Transaction};

// The native model this front end builds.
use crate::*;

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
/// resolved to the root `state` defaults, with no previews.
pub fn parse(text: &str) -> Result<Style, Diagnostics> {
    parse_with(text, Previews::NONE, &[])
}

/// Parses a style with `previews` enabled, whose global state is its root
/// `state` defaults overridden by `overrides`: `global-state` reads fold to
/// those values, so they classify as constants where they are the only
/// input.
pub fn parse_with(
    text: &str,
    previews: Previews,
    overrides: &[(String, Value<'static>)],
) -> Result<Style, Diagnostics> {
    let mut doc = read(text)?;
    deref_layers(&mut doc);
    parse_json(&doc, previews, overrides)
}

/// Reads a style document's JSON; a syntax error is its one diagnostic.
pub fn read(text: &str) -> Result<Json, Diagnostics> {
    json::parse(text).map_err(|e| {
        let mut cx = Cx::default();
        let at = Span {
            start: e.offset,
            end: e.offset,
        };
        cx.error("", at, e.message);
        cx.diagnostics
    })
}

/// [`parse_with`] of a document already read, whose layer references
/// [`deref_layers`] resolved.
pub(crate) fn parse_json(
    doc: &Json,
    previews: Previews,
    overrides: &[(String, Value<'static>)],
) -> Result<Style, Diagnostics> {
    let mut cx = Cx::new(&[], previews);
    let style = parse_root(&mut cx, doc, overrides);
    let (style, warnings) = cx.finish(style)?;
    Ok(Style { warnings, ..style })
}

/// The properties a layer with a `ref` takes from the layer it names.
const REF_PROPERTIES: [&str; 7] = [
    "type",
    "source",
    "source-layer",
    "minzoom",
    "maxzoom",
    "filter",
    "layout",
];

/// Replaces each layer that has a `ref` with a copy taking
/// [`REF_PROPERTIES`] from the layer it names, as GL JS's `derefLayers`
/// and MapLibre Native's parser read v8's deprecated layer references. A
/// layer takes them from the named layer as written, not as resolved.
pub(crate) fn deref_layers(doc: &mut Json) {
    let Some(JsonValue::Array(layers)) = doc.get_mut("layers").map(|l| &mut l.value) else {
        return;
    };
    let inherited: Vec<(usize, Vec<Member>)> = layers
        .iter()
        .enumerate()
        .filter_map(|(i, layer)| {
            let id = layer.get("ref")?.as_str()?;
            let parent = layers
                .iter()
                .rfind(|l| l.get("id").and_then(Json::as_str) == Some(id))?;
            let taken = (parent.as_object().unwrap_or_default().iter())
                .filter(|m| REF_PROPERTIES.contains(&m.key.as_str()))
                .cloned()
                .collect();
            Some((i, taken))
        })
        .collect();
    for (i, taken) in inherited {
        if let JsonValue::Object(members) = &mut layers[i].value {
            members.retain(|m| m.key != "ref" && !REF_PROPERTIES.contains(&m.key.as_str()));
            members.extend(taken);
        }
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
            | "sky" | "terrain" | "projection" | "transition" => {}
            key if UNMODELED_ROOT.contains(&key) => {
                cx.warn(
                    key,
                    m.key_span,
                    format!("\"{key}\" is not supported yet and is ignored"),
                );
            }
            _ => cx.warn_unknown("", m),
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
        Some(sources) => source::parse_sources(cx, sources)
            .into_iter()
            .map(std::sync::Arc::new)
            .collect(),
    };
    let layers = match doc.get("layers") {
        None => {
            cx.error("", doc.span, "missing required property \"layers\"");
            Vec::new()
        }
        Some(layers) => match cx.array(layers, "layers") {
            None => Vec::new(),
            Some(items) => {
                let mut ids = rustc_hash::FxHashSet::default();
                items
                    .iter()
                    .enumerate()
                    .filter_map(|(i, l)| layer::parse_layer(cx, l, i, &sources, &mut ids))
                    .map(std::sync::Arc::new)
                    .collect()
            }
        },
    };
    let (projection, globe) = doc
        .get("projection")
        .map_or((None, None), |p| parse_surface(cx, p));
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
        light: light::parse(cx, doc.get("light")),
        globe,
        projection,
        state,
        atmosphere: doc.get("sky").map(|json| sky::parse(cx, json)),
        terrain: doc
            .get("terrain")
            .and_then(|json| terrain::parse(cx, json, &sources)),
        sources,
        layers,
        warnings: Vec::new(),
        previews: cx.previews,
        color_space: doc
            .get("metadata")
            .map_or(ColorSpace::Srgb, |m| parse_color_space(cx, m)),
    })
}

/// The metadata key of the colour space colours mix in.
const COLOR_SPACE: &str = "dev.sargunv.orbis:color-space";

/// The colour space the root `metadata` names, behind its preview: GL JS,
/// MapLibre Native and validators ignore metadata, so a style written for
/// it still loads everywhere.
fn parse_color_space(cx: &mut Cx, metadata: &Json) -> ColorSpace {
    let Some(json) = metadata.get(COLOR_SPACE) else {
        return ColorSpace::Srgb;
    };
    let path = format!("metadata.{COLOR_SPACE}");
    if !cx.preview(
        Preview::ColorSpace,
        &path,
        json.span,
        format!("\"{COLOR_SPACE}\""),
    ) {
        return ColorSpace::Srgb;
    }
    match json.as_str() {
        Some("srgb") => ColorSpace::Srgb,
        Some("linear") => ColorSpace::Linear,
        _ => {
            cx.warn(
                path,
                json.span,
                format!("expected one of [\"srgb\", \"linear\"], {json} found; \"srgb\" used"),
            );
            ColorSpace::Srgb
        }
    }
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
pub(crate) fn resolve_state(
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
    sources: &[std::sync::Arc<Source>],
    state: &[(String, Value<'static>)],
    previews: Previews,
) -> Result<(Layer, Vec<Diagnostic>), Diagnostics> {
    let mut cx = Cx::new(state, previews);
    let layer = layer::parse_layer(&mut cx, json, index, sources, &mut Default::default());
    cx.finish(layer)
}

/// Parses a root light with the current global state.
pub(crate) fn parse_light(
    json: &Json,
    state: &[(String, Value<'static>)],
    previews: Previews,
) -> Result<Light, Diagnostics> {
    let mut cx = Cx {
        state: state.to_vec(),
        previews,
        ..Cx::default()
    };
    let light = light::parse(&mut cx, Some(json));
    if cx.diagnostics.errors.is_empty() {
        Ok(light)
    } else {
        Err(cx.diagnostics)
    }
}

/// Parses a root sky with the current global state, returning its warnings.
pub(crate) fn parse_sky(
    json: &Json,
    state: &[(String, Value<'static>)],
    previews: Previews,
) -> Result<(Atmosphere, Vec<Diagnostic>), Diagnostics> {
    let mut cx = Cx::new(state, previews);
    let sky = sky::parse(&mut cx, json);
    cx.finish(Some(sky))
}

/// Parses a root `terrain` against the style's sources, returning its
/// warnings.
pub(crate) fn parse_terrain(
    json: &Json,
    sources: &[std::sync::Arc<Source>],
) -> Result<(Terrain, Vec<Diagnostic>), Diagnostics> {
    let mut cx = Cx::default();
    let terrain = terrain::parse(&mut cx, json, sources);
    cx.finish(terrain)
}

/// Parses a root `projection`, returning its plane (`None` for Web
/// Mercator), its low-zoom surface and its warnings.
pub(crate) fn parse_projection(
    json: &Json,
    state: &[(String, Value<'static>)],
    previews: Previews,
) -> (
    Option<orb_crs::Projection>,
    Option<orb_crs::Globe>,
    Vec<Diagnostic>,
) {
    let mut cx = Cx::new(state, previews);
    let (plane, globe) = parse_surface(&mut cx, json);
    (plane, globe, cx.diagnostics.warnings)
}

/// v8 `projection`, returning the plane the map draws on and the low-zoom
/// surface its tiles draw on. Its `type` is `mercator`, `globe` (GL JS's
/// interpolation from a vertical perspective through zoom 11 to Mercator
/// from zoom 12), `vertical-perspective`, or GL JS's zoom interpolation
/// from one of them to `mercator`. Previews add more: a low-zoom plane
/// (`dev.sargunv.orbis:equal-earth`, style-spec#1865;
/// `dev.sargunv.orbis:equirectangular`, style-spec#1623;
/// `dev.sargunv.orbis:natural-earth`), handing over to Mercator between
/// zooms 6 and 7, and the zoom interpolation from one of them; a
/// coordinate reference system `orb_crs` registers (`EPSG:3413`, or a
/// parametric kind such as `dev.sargunv.orbis:polar-stereographic` with
/// PROJ-named numeric parameters beside the `type`), which is the plane
/// itself; and style-spec#1866's `dev.sargunv.orbis:transition`,
/// `[start, end]` or `false`, replacing the zooms it hands over between
/// (a CRS with a transition is a low-zoom plane over Web Mercator).
/// Others, and inputs of previews that are off, warn and stay Web
/// Mercator.
fn parse_surface(
    cx: &mut Cx,
    json: &Json,
) -> (Option<orb_crs::Projection>, Option<orb_crs::Globe>) {
    let path = "projection.type";
    let Some(ty) = json.get("type") else {
        return (None, None);
    };
    let unsupported = |cx: &mut Cx, message: &str| {
        cx.warn(path, ty.span, format!("{message}; drawing Web Mercator"));
        (None, None)
    };
    const SUPPORTED: &str = "only \"mercator\", \"globe\", \"vertical-perspective\" and zoom interpolations from the last to \"mercator\" are supported, with previews for more";
    // PROJ-named parameters of a plane: `lon_0`, `lat_ts` and the like.
    let mut descriptor = orb_crs::Descriptor::default();
    for m in json.as_object().unwrap_or_default() {
        match (m.key.as_str(), m.value.as_f64()) {
            ("type" | PROJECTION_TRANSITION, _) => {}
            (key, Some(n)) => {
                let (at, what) = (
                    format!("projection.{key}"),
                    format!("projection parameter \"{key}\""),
                );
                if cx.preview(Preview::Crs, at, m.key_span, what) {
                    descriptor.params.push((key.to_string(), n));
                }
            }
            (_, None) => cx.warn_unknown("projection", m),
        }
    }
    let transition = json.get(PROJECTION_TRANSITION).and_then(|t| {
        let at = format!("projection.{PROJECTION_TRANSITION}");
        let what = format!("\"{PROJECTION_TRANSITION}\"");
        if !cx.preview(Preview::ProjectionTransition, &at, t.span, what) {
            return None;
        }
        let parsed = parse_transition(t);
        if parsed.is_none() {
            cx.warn(
                at,
                t.span,
                "expected [start, end] zooms with start < end, or false",
            );
        }
        parsed
    });
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
            return (None, None);
        }
        let Some((kind, preview)) = projection_kind(name) else {
            return unsupported(cx, &format!("{SUPPORTED}, not \"{name}\""));
        };
        if let Some(preview) = preview
            && !cx.preview(preview, path, ty.span, format!("projection \"{name}\""))
        {
            return (None, None);
        }
        descriptor.kind = kind.to_string();
        match low_surface(&descriptor) {
            Some(Ok(surface)) => surface,
            Some(Err(e)) => return unsupported(cx, &e.to_string()),
            None => {
                // A coordinate reference system: the plane, or with a
                // transition a low-zoom plane.
                return match orb_crs::Projection::new(&descriptor) {
                    Ok(plane) => match transition {
                        Some(Some(handover)) => (
                            None,
                            Some(orb_crs::Globe {
                                handover: Some(handover),
                                plane: Some(plane),
                            }),
                        ),
                        _ => (Some(plane), None),
                    },
                    Err(e) => unsupported(cx, &format!("{SUPPORTED}, not {}", e.0)),
                };
            }
        }
    } else {
        // ["interpolate", ["linear"], ["zoom"], z0, low, z1, "mercator"]
        let Some(items) = ty.as_array() else {
            return unsupported(cx, SUPPORTED);
        };
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
        let low_name = text(4).filter(|&t| t != "globe" && t != "mercator");
        let low_kind = low_name.and_then(projection_kind);
        if let (Some(name), Some((kind, Some(preview)))) = (low_name, low_kind)
            && low_surface(&orb_crs::Descriptor::named(kind)).is_some()
            && !cx.preview(preview, path, ty.span, format!("projection \"{name}\""))
        {
            return (None, None);
        }
        let low = low_kind
            .and_then(|(kind, _)| low_surface(&orb_crs::Descriptor::named(kind)))
            .and_then(Result::ok);
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
            _ => return unsupported(cx, SUPPORTED),
        }
    };
    let globe = orb_crs::Globe {
        handover: transition.unwrap_or(handover),
        plane: low,
    };
    (None, Some(globe))
}

/// A low-zoom surface: the sphere (`None`) or a plane, and the zooms it
/// hands over to Mercator between.
type LowSurface = (Option<orb_crs::Projection>, Option<[f64; 2]>);

/// The low-zoom surface v8 names, with its default handover; `None` for
/// other names.
fn low_surface(
    descriptor: &orb_crs::Descriptor,
) -> Option<Result<LowSurface, orb_crs::UnknownProjection>> {
    match descriptor.kind.as_str() {
        "globe" => Some(Ok((None, orb_crs::Globe::V8.handover))),
        "vertical-perspective" => Some(Ok((None, None))),
        "equal-earth" | "natural-earth" | "equirectangular" => Some(
            orb_crs::Projection::new(descriptor)
                .map(|plane| (Some(plane), Some(orb_crs::Globe::PLANE_HANDOVER))),
        ),
        _ => None,
    }
}

/// The projection key of style-spec#1866's `transition`.
const PROJECTION_TRANSITION: &str = "dev.sargunv.orbis:transition";

/// What a projection `type` names: its `orb_crs` descriptor kind, and the
/// preview it needs if it is beyond the spec. Orbis's own kinds are in its
/// namespace; a CRS identifier (`EPSG:3413`, `OGC:CRS84`) keeps its
/// standard name. `None` for anything else.
fn projection_kind(name: &str) -> Option<(&str, Option<Preview>)> {
    if let Some(kind) = name.strip_prefix(crate::NAMESPACE) {
        let preview = match kind {
            "equal-earth" => Preview::EqualEarth,
            "natural-earth" => Preview::NaturalEarth,
            "equirectangular" => Preview::Equirectangular,
            "mercator" | "globe" | "vertical-perspective" => return None,
            _ if kind.contains(':') => return None,
            _ => Preview::Crs,
        };
        return Some((kind, Some(preview)));
    }
    match name {
        "mercator" | "globe" | "vertical-perspective" => Some((name, None)),
        _ if name.contains(':') => Some((name, Some(Preview::Crs))),
        _ => None,
    }
}

/// style-spec#1866's transition: `[start, end]` zooms, or `false` for no
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
pub fn parse_source(
    id: &str,
    json: &Json,
    previews: Previews,
) -> Result<(Source, Vec<Diagnostic>), Diagnostics> {
    let mut cx = Cx::new(&[], previews);
    let source = source::parse_source(&mut cx, id, json, &format!("sources.{id}"));
    cx.finish(source)
}

/// The parse context: collects diagnostics with absolute paths.
#[derive(Debug, Default)]
pub(crate) struct Cx {
    diagnostics: Diagnostics,
    /// The global state `global-state` reads resolve to.
    state: Vec<(String, Value<'static>)>,
    /// The previews whose inputs are accepted; others warn.
    pub(crate) previews: Previews,
    /// The properties whose values read an input beyond the spec, each
    /// reported once.
    not_portable: std::collections::HashSet<&'static str>,
}

impl Cx {
    /// A context resolving `global-state` reads to `state`, accepting the
    /// inputs of `previews`.
    fn new(state: &[(String, Value<'static>)], previews: Previews) -> Self {
        Self {
            diagnostics: Diagnostics::default(),
            state: state.to_vec(),
            previews,
            not_portable: Default::default(),
        }
    }

    /// What was parsed and its warnings; every diagnostic instead if
    /// nothing was, or anything was an error.
    fn finish<T>(self, parsed: Option<T>) -> Result<(T, Vec<Diagnostic>), Diagnostics> {
        match parsed {
            Some(parsed) if self.diagnostics.errors.is_empty() => {
                Ok((parsed, self.diagnostics.warnings))
            }
            _ => Err(self.diagnostics),
        }
    }

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

    /// Whether `preview` is on; when it is off, warns that `what`, at
    /// `path`, needs it, so the caller falls back to the default.
    pub(crate) fn preview(
        &mut self,
        preview: Preview,
        path: impl Into<String>,
        span: Span,
        what: impl fmt::Display,
    ) -> bool {
        let on = self.previews.contains(preview);
        if !on {
            self.warn(
                path,
                span,
                format!("{what} needs the \"{preview}\" preview, which is off; ignored"),
            );
        }
        on
    }

    /// Warns of a member `prefix` (an object's path) doesn't define.
    pub(crate) fn warn_unknown(&mut self, prefix: &str, member: &Member) {
        let path = match prefix {
            "" => member.key.clone(),
            prefix => format!("{prefix}.{}", member.key),
        };
        let message = format!("unknown property \"{}\"", member.key);
        self.warn(path, member.key_span, message);
    }

    /// Warns, once per property and document, that a value Orbis draws reads an input
    /// the spec doesn't allow there, so GL JS and MapLibre Native reject
    /// the style.
    pub(crate) fn not_portable(&mut self, path: &str, span: Span, prop: &Prop, beyond: Inputs) {
        if !self.not_portable.insert(prop.name) {
            return;
        }
        let mut inputs: Vec<&str> = beyond.names().collect();
        if inputs.is_empty() {
            inputs.push("inputs");
        }
        self.warn(
            path,
            span,
            format!(
                "this value of \"{}\" reads {}, beyond the spec: Orbis draws it, but GL JS and MapLibre Native reject the style (reported once per property)",
                prop.name,
                inputs.join(" and "),
            ),
        );
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
