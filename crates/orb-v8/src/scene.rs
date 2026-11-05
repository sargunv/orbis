//! The scene document, v0 (plan §11.4): what v8 can't say about a map,
//! next to an unmodified v8 style. v0 carries the CRS/TMS module (a
//! projection for the surface and a tile matrix set per source), the
//! lighting module (`lights` and per-layer `emissive-strength`; see
//! [`crate::lights`]) and the `atmosphere` (v8 `sky`'s properties with
//! `star-intensity` and `backdrop-color`; see [`crate::Atmosphere`]).
//!
//! ```json
//! {
//!   "style": "https://tiles.example/style.json",
//!   "projection": {"type": "EPSG:2193"},
//!   "surface": "plane",
//!   "sources": {"linz": {"tileMatrixSet": "NZTM2000Quad"}}
//! }
//! ```
//!
//! `style` is a URL (relative to the scene document's) or an inline v8
//! object; `tileMatrixSet` is a registered OGC identifier or an inline OGC
//! 2D-TMS document (version 2.0, or 1.0's `identifier`/`topLeftCorner`).

use orb_crs::{Descriptor, Projection, TileMatrix, TileMatrixSet, TileMatrixSetDefinition};
use orb_expr::json::{self, Json, JsonValue};

use crate::{Cx, Diagnostics};

/// A parsed scene document.
#[derive(Clone, Debug)]
pub struct SceneDocument {
    pub style: StyleRef,
    /// The surface's projection; Web Mercator when absent.
    pub projection: Option<Projection>,
    /// A low-zoom plane the surface's tiles draw on before handing over to
    /// it: a `projection` with a `transition` (style-spec#1866), or one
    /// that adapts by default (`equal-earth`, `natural-earth`,
    /// `equirectangular`). Its plane is then Web Mercator.
    pub globe: Option<orb_crs::Globe>,
    /// Tile matrix sets of sources by source id; the rest use
    /// `WebMercatorQuad`.
    pub sources: Vec<(String, TileMatrixSet)>,
    /// The scene's lights and emissive layers, which replace the style's
    /// lowered `light`.
    pub lights: Option<crate::Lights>,
    /// The scene's atmosphere, which replaces the style's lowered `sky`.
    pub atmosphere: Option<crate::Atmosphere>,
    /// What the scene adds to its style: glTF models by id (URLs relative
    /// to the scene document) and layers. Added layers are parsed in the
    /// scene dialect with the style ([`crate::Dialect::Scene`]).
    pub models: Vec<(String, String)>,
    pub layers: Vec<AddedLayer>,
    /// Symbols and circles are hidden by 3D layers as well as by the
    /// terrain (`symbol-occlusion`).
    pub symbol_occlusion: bool,
    pub warnings: Vec<crate::Diagnostic>,
}

/// A layer of the scene's `layers+`: its definition, placed before or
/// after a style layer, or above them all.
#[derive(Clone, Debug)]
pub struct AddedLayer {
    pub json: Json,
    pub before: Option<String>,
    pub after: Option<String>,
}

/// Where a scene document's v8 style is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StyleRef {
    Url(String),
    /// The text of an inline v8 document.
    Inline(String),
}

/// Whether a document is a scene document rather than a v8 style: an
/// object with a `style` member and no `version`.
pub fn is_scene(text: &str) -> bool {
    json::parse(text).is_ok_and(|doc| doc.get("style").is_some() && doc.get("version").is_none())
}

/// Parses a scene document.
pub fn parse_scene(text: &str) -> Result<SceneDocument, Diagnostics> {
    let mut cx = Cx::default();
    let doc = match json::parse(text) {
        Ok(doc) => doc,
        Err(e) => {
            let span = json::Span {
                start: e.offset,
                end: e.offset,
            };
            cx.error("", span, e.message);
            return Err(cx.diagnostics);
        }
    };
    let Some(root) = cx.object(&doc, "") else {
        return Err(cx.diagnostics);
    };
    let mut style = None;
    let mut projection = None;
    let mut sources = Vec::new();
    let (mut lights, mut materials) = (None, Vec::new());
    let mut atmosphere = None;
    let mut models = Vec::new();
    let mut added = Vec::new();
    let mut symbol_occlusion = false;
    for m in root {
        match m.key.as_str() {
            "$schema" => {}
            "style" => match &m.value.value {
                JsonValue::String(url) => style = Some(StyleRef::Url(url.clone())),
                JsonValue::Object(_) => {
                    let span = m.value.span;
                    style = Some(StyleRef::Inline(
                        text[span.start as usize..span.end as usize].to_string(),
                    ));
                }
                _ => cx.error("style", m.value.span, "expected a URL or a v8 style object"),
            },
            "projection" => projection = parse_projection(&mut cx, &m.value, "projection"),
            "surface" => match m.value.as_str() {
                Some("plane") => {}
                Some(other) => cx.error(
                    "surface",
                    m.value.span,
                    format!("surface `{other}` is not supported yet"),
                ),
                None => cx.error("surface", m.value.span, "expected a string"),
            },
            "sources" => {
                let Some(members) = cx.object(&m.value, "sources") else {
                    continue;
                };
                for source in members {
                    let path = format!("sources.{}", source.key);
                    let Some(fields) = cx.object(&source.value, &path) else {
                        continue;
                    };
                    for field in fields {
                        let path = format!("{path}.{}", field.key);
                        match field.key.as_str() {
                            "tileMatrixSet" => {
                                if let Some(tms) = parse_tms(&mut cx, &field.value, &path) {
                                    sources.push((source.key.clone(), tms));
                                }
                            }
                            key => cx.warn(
                                &path,
                                field.key_span,
                                format!("unknown property \"{key}\""),
                            ),
                        }
                    }
                }
            }
            "lights" => lights = Some(crate::lights::parse_lights(&mut cx, &m.value)),
            "layers" => materials = crate::lights::parse_layers(&mut cx, &m.value),
            "symbol-occlusion" => match m.value.as_bool() {
                Some(b) => symbol_occlusion = b,
                None => cx.error("symbol-occlusion", m.value.span, "expected a boolean"),
            },
            "models" => {
                for model in cx.object(&m.value, "models").unwrap_or_default() {
                    if let Some(url) = cx.string(&model.value, &format!("models.{}", model.key)) {
                        models.push((model.key.clone(), url));
                    }
                }
            }
            "layers+" => {
                for (i, layer) in cx
                    .array(&m.value, "layers+")
                    .unwrap_or_default()
                    .iter()
                    .enumerate()
                {
                    let path = format!("layers+[{i}]");
                    let Some(members) = cx.object(layer, &path) else {
                        continue;
                    };
                    let place =
                        |key: &str| layer.get(key).and_then(Json::as_str).map(str::to_owned);
                    let (before, after) = (place("before"), place("after"));
                    if before.is_some() && after.is_some() {
                        cx.error(
                            &path,
                            layer.span,
                            "a layer is placed before or after another, not both",
                        );
                        continue;
                    }
                    let json = Json::new(
                        JsonValue::Object(
                            members
                                .iter()
                                .filter(|m| m.key != "before" && m.key != "after")
                                .cloned()
                                .collect(),
                        ),
                        layer.span,
                    );
                    added.push(AddedLayer {
                        json,
                        before,
                        after,
                    });
                }
            }
            "atmosphere" => {
                atmosphere = Some(crate::sky::parse(
                    &mut cx,
                    &m.value,
                    crate::sky::Declared::Scene,
                ));
            }
            key => cx.warn(key, m.key_span, format!("unknown property \"{key}\"")),
        }
    }
    match &mut lights {
        Some(lights) => lights.materials = materials,
        None if !materials.is_empty() => cx.warn(
            "layers",
            doc.span,
            "layers' materials take effect with the scene's lights",
        ),
        None => {}
    }
    let Some(style) = style else {
        cx.error("", doc.span, "missing required property \"style\"");
        return Err(cx.diagnostics);
    };
    if !cx.diagnostics.errors.is_empty() {
        return Err(cx.diagnostics);
    }
    // An adapting projection is a low-zoom plane over Web Mercator.
    let (projection, globe) = match projection {
        Some((plane, Some(handover))) => (
            None,
            Some(orb_crs::Globe {
                handover: Some(handover),
                plane: Some(plane),
            }),
        ),
        Some((plane, None)) => (Some(plane), None),
        None => (None, None),
    };
    Ok(SceneDocument {
        style,
        projection,
        globe,
        sources,
        lights,
        atmosphere,
        models,
        layers: added,
        symbol_occlusion,
        warnings: cx.diagnostics.warnings,
    })
}

/// A projection descriptor: `{"type": ..., <PROJ-named parameters>}`,
/// and the zooms it hands over to Web Mercator between: its `transition`,
/// or by default `[6, 7]` for `equal-earth`, `natural-earth` and
/// `equirectangular` and none for the rest, which may well be the plane
/// their data is on.
pub(crate) fn parse_projection(
    cx: &mut Cx,
    json: &Json,
    path: &str,
) -> Option<(Projection, Option<[f64; 2]>)> {
    let members = cx.object(json, path)?;
    let mut descriptor = Descriptor::default();
    let mut transition = None;
    for m in members {
        match (m.key.as_str(), &m.value.value) {
            ("type", JsonValue::String(kind)) => descriptor.kind = kind.clone(),
            ("transition", _) => match crate::parse_transition(&m.value) {
                Some(t) => transition = Some(t),
                None => {
                    cx.error(
                        format!("{path}.transition"),
                        m.value.span,
                        "expected [start, end] zooms with start < end, or false",
                    );
                    return None;
                }
            },
            (key, JsonValue::Number(n)) if key != "type" => {
                descriptor.params.push((key.to_string(), *n));
            }
            (key, _) => {
                cx.error(format!("{path}.{key}"), m.value.span, "expected a number");
                return None;
            }
        }
    }
    let adapts = matches!(
        descriptor.kind.as_str(),
        "equal-earth" | "natural-earth" | "equirectangular"
    );
    let handover = transition.unwrap_or(adapts.then_some(orb_crs::Globe::PLANE_HANDOVER));
    match Projection::new(&descriptor) {
        Ok(p) => Some((p, handover)),
        Err(e) => {
            cx.error(path, json.span, e.to_string());
            None
        }
    }
}

fn parse_tms(cx: &mut Cx, json: &Json, path: &str) -> Option<TileMatrixSet> {
    let result = match &json.value {
        JsonValue::String(id) => TileMatrixSet::by_id(id),
        JsonValue::Object(_) => match tms_definition(json) {
            Ok(def) => TileMatrixSet::from_definition(&def),
            Err(message) => {
                cx.error(path, json.span, message);
                return None;
            }
        },
        _ => {
            cx.error(
                path,
                json.span,
                "expected an identifier or a tile matrix set",
            );
            return None;
        }
    };
    result
        .map_err(|e| cx.error(path, json.span, e.to_string()))
        .ok()
}

/// An OGC 2D-TMS document (2.0, or 1.0's names) as a definition.
pub fn tms_definition(json: &Json) -> Result<TileMatrixSetDefinition, String> {
    let text = |key: &str| json.get(key).and_then(Json::as_str);
    let id = text("id")
        .or_else(|| text("identifier"))
        .unwrap_or("inline")
        .to_string();
    let crs = json
        .get("crs")
        .or_else(|| json.get("supportedCRS"))
        .and_then(|c| c.as_str().or_else(|| c.get("uri").and_then(Json::as_str)))
        .ok_or("a tile matrix set needs a `crs`")?
        .to_string();
    let northing_first = json
        .get("orderedAxes")
        .and_then(Json::as_array)
        .and_then(|axes| axes.first())
        .and_then(Json::as_str)
        .map(|axis| {
            matches!(
                axis.to_ascii_lowercase().as_str(),
                "y" | "n" | "lat" | "latitude" | "north" | "northing"
            )
        });
    let matrices = json
        .get("tileMatrices")
        .or_else(|| json.get("tileMatrix"))
        .and_then(Json::as_array)
        .ok_or("a tile matrix set needs `tileMatrices`")?;
    let matrices = matrices
        .iter()
        .map(|m| {
            let number = |key: &str| m.get(key).and_then(Json::as_f64);
            let count = |key: &str| {
                number(key)
                    .filter(|n| *n >= 1.0 && n.fract() == 0.0 && *n <= f64::from(u32::MAX))
                    .map(|n| n as u32)
                    .ok_or_else(|| format!("a tile matrix needs a whole `{key}`"))
            };
            let id = m
                .get("id")
                .or_else(|| m.get("identifier"))
                .and_then(Json::as_str)
                .ok_or("a tile matrix needs an `id`")?;
            let origin = m
                .get("pointOfOrigin")
                .or_else(|| m.get("topLeftCorner"))
                .and_then(Json::as_array)
                .and_then(|p| Some([p.first()?.as_f64()?, p.get(1)?.as_f64()?]))
                .ok_or("a tile matrix needs a `pointOfOrigin`")?;
            Ok(TileMatrix {
                id: id.to_string(),
                scale_denominator: number("scaleDenominator")
                    .ok_or("a tile matrix needs a `scaleDenominator`")?,
                cell_size: number("cellSize"),
                point_of_origin: origin,
                bottom_left: m.get("cornerOfOrigin").and_then(Json::as_str) == Some("bottomLeft"),
                tile_width: count("tileWidth")?,
                tile_height: count("tileHeight")?,
                matrix_width: count("matrixWidth")?,
                matrix_height: count("matrixHeight")?,
                variable_widths: m.get("variableMatrixWidths").is_some(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(TileMatrixSetDefinition {
        id,
        crs,
        northing_first,
        matrices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_projection_and_source_matrix_sets() {
        let text = r#"{
            "style": "style.json",
            "projection": {"type": "EPSG:3413"},
            "surface": "plane",
            "sources": {
                "linz": {"tileMatrixSet": "NZTM2000Quad"},
                "gibs": {"tileMatrixSet": {
                    "id": "GIBS3413", "crs": "http://www.opengis.net/def/crs/EPSG/0/3413",
                    "tileMatrices": [
                        {"id": "0", "scaleDenominator": 29257142.85714286, "pointOfOrigin": [-4194304, 4194304],
                         "tileWidth": 512, "tileHeight": 512, "matrixWidth": 2, "matrixHeight": 2},
                        {"id": "1", "scaleDenominator": 14628571.42857143, "pointOfOrigin": [-4194304, 4194304],
                         "tileWidth": 512, "tileHeight": 512, "matrixWidth": 4, "matrixHeight": 4}
                    ]}}
            }
        }"#;
        assert!(is_scene(text));
        let scene = parse_scene(text).unwrap();
        assert_eq!(scene.style, StyleRef::Url("style.json".into()));
        let projection = scene.projection.unwrap();
        assert_eq!(projection.descriptor(), Descriptor::named("EPSG:3413"));
        assert_eq!(scene.sources.len(), 2);
        let gibs = &scene.sources[1].1;
        assert!(gibs.projection().same_plane(&projection));
        assert_eq!(gibs.matrix_size(0), (2, 2));
        // 8,192 m pixels, 512 per tile.
        assert!((gibs.tile_span(0) - 4_194_304.0).abs() < 1e-3);
        assert_eq!(scene.globe, None, "national planes stay fixed");
    }

    #[test]
    fn world_planes_adapt_to_mercator() {
        let scene = |projection: &str| {
            parse_scene(&format!(
                r#"{{"style": "s.json", "projection": {projection}}}"#
            ))
            .unwrap()
        };
        let equal_earth = Projection::from_descriptor("equal-earth").unwrap();
        let adaptive = scene(r#"{"type": "equal-earth"}"#);
        assert_eq!(adaptive.projection, None);
        assert_eq!(
            adaptive.globe,
            Some(orb_crs::Globe {
                handover: Some([6.0, 7.0]),
                plane: Some(equal_earth),
            })
        );
        let fixed = scene(r#"{"type": "equal-earth", "transition": false}"#);
        assert_eq!((fixed.projection, fixed.globe), (Some(equal_earth), None));
        let utm = scene(r#"{"type": "EPSG:32633", "transition": [8, 9]}"#);
        assert_eq!(utm.projection, None);
        assert_eq!(utm.globe.unwrap().handover, Some([8.0, 9.0]));
        assert!(parse_scene(r#"{"style": "s.json", "projection": {"type": "equal-earth", "transition": [9, 8]}}"#).is_err());
    }

    #[test]
    fn models_and_added_layers_parse_in_the_scene_dialect_only() {
        let scene = parse_scene(
            r#"{"style": "s.json", "models": {"tower": "tower.glb"},
                "layers+": [{"id": "towers", "type": "model", "source": "p",
                             "layout": {"model-id": "tower"}, "after": "roads"}]}"#,
        )
        .unwrap();
        assert_eq!(
            scene.models,
            [("tower".to_string(), "tower.glb".to_string())]
        );
        assert_eq!(scene.layers[0].after.as_deref(), Some("roads"));
        assert!(scene.layers[0].json.get("after").is_none());
        let style = r#"{"version": 8, "models": {"tower": "t.glb"},
            "sources": {"p": {"type": "geojson", "data": {"type": "FeatureCollection", "features": []}}},
            "layers": [{"id": "towers", "type": "model", "source": "p",
                        "layout": {"model-id": "tower"},
                        "paint": {"model-scale": [2, 2, 2], "model-rotation": [0, 0, 90]}}]}"#;
        // v8 has no model layers.
        assert!(crate::parse(style).is_err());
        let parsed = crate::parse_dialect(style, &[], crate::Dialect::Scene).unwrap();
        assert_eq!(parsed.models, [("tower".to_string(), "t.glb".to_string())]);
        let m = &parsed.layers[0];
        assert_eq!(m.ty, crate::LayerType::Model);
        assert_eq!(
            m.get(crate::MODEL_ID).evaluate(0.0),
            Some("tower".to_string())
        );
        assert_eq!(m.get(crate::MODEL_SCALE).evaluate(0.0), [2.0; 3]);
        assert_eq!(m.get(crate::MODEL_ROTATION).evaluate(0.0), [0.0, 0.0, 90.0]);
    }

    #[test]
    fn inline_styles_and_errors() {
        let scene =
            parse_scene(r#"{"style": {"version": 8, "sources": {}, "layers": []}}"#).unwrap();
        assert_eq!(
            scene.style,
            StyleRef::Inline(r#"{"version": 8, "sources": {}, "layers": []}"#.into())
        );
        assert!(!is_scene(r#"{"version": 8, "style": 1}"#));
        let errors = parse_scene(
            r#"{"style": "s.json", "projection": {"type": "EPSG:28992"},
                "sources": {"a": {"tileMatrixSet": "GNOSISGlobalGrid"}}, "surface": "sphere"}"#,
        )
        .unwrap_err();
        let paths: Vec<_> = errors.errors.iter().map(|e| e.path.as_str()).collect();
        assert_eq!(paths, ["projection", "sources.a.tileMatrixSet", "surface"]);
    }
}
