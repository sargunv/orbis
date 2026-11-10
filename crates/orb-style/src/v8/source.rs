//! Sources.

use crate::*;
use orb_crs::{TileMatrix, TileMatrixSet, TileMatrixSetDefinition};
use orb_expr::json::{Json, JsonValue};

use crate::v8::Cx;

const SOURCE_TYPES: &[&str] = &[
    "vector",
    "raster",
    "raster-dem",
    "geojson",
    "video",
    "image",
    "canvas",
];

pub(crate) fn parse_sources(cx: &mut Cx, json: &Json) -> Vec<Source> {
    let Some(members) = cx.object(json, "sources") else {
        return Vec::new();
    };
    members
        .iter()
        .filter_map(|m| {
            let path = format!("sources.{}", m.key);
            parse_source(cx, &m.key, &m.value, &path)
        })
        .collect()
}

pub(crate) fn parse_source(cx: &mut Cx, id: &str, json: &Json, path: &str) -> Option<Source> {
    cx.object(json, path)?;
    let Some(ty) = json.get("type") else {
        cx.error(path, json.span, "missing required property \"type\"");
        return None;
    };
    let type_path = format!("{path}.type");
    let ty = match ty.as_str() {
        Some(t) if SOURCE_TYPES.contains(&t) => t,
        _ => {
            cx.error(
                type_path,
                ty.span,
                format!("expected one of [{}], {ty} found", SOURCE_TYPES.join(", ")),
            );
            return None;
        }
    };
    let kind = match ty {
        "vector" => SourceKind::Vector(parse_vector(cx, json, path)?),
        "geojson" => SourceKind::GeoJson(parse_geojson_source(cx, json, path)?),
        "raster" => SourceKind::Raster(parse_raster(cx, json, path)?),
        "raster-dem" => SourceKind::RasterDem(parse_raster_dem(cx, json, path)?),
        "image" | "video" | "canvas" => SourceKind::Image(parse_image(cx, json, path, ty)?),
        _ => unreachable!("checked against SOURCE_TYPES"),
    };
    let key = Preview::TileMatrixSet.name();
    let tms_path = format!("{path}.{key}");
    let tile_matrix_set = match json.get(key).filter(|tms| {
        cx.preview(
            Preview::TileMatrixSet,
            &tms_path,
            tms.span,
            format!("\"{key}\""),
        )
    }) {
        None => None,
        // GeoJSON is geodetic: Orbis tiles it on the map's own plane.
        Some(tms) if ty == "geojson" => {
            cx.warn(
                &tms_path,
                tms.span,
                format!("a GeoJSON source tiles on the map's plane; its \"{key}\" is ignored"),
            );
            None
        }
        Some(tms) => Some(parse_tms(cx, tms, &tms_path)?),
    };
    Some(Source {
        id: id.to_string(),
        kind,
        tile_matrix_set,
    })
}

/// A source's `dev.sargunv.orbis:tileMatrixSet`: an identifier `orb_crs`
/// registers or an
/// inline OGC 2D-TMS document (version 2.0, or 1.0's `identifier` and
/// `topLeftCorner`).
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

/// An `image`, `video` or `canvas` source (GL JS's `canvas` source: its
/// `animate` doesn't apply, as the host hands in each frame).
fn parse_image(cx: &mut Cx, json: &Json, path: &str, ty: &str) -> Option<ImageSource> {
    let key = match ty {
        "video" => "urls",
        "canvas" => "canvas",
        _ => "url",
    };
    let url_path = format!("{path}.{key}");
    let Some(url) = json.get(key) else {
        cx.error(
            path,
            json.span,
            format!("missing required property \"{key}\""),
        );
        return None;
    };
    let content = match ty {
        "video" => {
            let items = cx.array(url, &url_path)?;
            let urls = items
                .iter()
                .enumerate()
                .map(|(i, item)| cx.string(item, &format!("{url_path}[{i}]")))
                .collect::<Option<Vec<_>>>()?;
            ImageContent::Video(urls)
        }
        "canvas" => ImageContent::Canvas(cx.string(url, &url_path)?),
        _ => ImageContent::Image(cx.string(url, &url_path)?),
    };
    let coordinates_path = format!("{path}.coordinates");
    let Some(coordinates) = json.get("coordinates") else {
        cx.error(path, json.span, "missing required property \"coordinates\"");
        return None;
    };
    let corners = cx.array(coordinates, &coordinates_path)?;
    if corners.len() != 4 {
        cx.error(
            coordinates_path,
            coordinates.span,
            format!("array length 4 expected, length {} found", corners.len()),
        );
        return None;
    }
    let mut out = [[0.0; 2]; 4];
    for (i, corner) in corners.iter().enumerate() {
        out[i] = cx.numbers::<2>(corner, &format!("{coordinates_path}[{i}]"))?;
    }
    Some(ImageSource {
        content,
        coordinates: out,
    })
}

/// The [`TileSet`] keys of a tiled source; `None` if any is invalid.
fn parse_tileset(cx: &mut Cx, json: &Json, path: &str, ty: &str) -> Option<TileSet> {
    let errors = cx.diagnostics.errors.len();
    let at = |key: &str| format!("{path}.{key}");
    let url = json.get("url").and_then(|u| cx.string(u, &at("url")));
    let tiles = json.get("tiles").and_then(|t| {
        let items = cx.array(t, &at("tiles"))?;
        items
            .iter()
            .enumerate()
            .map(|(i, item)| cx.string(item, &format!("{}[{i}]", at("tiles"))))
            .collect::<Option<Vec<_>>>()
    });
    let minzoom = json
        .get("minzoom")
        .and_then(|z| cx.number(z, &at("minzoom")));
    let maxzoom = json
        .get("maxzoom")
        .and_then(|z| cx.number(z, &at("maxzoom")));
    let scheme = json
        .get("scheme")
        .and_then(|s| cx.enumeration::<Scheme>(s, &at("scheme")));
    let bounds = json
        .get("bounds")
        .and_then(|b| cx.numbers::<4>(b, &at("bounds")));
    let attribution = json
        .get("attribution")
        .and_then(|a| cx.string(a, &at("attribution")));
    if url.is_none() && tiles.is_none() {
        cx.warn(
            path,
            json.span,
            format!("{ty} source has neither \"url\" nor \"tiles\""),
        );
    }
    (cx.diagnostics.errors.len() == errors).then_some(TileSet {
        url,
        tiles,
        minzoom,
        maxzoom,
        scheme,
        bounds,
        attribution,
    })
}

fn parse_vector(cx: &mut Cx, json: &Json, path: &str) -> Option<VectorSource> {
    let errors = cx.diagnostics.errors.len();
    let tileset = parse_tileset(cx, json, path, "vector");
    let encoding = json
        .get("encoding")
        .and_then(|e| cx.enumeration::<Encoding>(e, &format!("{path}.encoding")));
    let promote_id = json.get("promoteId").and_then(|p| {
        let at = format!("{path}.promoteId");
        match p.as_object() {
            Some(members) => members
                .iter()
                .map(|m| {
                    Some((
                        m.key.clone(),
                        cx.string(&m.value, &format!("{at}.{}", m.key))?,
                    ))
                })
                .collect::<Option<_>>()
                .map(PromoteId::PerLayer),
            None => cx.string(p, &at).map(PromoteId::All),
        }
    });
    let tileset = tileset?;
    (cx.diagnostics.errors.len() == errors).then_some(VectorSource {
        tileset,
        encoding,
        promote_id,
    })
}

fn parse_raster(cx: &mut Cx, json: &Json, path: &str) -> Option<RasterSource> {
    let ty = json.get("type").and_then(Json::as_str).unwrap_or("raster");
    let tileset = parse_tileset(cx, json, path, ty)?;
    let tile_size = match json.get("tileSize") {
        None => None,
        Some(v) => Some(cx.number_in(v, &format!("{path}.tileSize"), 1.0, 8192.0)?),
    };
    Some(RasterSource { tileset, tile_size })
}

fn parse_raster_dem(cx: &mut Cx, json: &Json, path: &str) -> Option<RasterDemSource> {
    let errors = cx.diagnostics.errors.len();
    let raster = parse_raster(cx, json, path);
    let encoding = json
        .get("encoding")
        .and_then(|e| cx.enumeration::<DemEncoding>(e, &format!("{path}.encoding")));
    let mut factors = [1.0, 1.0, 1.0, 0.0];
    for (key, factor) in ["redFactor", "greenFactor", "blueFactor", "baseShift"]
        .into_iter()
        .zip(&mut factors)
    {
        if let Some(v) = json.get(key)
            && let Some(n) = cx.number(v, &format!("{path}.{key}"))
        {
            *factor = n;
        }
    }
    let raster = raster?;
    (cx.diagnostics.errors.len() == errors).then_some(RasterDemSource {
        raster,
        encoding,
        factors,
    })
}

/// How a warning about a GeoJSON source's missing or invalid data ends: the
/// source loads empty and the rest of the style draws.
pub const EMPTY_SOURCE: &str = "the source draws nothing";

fn parse_geojson_source(cx: &mut Cx, json: &Json, path: &str) -> Option<GeoJsonSource> {
    let errors = cx.diagnostics.errors.len();
    let at = |key: &str| format!("{path}.{key}");
    // Missing or invalid inline data leaves the source empty and the rest
    // of the style drawing, as MapLibre Native does (GL JS rejects the
    // style).
    let data = match json.get("data") {
        None => {
            cx.warn(
                at("data"),
                json.span,
                format!("GeoJSON source needs data; {EMPTY_SOURCE}"),
            );
            GeoJsonData::Inline(Vec::new())
        }
        Some(data) => match data.as_str() {
            Some(url) => GeoJsonData::Url(url.to_string()),
            None => GeoJsonData::Inline(crate::v8::geojson::convert(data).unwrap_or_else(|e| {
                cx.warn(at("data"), data.span, format!("{e}; {EMPTY_SOURCE}"));
                Vec::new()
            })),
        },
    };
    let mut options = orb_tile::geojson::Options::default();
    for (key, target, scale, max) in [
        ("tolerance", &mut options.tolerance, 16.0, f64::MAX / 16.0),
        ("buffer", &mut options.buffer, 16.0, 512.0),
    ] {
        if let Some(v) = json.get(key)
            && let Some(n) = cx.number(v, &at(key))
        {
            if n < 0.0 || n > max {
                cx.error(at(key), v.span, format!("expected a number in 0..={max}"));
            } else {
                *target = n * scale;
            }
        }
    }
    if let Some(v) = json.get("maxzoom")
        && let Some(n) = cx.number(v, &at("maxzoom"))
    {
        if !(0.0..=24.0).contains(&n) || n.fract() != 0.0 {
            cx.error(at("maxzoom"), v.span, "expected an integer in 0..=24");
        } else {
            options.max_zoom = n as u8;
        }
    }
    for (key, target) in [
        ("lineMetrics", &mut options.line_metrics),
        ("generateId", &mut options.generate_id),
    ] {
        if let Some(v) = json.get(key) {
            match v.as_bool() {
                Some(b) => *target = b,
                None => cx.error(at(key), v.span, "expected a boolean"),
            }
        }
    }
    if let Some(v) = json.get("promoteId") {
        if let Some(members) = v.as_object() {
            for m in members {
                let key = format!("{}.{}", at("promoteId"), m.key);
                let value = cx.string(&m.value, &key);
                if m.key == orb_tile::geojson::LAYER_NAME {
                    options.promote_id = value;
                }
            }
        } else {
            options.promote_id = cx.string(v, &at("promoteId"));
        }
    }
    if options.generate_id && options.promote_id.is_some() {
        cx.error(
            path,
            json.span,
            "promoteId and generateId cannot be used together",
        );
    }
    let cluster = match json.get("cluster").map(|v| (v, v.as_bool())) {
        Some((_, Some(true))) => Some(cluster(json, path, options.max_zoom, cx)),
        Some((v, None)) => {
            cx.error(at("cluster"), v.span, "expected a boolean");
            None
        }
        _ => None,
    };
    if let Some(v) = json.get("filter") {
        cx.warn(
            at("filter"),
            v.span,
            "GeoJSON source filters are not supported yet",
        );
    }
    let cluster = cluster.map(|mut c| {
        c.options.generate_id = options.generate_id;
        c
    });
    (cx.diagnostics.errors.len() == errors).then_some(GeoJsonSource {
        data,
        options,
        cluster,
    })
}

/// A clustered source's options, as GL JS gives them to supercluster:
/// `clusterRadius` pixels (50) in tile units, `clusterMaxZoom` (the
/// source's `maxzoom` less one), `clusterMinPoints` (2, at least) and
/// `clusterProperties`.
fn cluster(json: &Json, path: &str, max_zoom: u8, cx: &mut Cx) -> crate::Cluster {
    let at = |key: &str| format!("{path}.{key}");
    let mut options = orb_tile::geojson::ClusterOptions {
        max_zoom: max_zoom.saturating_sub(1),
        ..Default::default()
    };
    let number = |cx: &mut Cx, key: &str, max: f64, integer: bool| {
        let v = json.get(key)?;
        let n = cx.number(v, &at(key))?;
        if n < 0.0 || n > max || (integer && n.fract() != 0.0) {
            let what = if integer { "an integer" } else { "a number" };
            cx.error(at(key), v.span, format!("expected {what} in 0..={max}"));
            return None;
        }
        Some(n)
    };
    if let Some(radius) = number(cx, "clusterRadius", f64::MAX / 16.0, false) {
        options.radius = if radius > 0.0 { radius } else { 50.0 } * 16.0;
    }
    if let Some(zoom) = number(cx, "clusterMaxZoom", 24.0, true) {
        options.max_zoom = zoom as u8;
    }
    if let Some(points) = number(cx, "clusterMinPoints", u32::MAX as f64, true) {
        options.min_points = (points as usize).max(2);
    }
    let mut properties = Vec::new();
    if let Some(v) = json.get("clusterProperties") {
        let Some(members) = v.as_object() else {
            cx.error(at("clusterProperties"), v.span, "expected an object");
            return crate::Cluster {
                options,
                properties,
            };
        };
        for m in members {
            let key = format!("{}.{}", at("clusterProperties"), m.key);
            let parts = m.value.as_array().filter(|a| a.len() == 2);
            let Some([operator, map]) = parts.map(|a| [&a[0], &a[1]]) else {
                cx.error(&key, m.value.span, "expected [operator, map expression]");
                continue;
            };
            // An operator's name folds the accumulated value with the
            // other's: [operator, ["accumulated"], ["get", name]].
            let reduce = match operator.as_str() {
                Some(name) => orb_expr::json::parse(&format!(
                    r#"["{name}", ["accumulated"], ["get", {}]]"#,
                    orb_expr::Value::String(m.key.as_str().into()).to_json_string()
                ))
                .expect("well-formed"),
                None => operator.clone(),
            };
            let mut ok = true;
            for (expression, suffix) in [(map, "[1]"), (&reduce, "[0]")] {
                if let Err(e) = orb_expr::Expression::parse(expression, None) {
                    cx.errors_at(&format!("{key}{suffix}"), e);
                    ok = false;
                }
            }
            if ok {
                properties.push(crate::ClusterProperty {
                    name: m.key.clone(),
                    map: map.clone(),
                    reduce,
                });
            }
        }
    }
    crate::Cluster {
        options,
        properties,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parse(source: &str) -> Result<crate::v8::Style, crate::v8::Diagnostics> {
        crate::v8::parse(&format!(
            r#"{{"version":8,"sources":{{"s":{source}}},"layers":[]}}"#
        ))
    }
    #[test]
    fn geojson_defaults_and_source_options() {
        let style =
            parse(r#"{"type":"geojson","data":{"type":"FeatureCollection","features":[]}}"#)
                .unwrap();
        let SourceKind::GeoJson(s) = &style.sources[0].kind else {
            panic!()
        };
        assert_eq!(s.options, orb_tile::geojson::Options::default());
        assert!(matches!(&s.data, GeoJsonData::Inline(features) if features.is_empty()));
        let style = parse(r#"{"type":"geojson","data":"data.json","maxzoom":12,"tolerance":1.25,"buffer":32,"lineMetrics":true,"generateId":true}"#).unwrap();
        let SourceKind::GeoJson(s) = &style.sources[0].kind else {
            panic!()
        };
        assert_eq!(s.data, GeoJsonData::Url("data.json".into()));
        assert_eq!(s.options.max_zoom, 12);
        assert_eq!(s.options.tolerance, 20.0);
        assert_eq!(s.options.buffer, 512.0);
        assert_eq!(s.options.extent, 8192);
        assert!(s.options.line_metrics && s.options.generate_id);
        let style = parse(r#"{"type":"geojson","data":"data.json","promoteId":{"_geojsonTileLayer":"name"},"cluster":true}"#).unwrap();
        let SourceKind::GeoJson(s) = &style.sources[0].kind else {
            panic!()
        };
        assert_eq!(s.options.promote_id.as_deref(), Some("name"));
        let cluster = s.cluster.as_ref().expect("clustered");
        assert_eq!(cluster.options.max_zoom, 17);
        assert_eq!(cluster.options.radius, 800.0);
        let style = parse(
            r#"{"type":"geojson","data":"data.json","cluster":true,"maxzoom":12,
              "clusterRadius":20,"clusterMinPoints":1,
              "clusterProperties":{"sum":["+",["get","n"]],
                "max":[["max",["accumulated"],["get","max"]],["get","n"]]}}"#,
        )
        .unwrap();
        let SourceKind::GeoJson(s) = &style.sources[0].kind else {
            panic!()
        };
        let cluster = s.cluster.as_ref().expect("clustered");
        assert_eq!(
            (
                cluster.options.max_zoom,
                cluster.options.radius,
                cluster.options.min_points
            ),
            (11, 320.0, 2)
        );
        let names: Vec<_> = cluster.properties.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["sum", "max"]);
        assert_eq!(
            cluster.properties[0].reduce,
            orb_expr::json::parse(r#"["+", ["accumulated"], ["get", "sum"]]"#).unwrap()
        );
        assert!(
            parse(
                r#"{"type":"geojson","data":"x","cluster":true,"clusterProperties":{"a":["+"]}}"#
            )
            .is_err()
        );
    }
    #[test]
    fn invalid_geojson_options_are_errors() {
        for option in [
            r#""maxzoom":25"#,
            r#""maxzoom":1.5"#,
            r#""buffer":-1"#,
            r#""buffer":513"#,
            r#""tolerance":-1"#,
            r#""lineMetrics":0"#,
            r#""generateId":true,"promoteId":"name"#,
        ] {
            assert!(
                parse(&format!(
                    r#"{{"type":"geojson","data":"data.json",{option}}}"#
                ))
                .is_err(),
                "{option}"
            );
        }
    }

    #[test]
    fn missing_or_invalid_inline_geojson_draws_nothing() {
        for source in [
            r#"{"type":"geojson"}"#,
            r#"{"type":"geojson","data":42}"#,
            r#"{"type":"geojson","data":{"type":"Model","geometry":[1,2]}}"#,
        ] {
            let style = parse(source).unwrap_or_else(|e| panic!("{source}: {e:?}"));
            assert_eq!(style.warnings.len(), 1, "{source}");
            let SourceKind::GeoJson(geojson) = &style.sources[0].kind else {
                panic!("{source}");
            };
            assert_eq!(geojson.data, GeoJsonData::Inline(Vec::new()));
        }
    }
}
