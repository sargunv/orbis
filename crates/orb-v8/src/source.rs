//! Sources.

use orb_expr::json::Json;
use orb_style::*;

use crate::Cx;

const SOURCE_TYPES: &[&str] = &[
    "vector",
    "raster",
    "raster-dem",
    "geojson",
    "video",
    "image",
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
        "image" | "video" => SourceKind::Image(parse_image(cx, json, path, ty == "video")?),
        _ => unreachable!("checked against SOURCE_TYPES"),
    };
    Some(Source {
        id: id.to_string(),
        kind,
    })
}

fn parse_image(cx: &mut Cx, json: &Json, path: &str, video: bool) -> Option<ImageSource> {
    let key = if video { "urls" } else { "url" };
    let url_path = format!("{path}.{key}");
    let Some(url) = json.get(key) else {
        cx.error(
            path,
            json.span,
            format!("missing required property \"{key}\""),
        );
        return None;
    };
    let content = if video {
        let items = cx.array(url, &url_path)?;
        let urls = items
            .iter()
            .enumerate()
            .map(|(i, item)| cx.string(item, &format!("{url_path}[{i}]")))
            .collect::<Option<Vec<_>>>()?;
        ImageContent::Video(urls)
    } else {
        ImageContent::Image(cx.string(url, &url_path)?)
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

fn parse_geojson_source(cx: &mut Cx, json: &Json, path: &str) -> Option<GeoJsonSource> {
    let errors = cx.diagnostics.errors.len();
    let at = |key: &str| format!("{path}.{key}");
    let Some(data) = json.get("data") else {
        cx.error(at("data"), json.span, "GeoJSON source needs data");
        return None;
    };
    let data = if let Some(url) = data.as_str() {
        GeoJsonData::Url(url.to_string())
    } else {
        match crate::geojson::convert(data) {
            Ok(features) => GeoJsonData::Inline(features),
            Err(e) => {
                cx.error(at("data"), data.span, e);
                return None;
            }
        }
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
    for key in ["maxzoom", "indexMaxZoom", "indexMaxPoints"] {
        if let Some(v) = json.get(key)
            && let Some(n) = cx.number(v, &at(key))
        {
            let max = if key == "indexMaxPoints" {
                usize::MAX as f64
            } else {
                24.0
            };
            if n < 0.0 || n > max || n.fract() != 0.0 {
                cx.error(at(key), v.span, format!("expected an integer in 0..={max}"));
            } else {
                match key {
                    "maxzoom" => options.max_zoom = n as u8,
                    "indexMaxZoom" => options.index_max_zoom = n as u8,
                    _ => options.index_max_points = n as usize,
                }
            }
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
    let cluster = json
        .get("cluster")
        .and_then(|v| match v.as_bool() {
            Some(b) => Some(b),
            None => {
                cx.error(at("cluster"), v.span, "expected a boolean");
                None
            }
        })
        .unwrap_or(false);
    if cluster {
        cx.warn(
            at("cluster"),
            json.span,
            "GeoJSON clustering is not supported yet; the source is skipped",
        );
    }
    if let Some(v) = json.get("filter") {
        cx.warn(
            at("filter"),
            v.span,
            "GeoJSON source filters are not supported yet",
        );
    }
    (cx.diagnostics.errors.len() == errors).then_some(GeoJsonSource {
        data,
        options,
        cluster,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parse(source: &str) -> Result<crate::Scene, crate::Diagnostics> {
        crate::parse(&format!(
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
        let style = parse(r#"{"type":"geojson","data":"data.json","maxzoom":12,"tolerance":1.25,"buffer":32,"lineMetrics":true,"generateId":true,"indexMaxZoom":3,"indexMaxPoints":17}"#).unwrap();
        let SourceKind::GeoJson(s) = &style.sources[0].kind else {
            panic!()
        };
        assert_eq!(s.data, GeoJsonData::Url("data.json".into()));
        assert_eq!(s.options.max_zoom, 12);
        assert_eq!(s.options.tolerance, 20.0);
        assert_eq!(s.options.buffer, 512.0);
        assert_eq!(s.options.extent, 8192);
        assert!(s.options.line_metrics && s.options.generate_id);
        assert_eq!(s.options.index_max_zoom, 3);
        assert_eq!(s.options.index_max_points, 17);
        let style = parse(r#"{"type":"geojson","data":"data.json","promoteId":{"_geojsonTileLayer":"name"},"cluster":true}"#).unwrap();
        let SourceKind::GeoJson(s) = &style.sources[0].kind else {
            panic!()
        };
        assert_eq!(s.options.promote_id.as_deref(), Some("name"));
        assert!(s.cluster);
        assert!(
            style
                .warnings
                .iter()
                .any(|w| w.message.contains("source is skipped"))
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
        assert!(parse(r#"{"type":"geojson"}"#).is_err());
    }
}
