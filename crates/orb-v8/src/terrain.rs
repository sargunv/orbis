//! The style's `terrain`: the `raster-dem` source that raises the map's
//! surface, and how far.
use crate::{Cx, Source, SourceKind};
use orb_expr::json::Json;
use orb_style::*;

pub(crate) fn parse(
    cx: &mut Cx,
    json: &Json,
    sources: &[std::sync::Arc<Source>],
) -> Option<Terrain> {
    let members = cx.object(json, "terrain")?;
    for m in members {
        if !matches!(m.key.as_str(), "source" | "exaggeration") {
            cx.warn(
                format!("terrain.{}", m.key),
                m.key_span,
                format!("unknown terrain property \"{}\"", m.key),
            );
        }
    }
    let Some(source_json) = json.get("source") else {
        cx.error("terrain", json.span, "missing required property \"source\"");
        return None;
    };
    let source = cx.string(source_json, "terrain.source")?;
    match sources.iter().find(|s| s.id == source).map(|s| &s.kind) {
        Some(SourceKind::RasterDem(_)) => {}
        Some(_) => {
            cx.error(
                "terrain.source",
                source_json.span,
                format!("terrain cannot be used with a source of a type other than raster-dem: \"{source}\""),
            );
            return None;
        }
        None => {
            cx.error(
                "terrain.source",
                source_json.span,
                format!("source \"{source}\" not found"),
            );
            return None;
        }
    }
    let exaggeration = match json.get("exaggeration") {
        None => Terrain::DEFAULT_EXAGGERATION,
        Some(e) => cx.number_in(e, "terrain.exaggeration", 0.0, f64::INFINITY)?,
    };
    Some(Terrain {
        source,
        exaggeration,
    })
}

#[cfg(test)]
mod tests {
    fn parse(terrain: &str) -> Result<crate::Scene, crate::Diagnostics> {
        crate::parse(&format!(
            r#"{{"version":8,"sources":{{"dem":{{"type":"raster-dem","tiles":["x/{{z}}/{{x}}/{{y}}.png"]}},"v":{{"type":"vector","tiles":["x"]}}}},"layers":[],"terrain":{terrain}}}"#
        ))
    }

    #[test]
    fn terrain_names_a_raster_dem_source() {
        let style = parse(r#"{"source":"dem","exaggeration":1.5}"#).unwrap();
        let terrain = style.terrain.unwrap();
        assert_eq!(terrain.source, "dem");
        assert_eq!(terrain.exaggeration, 1.5);
        assert!(style.warnings.is_empty(), "{:?}", style.warnings);
        assert_eq!(
            parse(r#"{"source":"dem"}"#)
                .unwrap()
                .terrain
                .unwrap()
                .exaggeration,
            1.0
        );
        for bad in [
            r#"{"source":"v"}"#,
            r#"{"source":"missing"}"#,
            r#"{"exaggeration":1}"#,
            r#"{"source":"dem","exaggeration":-1}"#,
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }
}
