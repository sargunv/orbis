//! Diagnostics, rather than fixture names, decide renderer support.

use std::collections::BTreeSet;

use orb_style::Style;

/// Diagnostic locations contain source ids and layer indices; the message's
/// first quoted spec name gives a short, stable aggregation key instead.
pub(crate) fn parse_reason(message: &str) -> String {
    let name = message.split('"').nth(1).unwrap_or("diagnostic");
    let category = if message.starts_with("layer type") {
        "layer"
    } else if message.starts_with("Expression") {
        "expression"
    } else {
        "property"
    };
    format!("{category}:{name}")
}

pub(crate) fn runtime_reason(message: &str) -> Option<&'static str> {
    if message.contains(orb_runtime::NO_VIDEO) {
        Some("host:video-decoder")
    } else if message.contains("only vector, GeoJSON, raster and raster-dem sources") {
        Some("source:unsupported")
    } else if message.contains("need a source-layer") {
        Some("source-layer")
    } else {
        None
    }
}

/// Whether a runtime warning is one GL JS gives too for the same input,
/// which a test may provoke on purpose (a glyph range that 404s, drawn
/// locally instead; GeoJSON data that isn't GeoJSON, which leaves its
/// source empty).
pub(crate) fn mirrors_gl_js(message: &str) -> bool {
    message.contains("its glyphs are drawn with local fonts instead") || message.contains("GeoJSON")
}

/// The skip reason of a parser warning that an edit raised at runtime:
/// `layers[3].type: layer type "symbol" is not supported yet; …`.
pub(crate) fn edit_reason(message: &str) -> Option<String> {
    let (path, message) = message.split_once(": ")?;
    let parsed = path.starts_with("layers[") || path.starts_with("sources.");
    (parsed && message.contains("not supported yet")).then(|| parse_reason(message))
}

pub(crate) fn reasons(style: &Style) -> BTreeSet<String> {
    style
        .warnings
        .iter()
        .filter(|w| {
            // Informational, and bad GeoJSON data, which leaves its source
            // empty, not something Orbis doesn't support.
            !w.message
                .starts_with("GeoJSON nested object and array property values")
                && !w.message.ends_with("draws nothing")
        })
        .map(|w| parse_reason(&w.message))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasons_do_not_contain_fixture_ids_or_layer_indices() {
        assert_eq!(
            parse_reason("layer type \"symbol\" is not supported yet; the layer is skipped"),
            "layer:symbol"
        );
        assert_eq!(
            parse_reason("\"line-dasharray\" is not rendered yet"),
            "property:line-dasharray"
        );
        assert_eq!(runtime_reason("source water: not found"), None);
    }
}
