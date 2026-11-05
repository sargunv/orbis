//! The root `font-faces` property: font files a style declares for the
//! names its `text-font` stacks use, each covering some code points (GL JS
//! `FontFaceManager`).

use orb_expr::json::Json;
use orb_style::*;

use crate::Cx;

const MAX_CODE_POINT: u32 = 0x10FFFF;

/// One CSS `unicode-range` entry: `U+A5`, `U+0-10FFFF` or `U+4??`.
fn unicode_range(value: &str) -> Option<[u32; 2]> {
    let rest = value
        .strip_prefix("U+")
        .or_else(|| value.strip_prefix("u+"))?;
    let hex = |s: &str| {
        (!s.is_empty() && s.len() <= 6 && s.bytes().all(|b| b.is_ascii_hexdigit()))
            .then(|| u32::from_str_radix(s, 16).ok())
            .flatten()
    };
    let prefix = rest.trim_end_matches('?');
    let (start, end) = if prefix.len() < rest.len() {
        let marks = rest.len() - prefix.len();
        (
            hex(&format!("{prefix}{}", "0".repeat(marks)))?,
            hex(&format!("{prefix}{}", "f".repeat(marks)))?,
        )
    } else {
        match rest.split_once('-') {
            Some((a, b)) => (hex(a)?, hex(b)?),
            None => (hex(rest)?, hex(rest)?),
        }
    };
    (start <= end && start <= MAX_CODE_POINT).then_some([start, end.min(MAX_CODE_POINT)])
}

/// Parses `font-faces`. Entries GL JS ignores (no URL, no valid range) are
/// dropped with a warning.
pub(crate) fn parse(cx: &mut Cx, json: Option<&Json>) -> FontFaces {
    let Some(json) = json else {
        return Vec::new();
    };
    let Some(members) = cx.object(json, "font-faces") else {
        return Vec::new();
    };
    let mut faces = Vec::new();
    for m in members {
        let path = format!("font-faces.{}", m.key);
        let declarations = match m.value.as_array() {
            Some(items) => items.iter().collect(),
            None => vec![&m.value],
        };
        let mut declared = Vec::new();
        for (i, d) in declarations.into_iter().enumerate() {
            let at = format!("{path}[{i}]");
            let (url, ranges) = match d.as_str() {
                Some(url) => (Some(url.to_string()), None),
                None if d.as_object().is_some() => (
                    d.get("url").and_then(Json::as_str).map(str::to_string),
                    d.get("unicode-range"),
                ),
                None => (None, None),
            };
            let Some(url) = url else {
                cx.warn(
                    &at,
                    d.span,
                    format!(
                        "Ignoring the font face declared for \"{}\": it has no URL.",
                        m.key
                    ),
                );
                continue;
            };
            let ranges = match ranges.and_then(Json::as_array) {
                None | Some([]) => vec![[0, MAX_CODE_POINT]],
                Some(entries) => {
                    let mut ranges = Vec::new();
                    for entry in entries {
                        match entry.as_str().and_then(unicode_range) {
                            Some(range) => ranges.push(range),
                            None => cx.warn(
                                format!("{at}.unicode-range"),
                                entry.span,
                                format!(
                                    "Ignoring the unicode range {entry} of the font face at {url}: it is not a valid range."
                                ),
                            ),
                        }
                    }
                    if ranges.is_empty() {
                        continue;
                    }
                    ranges
                }
            };
            declared.push(FontFace { url, ranges });
        }
        faces.push((m.key.clone(), declared));
    }
    faces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_ranges_follow_css() {
        assert_eq!(unicode_range("U+A5"), Some([0xA5, 0xA5]));
        assert_eq!(unicode_range("U+0000-024F"), Some([0, 0x24F]));
        assert_eq!(unicode_range("u+4??"), Some([0x400, 0x4FF]));
        assert_eq!(unicode_range("U+0-10FFFF"), Some([0, MAX_CODE_POINT]));
        assert_eq!(unicode_range("U+20-1FFFFF"), Some([0x20, MAX_CODE_POINT]));
        for bad in ["A5", "U+", "U+24F-20", "U+1234567", "U+??????? ", "U+G"] {
            assert_eq!(unicode_range(bad), None, "{bad}");
        }
    }

    #[test]
    fn parses_declarations_like_gl_js() {
        let style = crate::parse(
            r#"{"version":8,"sources":{},"layers":[],"font-faces":{
                "A":"a.ttf",
                "B":[{"url":"b.woff2","unicode-range":["U+0-7F","bad"]},{"unicode-range":["U+80"]}],
                "C":{"url":"c.otf","unicode-range":["nope"]}
            }}"#,
        )
        .unwrap();
        assert_eq!(
            style.font_faces,
            vec![
                (
                    "A".into(),
                    vec![FontFace {
                        url: "a.ttf".into(),
                        ranges: vec![[0, MAX_CODE_POINT]]
                    }]
                ),
                (
                    "B".into(),
                    vec![FontFace {
                        url: "b.woff2".into(),
                        ranges: vec![[0, 0x7F]]
                    }]
                ),
                ("C".into(), vec![]),
            ]
        );
        assert_eq!(style.warnings.len(), 3, "{:?}", style.warnings);
    }
}
