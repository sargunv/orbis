//! The root `font-faces` property: font files a style declares for the
//! names its `text-font` stacks use, each covering some code points (GL JS
//! `FontFaceManager`).

use crate::*;
use orb_expr::json::Json;

use crate::v8::Cx;

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

/// Parses a list of font files, each `{"url", "unicode-range"?}` as a
/// font face of `font-faces` declares it (MapLibre Native's
/// `fallbackFontFaces`), refusing what isn't one rather than dropping it:
/// a host's option, not a style to draw as best it can.
pub fn parse_font_files(text: &str) -> Result<Vec<FontFace>, String> {
    let json = orb_expr::json::parse(text).map_err(|e| e.message)?;
    let items = json.as_array().ok_or("not an array of font files")?;
    let mut files = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let url = item
            .get("url")
            .and_then(Json::as_str)
            .ok_or_else(|| format!("font file {i} has no URL"))?;
        let ranges = match item.get("unicode-range") {
            None => vec![[0, MAX_CODE_POINT]],
            Some(ranges) => {
                let entries = ranges
                    .as_array()
                    .ok_or_else(|| format!("font file {i}'s unicode-range isn't an array"))?;
                let parsed = entries
                    .iter()
                    .map(|e| e.as_str().and_then(unicode_range))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| format!("font file {i} has an invalid unicode range"))?;
                match parsed.is_empty() {
                    true => vec![[0, MAX_CODE_POINT]],
                    false => parsed,
                }
            }
        };
        files.push(FontFace {
            url: url.to_string(),
            ranges,
        });
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_files_parse_strictly() {
        let files = parse_font_files(
            r#"[{"url":"a.ttf","unicode-range":["U+1000-109F"]},{"url":"b.otf"}]"#,
        )
        .unwrap();
        assert_eq!(files[0].ranges, vec![[0x1000, 0x109F]]);
        assert_eq!(files[1].ranges, vec![[0, MAX_CODE_POINT]]);
        assert_eq!(parse_font_files("[]").unwrap(), Vec::new());
        for bad in [
            "{}",
            r#"[{"unicode-range":["U+1000-109F"]}]"#,
            r#"[{"url":"a.ttf","unicode-range":["U+XYZ-FF"]}]"#,
            "not json",
        ] {
            assert!(parse_font_files(bad).is_err(), "{bad}");
        }
    }

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
        let style = crate::v8::parse(
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
