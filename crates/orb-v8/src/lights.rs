//! The scene document's lighting module (plan §9.4 and §11.4, decision
//! 0026): lights for the scene's 3D content, in the shape of the style
//! spec's lights proposal (style-spec#1797), and how strongly layers emit
//! their own colour.
//!
//! ```json
//! "lights": [
//!   {"id": "sky", "type": "ambient", "properties": {"color": "#e8f0ff", "intensity": 0.45}},
//!   {"id": "sun", "type": "sun", "properties": {"time": "2025-06-21T15:30:00Z", "intensity": 0.7}}
//! ],
//! "layers": {"buildings": {"emissive-strength": 0.2}}
//! ```
//!
//! A `directional` light takes `direction`, `[azimuth, polar]` in degrees
//! (clockwise from north, from the zenith); a `sun` takes a UTC `time`
//! (RFC 3339) and shines from where the sun stands then over the camera's
//! target. Values are constants.

use orb_expr::Color;
use orb_expr::json::{Json, JsonValue};
use orb_style::*;

use crate::{Cx, Property, TranslateAnchor};

/// The scene document's `lights`: at most one ambient and one directional
/// or sun light.
pub(crate) fn parse_lights(cx: &mut Cx, json: &Json) -> Lights {
    let mut lights = Lights {
        ambient: None,
        directional: None,
        terrain: true,
        emissive: Vec::new(),
    };
    let Some(items) = json.as_array() else {
        cx.error("lights", json.span, "expected an array of lights");
        return lights;
    };
    for (i, item) in items.iter().enumerate() {
        let path = format!("lights[{i}]");
        let Some(members) = cx.object(item, &path) else {
            continue;
        };
        let (mut kind, mut properties) = (None, None);
        for m in members {
            match (m.key.as_str(), &m.value.value) {
                ("id", JsonValue::String(_)) => {}
                ("type", JsonValue::String(s)) => kind = Some((s.clone(), m.value.span)),
                ("properties", JsonValue::Object(_)) => properties = Some(&m.value),
                (key, _) => cx.warn(
                    &path,
                    m.key_span,
                    format!("unknown or invalid property \"{key}\""),
                ),
            }
        }
        let Some((kind, kind_span)) = kind else {
            cx.error(&path, item.span, "a light needs a \"type\"");
            continue;
        };
        if !["ambient", "directional", "sun"].contains(&kind.as_str()) {
            cx.error(&path, kind_span, format!("unknown light type \"{kind}\""));
            continue;
        }
        let (mut color, mut intensity) = (Color::parse("white").expect("white"), 0.5);
        let (mut direction, mut time) = ([210.0, 30.0], f64::NAN);
        let props = properties
            .and_then(|p| cx.object(p, &format!("{path}.properties")))
            .unwrap_or(&[]);
        for m in props {
            let key = format!("{path}.properties.{}", m.key);
            match (m.key.as_str(), &m.value.value) {
                ("color", JsonValue::String(s)) => match Color::parse(s) {
                    Some(c) => color = c,
                    None => cx.error(&key, m.value.span, "expected a colour"),
                },
                ("intensity", JsonValue::Number(n)) if (0.0..=1.0).contains(n) => {
                    intensity = *n;
                }
                ("direction", JsonValue::Array(a)) if kind == "directional" => match a.as_slice() {
                    [x, y] => match (&x.value, &y.value) {
                        (JsonValue::Number(x), JsonValue::Number(y))
                            if (0.0..=90.0).contains(y) =>
                        {
                            direction = [*x, *y];
                        }
                        _ => cx.error(&key, m.value.span, "expected [azimuth, polar ≤ 90]"),
                    },
                    _ => cx.error(&key, m.value.span, "expected [azimuth, polar]"),
                },
                ("time", JsonValue::String(s)) if kind == "sun" => match parse_time(s) {
                    Some(t) => time = t,
                    None => cx.error(&key, m.value.span, "expected an RFC 3339 time"),
                },
                (other, _) => cx.warn(
                    &key,
                    m.key_span,
                    format!("unknown or invalid property \"{other}\""),
                ),
            }
        }
        let color = Property::constant(color);
        let level = Level::declared(intensity as f32);
        if kind == "ambient" {
            if lights.ambient.is_some() {
                cx.warn(&path, item.span, "only the first ambient light shines");
                continue;
            }
            lights.ambient = Some(AmbientLight { color, level });
            continue;
        }
        if kind == "sun" && time.is_nan() {
            cx.error(&path, item.span, "a sun light needs a \"time\"");
            continue;
        }
        if lights.directional.is_some() {
            cx.warn(
                &path,
                item.span,
                "only the first directional or sun light shines",
            );
            continue;
        }
        lights.directional = Some(DirectionalLight {
            color,
            level,
            anchor: Property::constant(TranslateAnchor::Map),
            direction: if kind == "sun" {
                Direction::Sun { time }
            } else {
                Direction::Position(Property::constant([
                    1.0,
                    direction[0] as f32,
                    direction[1] as f32,
                ]))
            },
        });
    }
    lights
}

/// `layers`: per layer id, its `emissive-strength`.
pub(crate) fn parse_layers(cx: &mut Cx, json: &Json) -> Vec<(String, f64)> {
    let mut emissive = Vec::new();
    for layer in cx.object(json, "layers").unwrap_or(&[]) {
        let path = format!("layers.{}", layer.key);
        for m in cx.object(&layer.value, &path).unwrap_or(&[]) {
            match (m.key.as_str(), &m.value.value) {
                ("emissive-strength", JsonValue::Number(n)) if (0.0..=1.0).contains(n) => {
                    emissive.push((layer.key.clone(), *n));
                }
                (key, _) => cx.warn(
                    &path,
                    m.key_span,
                    format!("unknown or invalid property \"{key}\""),
                ),
            }
        }
    }
    emissive
}

/// Seconds since the Unix epoch of an RFC 3339 date-time:
/// `YYYY-MM-DDTHH:MM[:SS[.frac]]` then `Z` or `±HH:MM`.
pub fn parse_time(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    let num = |r: std::ops::Range<usize>| -> Option<i64> {
        let part = s.get(r)?;
        part.bytes()
            .all(|c| c.is_ascii_digit())
            .then(|| part.parse().ok())?
    };
    if b.len() < 16
        || b[4] != b'-'
        || b[7] != b'-'
        || !matches!(b[10], b'T' | b't' | b' ')
        || b[13] != b':'
    {
        return None;
    }
    let (year, month, day) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (hour, minute) = (num(11..13)?, num(14..16)?);
    let mut rest = &s[16..];
    let mut seconds = 0.0;
    if let Some(after) = rest.strip_prefix(':') {
        let end = after
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(after.len());
        seconds = after[..end].parse().ok()?;
        rest = &after[end..];
    }
    let offset = match rest {
        "Z" | "z" => 0,
        _ if rest.len() == 6
            && matches!(rest.as_bytes()[0], b'+' | b'-')
            && rest.as_bytes()[3] == b':' =>
        {
            let sign = if rest.starts_with('-') { -1 } else { 1 };
            sign * (rest[1..3].parse::<i64>().ok()? * 60 + rest[4..6].parse::<i64>().ok()?)
        }
        _ => return None,
    };
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || seconds >= 61.0
    {
        return None;
    }
    // Days from the civil date (Howard Hinnant's algorithm).
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some((days * 86_400 + hour * 3600 + (minute - offset) * 60) as f64 + seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_parse_to_unix_seconds() {
        assert_eq!(parse_time("1970-01-01T00:00:00Z"), Some(0.0));
        assert_eq!(parse_time("2000-03-01T12:30Z"), Some(951_913_800.0));
        assert_eq!(parse_time("2000-03-01T14:30:00+02:00"), Some(951_913_800.0));
        assert_eq!(parse_time("2024-06-21T12:00:00.5Z"), Some(1_718_971_200.5));
        assert_eq!(parse_time("2024-06-21"), None);
        assert_eq!(parse_time("2024-13-01T00:00Z"), None);
    }

    #[test]
    fn scenes_carry_lights_and_emissive_layers() {
        let scene = crate::parse_scene(
            r##"{"style": "s.json",
                "lights": [
                    {"id": "a", "type": "ambient", "properties": {"color": "#ff0000", "intensity": 0.25}},
                    {"type": "directional", "properties": {"direction": [90, 45]}},
                    {"type": "sun", "properties": {"time": "2024-06-21T12:00:00Z"}}
                ],
                "layers": {"b": {"emissive-strength": 0.5}}}"##,
        )
        .unwrap();
        let l = scene.lights.unwrap();
        let ambient = l.ambient.clone().unwrap();
        assert_eq!(ambient.level.evaluate(0.0), 0.25);
        assert_eq!(ambient.color.evaluate(0.0).straight(), [1.0, 0.0, 0.0, 1.0]);
        let directional = l.directional.clone().unwrap();
        let Direction::Position(position) = directional.direction else {
            panic!("the first directional light shines");
        };
        assert_eq!(position.evaluate(0.0), [1.0, 90.0, 45.0]);
        assert!(l.terrain);
        assert_eq!(l.emissive("b"), 0.5);
        assert_eq!(l.emissive("c"), 0.0);
        assert!(
            scene
                .warnings
                .iter()
                .any(|w| w.to_string().contains("only the first"))
        );
        assert!(crate::parse_scene(r#"{"style": "s.json", "lights": [{"type": "sun"}]}"#).is_err());
        assert!(
            crate::parse_scene(r#"{"style": "s.json", "lights": [{"type": "spot"}]}"#).is_err()
        );
    }

    #[test]
    fn v8_light_lowers_to_an_ambient_and_a_directional_term() {
        let lights = crate::light::default_lights();
        assert_eq!(lights.ambient.unwrap().level.evaluate(0.0), 0.5);
        let directional = lights.directional.unwrap();
        assert_eq!(directional.level.evaluate(0.0), 1.0);
        assert!(!lights.terrain);
    }
}
