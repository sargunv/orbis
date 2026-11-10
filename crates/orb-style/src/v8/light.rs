//! v8's `light`, parsed into the style's [`Light`].
use crate::v8::{Cx, layer::typed_property_at};
use crate::*;
use orb_expr::{PropertyType, Type, json::Json};

/// v8's default `light`: every property at its spec default.
pub fn default_light() -> Light {
    parse(&mut Cx::default(), None)
}

/// The light's spec properties that transition (`anchor` doesn't).
const TRANSITIONED: &[&str] = &["position", "color", "intensity"];

/// style-spec#1797's shadows, as maplibre/maplibre#555 has them.
const SHADOW_INTENSITY: &str = "dev.sargunv.orbis:shadow-intensity";
const GROUND_LIGHTING: &str = "dev.sargunv.orbis:ground-lighting";
const AMBIENT_COLOR: &str = "dev.sargunv.orbis:ambient-color";
const AMBIENT_INTENSITY: &str = "dev.sargunv.orbis:ambient-intensity";
const TWILIGHT: &str = "dev.sargunv.orbis:twilight";
pub(crate) const DATETIME: &str = "dev.sargunv.orbis:datetime";

/// The light's properties beyond the spec, each behind its preview; all
/// transition.
const PREVIEWS: [(&str, Preview); 6] = [
    (SHADOW_INTENSITY, Preview::ShadowIntensity),
    (GROUND_LIGHTING, Preview::GroundLighting),
    (AMBIENT_COLOR, Preview::AmbientColor),
    (AMBIENT_INTENSITY, Preview::AmbientIntensity),
    (TWILIGHT, Preview::Twilight),
    (DATETIME, Preview::Datetime),
];

/// `dev.sargunv.orbis:datetime` as milliseconds since 1970 UTC: a number
/// as is, or an RFC 3339 date and time; `None` (warned) for anything else.
fn datetime(cx: &mut Cx, json: &Json) -> Option<f64> {
    let path = format!("light.{DATETIME}");
    let ms = match &json.value {
        orb_expr::json::JsonValue::Number(n) => Some(*n).filter(|n| n.is_finite()),
        orb_expr::json::JsonValue::String(s) => rfc3339(s),
        _ => None,
    };
    if ms.is_none() {
        cx.warn(
            path,
            json.span,
            format!("an RFC 3339 date and time or milliseconds since 1970 expected, {json} found; ignored"),
        );
    }
    ms
}

/// An RFC 3339 date and time (`2026-06-21T12:00:00Z`, fractional seconds
/// and a `±hh:mm` offset allowed) as milliseconds since 1970 UTC.
fn rfc3339(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    let digits = |r: std::ops::Range<usize>| -> Option<i64> {
        let part = b.get(r)?;
        part.iter()
            .all(u8::is_ascii_digit)
            .then(|| (part.iter()).fold(0, |n, d| n * 10 + i64::from(d - b'0')))
    };
    let at = |i: usize, c: &[u8]| b.get(i).is_some_and(|x| c.contains(x));
    if !(at(4, b"-") && at(7, b"-") && at(10, b"Tt ") && at(13, b":") && at(16, b":")) {
        return None;
    }
    let (y, mo, d) = (digits(0..4)?, digits(5..7)?, digits(8..10)?);
    let (h, mi, sec) = (digits(11..13)?, digits(14..16)?, digits(17..19)?);
    let mut i = 19;
    let mut fraction = 0.0;
    if at(i, b".") {
        let end = (i + 1..b.len())
            .find(|&j| !b[j].is_ascii_digit())
            .unwrap_or(b.len());
        fraction = s.get(i..end)?.parse::<f64>().ok()?;
        i = end;
    }
    let offset = if at(i, b"Zz") && i + 1 == b.len() {
        0
    } else if at(i, b"+-") && at(i + 3, b":") && i + 6 == b.len() {
        let minutes = digits(i + 1..i + 3)? * 60 + digits(i + 4..i + 6)?;
        if b[i] == b'-' { -minutes } else { minutes }
    } else {
        return None;
    };
    let days_in = |m: i64| match m {
        2 if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    if !(1..=12).contains(&mo) || !(1..=days_in(mo)).contains(&d) || h > 23 || mi > 59 || sec > 60 {
        return None;
    }
    // Days since 1970 from the civil date (Howard Hinnant's algorithm).
    let y = if mo <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((mo + 9) % 12) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    let seconds = ((days * 24 + h) * 60 + mi - offset) * 60 + sec;
    Some((seconds as f64 + fraction) * 1000.0)
}

pub(crate) fn parse(cx: &mut Cx, json: Option<&Json>) -> Light {
    let prop = |name, ty, default, interpolated, minimum, maximum| Prop {
        name,
        ty,
        default: Some(default),
        zoom: true,
        feature: false,
        interpolated,
        minimum,
        maximum,
        tokens: false,
        ..Prop::ROOT
    };
    let number = |name, default, maximum| {
        prop(
            name,
            PropertyType::Number,
            default,
            true,
            Some(0.0),
            Some(maximum),
        )
    };
    let anchor = prop(
        "anchor",
        PropertyType::Enum(&["map", "viewport"]),
        "\"viewport\"",
        false,
        None,
        None,
    );
    let position = prop(
        "position",
        PropertyType::Array(Type::Number, Some(3)),
        "[1.15,210,30]",
        true,
        None,
        None,
    );
    let color = prop("color", PropertyType::Color, "\"white\"", true, None, None);
    // Each property's preview, on when the style uses it with its preview on.
    let mut on = [false; PREVIEWS.len()];
    if let Some(json) = json {
        for member in cx.object(json, "light").unwrap_or_default() {
            let name = member.key.strip_suffix("-transition");
            let base = name.unwrap_or(&member.key);
            if let Some(i) = PREVIEWS.iter().position(|&(n, _)| n == base) {
                let what = format!("\"light.{}\"", member.key);
                let path = format!("light.{}", member.key);
                on[i] |= cx.preview(PREVIEWS[i].1, path, member.key_span, what);
                continue;
            }
            let known = match name {
                Some(name) => TRANSITIONED.contains(&name),
                None => ["anchor", "position", "color", "intensity"].contains(&member.key.as_str()),
            };
            if !known {
                cx.warn_unknown("light", member);
            }
        }
    }
    let mut transitioned = TRANSITIONED.to_vec();
    transitioned.extend((PREVIEWS.iter().zip(on)).filter_map(|(&(n, _), on)| on.then_some(n)));
    // A property beyond the spec reads the style only with its preview on.
    fn read<T: crate::PropertyValue>(
        cx: &mut Cx,
        json: Option<&Json>,
        on: &[bool],
        prop: &Prop,
    ) -> crate::Property<T> {
        let enabled = (PREVIEWS.iter())
            .position(|&(n, _)| n == prop.name)
            .is_none_or(|i| on[i]);
        let value = json.and_then(|j| j.get(prop.name)).filter(|_| enabled);
        typed_property_at(cx, value, &format!("light.{}", prop.name), prop)
    }
    Light {
        anchor: read(cx, json, &on, &anchor),
        position: read(cx, json, &on, &position),
        color: read(cx, json, &on, &color),
        intensity: read(cx, json, &on, &number("intensity", "0.5", 1.0)),
        shadow_intensity: read(cx, json, &on, &number(SHADOW_INTENSITY, "0", 1.0)),
        ground_lighting: read(cx, json, &on, &number(GROUND_LIGHTING, "0", 1.0)),
        ambient_color: read(
            cx,
            json,
            &on,
            &prop(
                AMBIENT_COLOR,
                PropertyType::Color,
                "\"white\"",
                true,
                None,
                None,
            ),
        ),
        ambient_intensity: read(cx, json, &on, &number(AMBIENT_INTENSITY, "0.5", 1.0)),
        ambient: [AMBIENT_COLOR, AMBIENT_INTENSITY].into_iter().any(|key| {
            let i = PREVIEWS
                .iter()
                .position(|&(n, _)| n == key)
                .expect("a preview");
            on[i] && json.is_some_and(|j| j.get(key).is_some())
        }),
        twilight: read(cx, json, &on, &number(TWILIGHT, "12", 90.0)),
        datetime: (json.and_then(|j| j.get(DATETIME)))
            .filter(|_| on[PREVIEWS.len() - 1])
            .and_then(|value| datetime(cx, value)),
        transitions: crate::v8::paint::root_transitions(cx, json, "light", &transitioned),
    }
}

#[cfg(test)]
mod tests {
    use crate::{Preview, Previews};

    const DOC: &str = r##"{"version": 8, "sources": {}, "layers": [],
        "light": {"dev.sargunv.orbis:ground-lighting": 1,
                  "dev.sargunv.orbis:ground-lighting-transition": {"duration": 0},
                  "dev.sargunv.orbis:ambient-color": "#203050",
                  "dev.sargunv.orbis:ambient-intensity": ["interpolate", ["linear"], ["zoom"], 0, 0.2, 10, 0.6],
                  "dev.sargunv.orbis:twilight": 20}}"##;

    #[test]
    fn ground_lighting_reads_each_key_behind_its_preview() {
        let on = crate::v8::parse_with(DOC, Previews::ALL, &[]).unwrap();
        assert!(on.warnings.is_empty(), "{:?}", on.warnings);
        let light = &on.light;
        assert_eq!(light.ground_lighting.evaluate(0.0), 1.0);
        assert_eq!(light.ambient_intensity.evaluate(5.0), 0.4);
        assert_eq!(light.twilight.evaluate(0.0), 20.0);
        assert!(light.ambient, "the extrusions take the ambient light");
        assert_eq!(
            light.ambient_color.evaluate(0.0).straight()[2],
            0x50 as f32 / 255.0
        );
        let names: Vec<&str> = (light.transitions.iter())
            .map(|(n, _)| n.as_str())
            .collect();
        assert_eq!(names, ["dev.sargunv.orbis:ground-lighting"]);

        // One preview on gates only its own key.
        let only = Previews::NONE.with(Preview::GroundLighting);
        let some = crate::v8::parse_with(DOC, only, &[]).unwrap();
        assert_eq!(some.warnings.len(), 3, "{:?}", some.warnings);
        assert_eq!(some.light.ground_lighting.evaluate(0.0), 1.0);
        assert_eq!(some.light.twilight.evaluate(0.0), 12.0);
        assert_eq!(some.light.ambient_intensity.evaluate(0.0), 0.5);
        assert!(!some.light.ambient);

        let off = crate::v8::parse(DOC).unwrap();
        assert_eq!(off.warnings.len(), 5, "{:?}", off.warnings);
        for (w, preview) in off.warnings.iter().zip([
            Preview::GroundLighting,
            Preview::GroundLighting,
            Preview::AmbientColor,
            Preview::AmbientIntensity,
            Preview::Twilight,
        ]) {
            assert!(w.message.contains(preview.name()), "{}", w.message);
        }
        assert_eq!(off.light.ground_lighting.evaluate(0.0), 0.0);
        assert!(off.light.transitions.is_empty());
    }

    #[test]
    fn datetime_reads_rfc_3339_or_milliseconds_behind_its_preview() {
        let parse = |value: &str, previews| {
            let doc = format!(
                r#"{{"version": 8, "sources": {{}}, "layers": [],
                    "light": {{"anchor": "map", "dev.sargunv.orbis:datetime": {value}}}}}"#
            );
            let style = crate::v8::parse_with(&doc, previews, &[]).unwrap();
            (style.light.datetime, style.warnings.len())
        };
        let on = Previews::NONE.with(Preview::Datetime);
        // 2026-06-21T12:00:00Z, as `Date.UTC(2026, 5, 21, 12)`.
        let noon = 1_782_043_200_000.0;
        assert_eq!(parse("\"2026-06-21T12:00:00Z\"", on), (Some(noon), 0));
        assert_eq!(parse(&noon.to_string(), on), (Some(noon), 0));
        assert_eq!(
            parse("\"2026-06-21T21:30:00.5+09:30\"", on),
            (Some(noon + 500.0), 0)
        );
        assert_eq!(
            parse("\"1969-12-31T23:59:59-00:00\"", on),
            (Some(-1000.0), 0)
        );
        assert_eq!(
            parse("\"2024-02-29T00:00:00Z\"", on).0,
            Some(1_709_164_800_000.0)
        );
        for bad in [
            "\"2026-02-29T00:00:00Z\"",
            "\"2026-06-21\"",
            "\"2026-06-21T12:00:00\"",
            "true",
        ] {
            assert_eq!(parse(bad, on), (None, 1), "{bad}");
        }
        assert_eq!(parse("\"2026-06-21T12:00:00Z\"", Previews::NONE), (None, 1));
    }
}
