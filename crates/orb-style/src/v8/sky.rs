//! v8's `sky`, lowered into the style's [`Atmosphere`]: one sky and fog
//! model.
use crate::v8::{Cx, layer::typed_property_at};
use crate::*;
use orb_expr::{PropertyType, json::Json};

pub(crate) fn parse(cx: &mut Cx, json: &Json) -> Atmosphere {
    const NAMES: &[&str] = &[
        "sky-color",
        "horizon-color",
        "fog-color",
        "fog-ground-blend",
        "horizon-fog-blend",
        "sky-horizon-blend",
        "atmosphere-blend",
    ];
    /// style-spec#1889's properties, behind its preview.
    const STARS: &[&str] = &["backdrop-color", "star-opacity"];
    let mut stars = false;
    for m in cx.object(json, "sky").unwrap_or_default() {
        let name = m.key.strip_suffix("-transition").unwrap_or(&m.key);
        if STARS.contains(&name) {
            let what = format!("\"sky.{}\"", m.key);
            stars |= cx.preview(Preview::Stars, format!("sky.{}", m.key), m.key_span, what);
        } else if !NAMES.contains(&m.key.as_str()) && !m.key.ends_with("-transition") {
            cx.warn_unknown("sky", m);
        }
    }
    let prop = |name, ty: PropertyType, default| {
        let number = ty == PropertyType::Number;
        Prop {
            name,
            ty,
            default: Some(default),
            zoom: true,
            feature: false,
            interpolated: true,
            minimum: number.then_some(0.0),
            maximum: number.then_some(1.0),
            tokens: false,
            ..Prop::ROOT
        }
    };
    let color = |cx: &mut Cx, name: &'static str, default| {
        let on = stars || !STARS.contains(&name);
        typed_property_at(
            cx,
            json.get(name).filter(|_| on),
            &format!("sky.{name}"),
            &prop(name, PropertyType::Color, default),
        )
    };
    let number = |cx: &mut Cx, name: &'static str, default| {
        let on = stars || !STARS.contains(&name);
        typed_property_at(
            cx,
            json.get(name).filter(|_| on),
            &format!("sky.{name}"),
            &prop(name, PropertyType::Number, default),
        )
    };
    Atmosphere {
        sky_color: color(cx, "sky-color", "\"#88C6FC\""),
        horizon_color: color(cx, "horizon-color", "\"white\""),
        fog_color: color(cx, "fog-color", "\"white\""),
        fog_ground_blend: number(cx, "fog-ground-blend", "0.5"),
        horizon_fog_blend: number(cx, "horizon-fog-blend", "0.8"),
        sky_horizon_blend: number(cx, "sky-horizon-blend", "0.8"),
        atmosphere_blend: number(cx, "atmosphere-blend", "0.8"),
        backdrop_color: color(cx, "backdrop-color", "\"transparent\""),
        star_opacity: number(cx, "star-opacity", "0"),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn absent_sky_is_transparent_but_empty_sky_uses_spec_defaults() {
        let absent = crate::v8::parse(r#"{"version":8,"sources":{},"layers":[]}"#).unwrap();
        assert!(absent.atmosphere.is_none());
        let defaults =
            crate::v8::parse(r#"{"version":8,"sky":{},"sources":{},"layers":[]}"#).unwrap();
        let sky = defaults.atmosphere.unwrap();
        assert_eq!(
            sky.sky_color.evaluate(0.0).straight(),
            [136.0 / 255.0, 198.0 / 255.0, 252.0 / 255.0, 1.0]
        );
        assert_eq!(sky.sky_horizon_blend.evaluate(0.0), 0.8);
    }

    #[test]
    fn sky_zoom_expressions_and_ranges_follow_the_spec() {
        let style = crate::v8::parse(r#"{"version":8,"sky":{"sky-horizon-blend":["interpolate",["linear"],["zoom"],0,0,10,1]},"sources":{},"layers":[]}"#).unwrap();
        assert_eq!(
            style.atmosphere.unwrap().sky_horizon_blend.evaluate(5.0),
            0.5
        );
        assert!(
            crate::v8::parse(
                r#"{"version":8,"sky":{"sky-horizon-blend":2},"sources":{},"layers":[]}"#
            )
            .is_err()
        );
    }

    #[test]
    fn stars_and_backdrop_need_their_preview() {
        let json = r##"{"version":8,"sky":{"star-opacity":["interpolate",["linear"],["zoom"],0,1,6,0],
            "backdrop-color":"#000010","star-opacity-transition":{"duration":0}},"sources":{},"layers":[]}"##;
        let off = crate::v8::parse(json).unwrap();
        assert_eq!(off.warnings.len(), 3);
        let sky = off.atmosphere.unwrap();
        assert_eq!(sky.star_opacity.evaluate(0.0), 0.0);
        assert_eq!(sky.backdrop_color.evaluate(0.0).straight()[3], 0.0);
        let on = crate::v8::parse_with(
            json,
            crate::Previews::default().with(crate::Preview::Stars),
            &[],
        )
        .unwrap();
        assert!(on.warnings.is_empty(), "{:?}", on.warnings);
        let sky = on.atmosphere.unwrap();
        assert_eq!(sky.star_opacity.evaluate(3.0), 0.5);
        assert_eq!(
            sky.backdrop_color.evaluate(0.0).straight(),
            [0.0, 0.0, 16.0 / 255.0, 1.0]
        );
        assert!(
            crate::v8::parse_with(
                r#"{"version":8,"sky":{"star-opacity":2},"sources":{},"layers":[]}"#,
                crate::Previews::ALL,
                &[]
            )
            .is_err()
        );
    }

    #[test]
    fn sky_has_v8_properties_only() {
        let style = crate::v8::parse(
            r#"{"version":8,"sky":{"star-intensity":1},"sources":{},"layers":[]}"#,
        )
        .unwrap();
        assert_eq!(style.warnings.len(), 1);
    }
}
