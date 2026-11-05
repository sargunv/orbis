//! v8's `sky`, lowered into the style's [`Atmosphere`] (decision 0025):
//! one sky and fog model, which the scene document's `atmosphere` sets
//! with the same names and two of its own, `star-intensity` and
//! `backdrop-color` (decision 0035).
use crate::{Color, Cx, Property, layer::property_at, spec::Prop};
use orb_expr::{PropertyType, json::Json};

#[derive(Clone, Debug)]
/// The style's atmosphere: the sky above the horizon and the fog over
/// terrain.
pub struct Atmosphere {
    pub sky_color: Property<Color>,
    pub horizon_color: Property<Color>,
    pub fog_color: Property<Color>,
    pub fog_ground_blend: Property<f32>,
    pub horizon_fog_blend: Property<f32>,
    pub sky_horizon_blend: Property<f32>,
    pub atmosphere_blend: Property<f32>,
    /// The stars' brightness, 0 (none) to 1, behind the sky.
    pub stars: Property<f32>,
    /// The colour behind everything, stars before it: where neither the
    /// map nor the sky covers it. Transparent shows the host's backing.
    pub backdrop: Property<Color>,
}

/// Where an atmosphere is declared: v8's `sky`, or the scene document's
/// `atmosphere`, which adds the stars and the backdrop.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Declared {
    Sky,
    Scene,
}

pub(crate) fn parse(cx: &mut Cx, json: &Json, declared: Declared) -> Atmosphere {
    const NAMES: &[&str] = &[
        "sky-color",
        "horizon-color",
        "fog-color",
        "fog-ground-blend",
        "horizon-fog-blend",
        "sky-horizon-blend",
        "atmosphere-blend",
    ];
    const SCENE: &[&str] = &["star-intensity", "backdrop-color"];
    let root = match declared {
        Declared::Sky => "sky",
        Declared::Scene => "atmosphere",
    };
    for m in cx.object(json, root).unwrap_or_default() {
        let known = NAMES.contains(&m.key.as_str())
            || (declared == Declared::Scene && SCENE.contains(&m.key.as_str()));
        if !known && !m.key.ends_with("-transition") {
            cx.warn(
                format!("{root}.{}", m.key),
                m.key_span,
                format!("unknown {root} property \"{}\"", m.key),
            );
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
        }
    };
    let color = |cx: &mut Cx, name, default| {
        property_at(
            cx,
            json.get(name),
            &format!("{root}.{name}"),
            &prop(name, PropertyType::Color, default),
        )
    };
    let number = |cx: &mut Cx, name, default| {
        property_at(
            cx,
            json.get(name),
            &format!("{root}.{name}"),
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
        // In v8's sky (style-spec#1889's `star-opacity`, `backdrop-color`,
        // once adopted, lower here) they keep their defaults.
        stars: match declared {
            Declared::Scene => number(cx, "star-intensity", "0"),
            Declared::Sky => Property::constant(0.0),
        },
        backdrop: match declared {
            Declared::Scene => color(cx, "backdrop-color", "\"transparent\""),
            Declared::Sky => Property::constant(Color::TRANSPARENT),
        },
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn absent_sky_is_transparent_but_empty_sky_uses_spec_defaults() {
        let absent = crate::parse(r#"{"version":8,"sources":{},"layers":[]}"#).unwrap();
        assert!(absent.atmosphere.is_none());
        let defaults = crate::parse(r#"{"version":8,"sky":{},"sources":{},"layers":[]}"#).unwrap();
        let sky = defaults.atmosphere.unwrap();
        assert_eq!(
            sky.sky_color.evaluate(0.0).straight(),
            [136.0 / 255.0, 198.0 / 255.0, 252.0 / 255.0, 1.0]
        );
        assert_eq!(sky.sky_horizon_blend.evaluate(0.0), 0.8);
    }

    #[test]
    fn sky_zoom_expressions_and_ranges_follow_the_spec() {
        let style = crate::parse(r#"{"version":8,"sky":{"sky-horizon-blend":["interpolate",["linear"],["zoom"],0,0,10,1]},"sources":{},"layers":[]}"#).unwrap();
        assert_eq!(
            style.atmosphere.unwrap().sky_horizon_blend.evaluate(5.0),
            0.5
        );
        assert!(
            crate::parse(r#"{"version":8,"sky":{"sky-horizon-blend":2},"sources":{},"layers":[]}"#)
                .is_err()
        );
    }

    #[test]
    fn scenes_add_stars_and_a_backdrop_which_v8_sky_lacks() {
        let scene = crate::parse_scene(
            r##"{"style": "s.json", "atmosphere": {"sky-color": "#000010",
                "star-intensity": 0.8, "backdrop-color": "#020308"}}"##,
        )
        .unwrap();
        assert!(scene.warnings.is_empty(), "{:?}", scene.warnings);
        let a = scene.atmosphere.unwrap();
        assert_eq!(a.stars.evaluate(0.0), 0.8);
        assert_eq!(
            a.backdrop.evaluate(0.0).straight(),
            [2.0 / 255.0, 3.0 / 255.0, 8.0 / 255.0, 1.0]
        );
        // Not (yet) v8: warned, and the stars stay off.
        let style =
            crate::parse(r#"{"version":8,"sky":{"star-intensity":1},"sources":{},"layers":[]}"#)
                .unwrap();
        assert_eq!(style.warnings.len(), 1);
        assert_eq!(style.atmosphere.unwrap().stars.evaluate(0.0), 0.0);
    }
}
