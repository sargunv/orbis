//! The style's sky. Fog and atmosphere values are retained for the terrain
//! and globe paths; on a flat surface GL JS draws only the sky gradient.
use crate::{Color, Cx, Property, layer::property_at, spec::Prop};
use orb_expr::{PropertyType, json::Json};

#[derive(Clone, Debug)]
pub struct Sky {
    pub color: Property<Color>,
    pub horizon_color: Property<Color>,
    pub fog_color: Property<Color>,
    pub fog_ground_blend: Property<f32>,
    pub horizon_fog_blend: Property<f32>,
    pub sky_horizon_blend: Property<f32>,
    pub atmosphere_blend: Property<f32>,
}

pub(crate) fn parse(cx: &mut Cx, json: &Json) -> Sky {
    const NAMES: &[&str] = &[
        "sky-color",
        "horizon-color",
        "fog-color",
        "fog-ground-blend",
        "horizon-fog-blend",
        "sky-horizon-blend",
        "atmosphere-blend",
    ];
    for m in cx.object(json, "sky").unwrap_or_default() {
        if !NAMES.contains(&m.key.as_str()) && !m.key.ends_with("-transition") {
            cx.warn(
                format!("sky.{}", m.key),
                m.key_span,
                format!("unknown sky property \"{}\"", m.key),
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
            &format!("sky.{name}"),
            &prop(name, PropertyType::Color, default),
        )
    };
    let number = |cx: &mut Cx, name, default| {
        property_at(
            cx,
            json.get(name),
            &format!("sky.{name}"),
            &prop(name, PropertyType::Number, default),
        )
    };
    Sky {
        color: color(cx, "sky-color", "\"#88C6FC\""),
        horizon_color: color(cx, "horizon-color", "\"white\""),
        fog_color: color(cx, "fog-color", "\"white\""),
        fog_ground_blend: number(cx, "fog-ground-blend", "0.5"),
        horizon_fog_blend: number(cx, "horizon-fog-blend", "0.8"),
        sky_horizon_blend: number(cx, "sky-horizon-blend", "0.8"),
        atmosphere_blend: number(cx, "atmosphere-blend", "0.8"),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn absent_sky_is_transparent_but_empty_sky_uses_spec_defaults() {
        let absent = crate::parse(r#"{"version":8,"sources":{},"layers":[]}"#).unwrap();
        assert!(absent.sky.is_none());
        let defaults = crate::parse(r#"{"version":8,"sky":{},"sources":{},"layers":[]}"#).unwrap();
        let sky = defaults.sky.unwrap();
        assert_eq!(
            sky.color.evaluate(0.0).straight(),
            [136.0 / 255.0, 198.0 / 255.0, 252.0 / 255.0, 1.0]
        );
        assert_eq!(sky.sky_horizon_blend.evaluate(0.0), 0.8);
    }

    #[test]
    fn sky_zoom_expressions_and_ranges_follow_the_spec() {
        let style = crate::parse(r#"{"version":8,"sky":{"sky-horizon-blend":["interpolate",["linear"],["zoom"],0,0,10,1]},"sources":{},"layers":[]}"#).unwrap();
        assert_eq!(style.sky.unwrap().sky_horizon_blend.evaluate(5.0), 0.5);
        assert!(
            crate::parse(r#"{"version":8,"sky":{"sky-horizon-blend":2},"sources":{},"layers":[]}"#)
                .is_err()
        );
    }
}
