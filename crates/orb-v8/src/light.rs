//! The v8 flat-extrusion light; later scene lights remain separate.
use crate::{Color, Cx, Property, TranslateAnchor, layer::property_at, spec::Prop};
use orb_expr::{PropertyType, Type, json::Json};

#[derive(Clone, Debug)]
pub struct Light {
    pub anchor: Property<TranslateAnchor>,
    /// Radius, azimuth clockwise from north, polar angle from the zenith.
    pub position: Property<[f32; 3]>,
    pub color: Property<Color>,
    pub intensity: Property<f32>,
}

/// The spec's light: every property at its default.
impl Default for Light {
    fn default() -> Self {
        parse(&mut Cx::default(), None)
    }
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
    let intensity = prop(
        "intensity",
        PropertyType::Number,
        "0.5",
        true,
        Some(0.0),
        Some(1.0),
    );
    if let Some(json) = json {
        for member in cx.object(json, "light").unwrap_or_default() {
            if !["anchor", "position", "color", "intensity"].contains(&member.key.as_str()) {
                cx.warn(
                    format!("light.{}", member.key),
                    member.key_span,
                    format!("unknown light property \"{}\"", member.key),
                );
            }
        }
    }
    Light {
        anchor: property_at(
            cx,
            json.and_then(|j| j.get("anchor")),
            "light.anchor",
            &anchor,
        ),
        position: property_at(
            cx,
            json.and_then(|j| j.get("position")),
            "light.position",
            &position,
        ),
        color: property_at(cx, json.and_then(|j| j.get("color")), "light.color", &color),
        intensity: property_at(
            cx,
            json.and_then(|j| j.get("intensity")),
            "light.intensity",
            &intensity,
        ),
    }
}
