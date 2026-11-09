//! v8's `light`, parsed into the style's [`Light`].
use crate::v8::{Cx, layer::typed_property_at};
use crate::*;
use orb_expr::{PropertyType, Type, json::Json};

/// v8's default `light`: every property at its spec default.
pub fn default_light() -> Light {
    parse(&mut Cx::default(), None)
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
                cx.warn_unknown("light", member);
            }
        }
    }
    Light {
        anchor: typed_property_at(
            cx,
            json.and_then(|j| j.get("anchor")),
            "light.anchor",
            &anchor,
        ),
        position: typed_property_at(
            cx,
            json.and_then(|j| j.get("position")),
            "light.position",
            &position,
        ),
        color: typed_property_at(cx, json.and_then(|j| j.get("color")), "light.color", &color),
        intensity: typed_property_at(
            cx,
            json.and_then(|j| j.get("intensity")),
            "light.intensity",
            &intensity,
        ),
    }
}
