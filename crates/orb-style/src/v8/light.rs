//! v8's `light`, parsed into the style's [`Light`].
use crate::v8::{Cx, layer::typed_property_at};
use crate::*;
use orb_expr::{PropertyType, Type, json::Json};

/// v8's default `light`: every property at its spec default.
pub fn default_light() -> Light {
    parse(&mut Cx::default(), None)
}

/// The light's properties that transition (`anchor` doesn't).
const TRANSITIONED: &[&str] = &["position", "color", "intensity"];

/// style-spec#1797's shadows, as maplibre/maplibre#555 names them.
const SHADOW_INTENSITY: &str = "shadow-intensity";

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
    // style-spec#1797's, behind its preview.
    let shadow_intensity = prop(
        SHADOW_INTENSITY,
        PropertyType::Number,
        "0",
        true,
        Some(0.0),
        Some(1.0),
    );
    let mut shadows = false;
    if let Some(json) = json {
        for member in cx.object(json, "light").unwrap_or_default() {
            let name = member.key.strip_suffix("-transition");
            if name.unwrap_or(&member.key) == SHADOW_INTENSITY {
                let what = format!("\"light.{}\"", member.key);
                let path = format!("light.{}", member.key);
                shadows |= cx.preview(Preview::Shadows, path, member.key_span, what);
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
    if shadows {
        transitioned.push(SHADOW_INTENSITY);
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
        shadow_intensity: typed_property_at(
            cx,
            json.and_then(|j| j.get(SHADOW_INTENSITY))
                .filter(|_| shadows),
            "light.shadow-intensity",
            &shadow_intensity,
        ),
        transitions: crate::v8::paint::root_transitions(cx, json, "light", &transitioned),
    }
}
