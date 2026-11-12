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
    if let Some(json) = json {
        for member in cx.object(json, "light").unwrap_or_default() {
            let known = match member.key.strip_suffix("-transition") {
                Some(name) => TRANSITIONED.contains(&name),
                None => ["anchor", "position", "color", "intensity"].contains(&member.key.as_str()),
            };
            if !known {
                cx.warn_unknown("light", member);
            }
        }
    }
    fn read<T: crate::PropertyValue>(
        cx: &mut Cx,
        json: Option<&Json>,
        prop: &Prop,
    ) -> crate::Property<T> {
        let value = json.and_then(|j| j.get(prop.name));
        typed_property_at(cx, value, &format!("light.{}", prop.name), prop)
    }
    let intensity = prop(
        "intensity",
        PropertyType::Number,
        "0.5",
        true,
        Some(0.0),
        Some(1.0),
    );
    Light {
        anchor: read(cx, json, &anchor),
        position: read(cx, json, &position),
        color: read(cx, json, &color),
        intensity: read(cx, json, &intensity),
        transitions: crate::v8::paint::root_transitions(cx, json, "light", TRANSITIONED),
    }
}
