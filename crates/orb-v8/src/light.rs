//! v8's `light`, lowered into the style's [`Lights`]: an
//! ambient and a directional term of GL JS's flat extrusion response.
use crate::{Color, Cx, Property, TranslateAnchor, layer::typed_property_at};
use orb_expr::{PropertyType, Type, json::Json};
use orb_style::*;
use orb_style::{AmbientLight, Direction, DirectionalLight, Level, Lights};

#[derive(Clone, Debug)]
pub(crate) struct Light {
    pub anchor: Property<TranslateAnchor>,
    /// Radius, azimuth clockwise from north, polar angle from the zenith.
    pub position: Property<[f32; 3]>,
    pub color: Property<Color>,
    pub intensity: Property<f32>,
}

impl Light {
    /// GL JS shades an extrusion face `mix(1 − i, max(1 − lum + i, 1), n·l)`
    /// of its colour: an ambient level `1 − i` and a directional one `2i`
    /// of the light's colour, in the response `mix(a, max(a + d − lum, 1),
    /// n·l)` that scene lights share.
    pub(crate) fn lower(self) -> Lights {
        Lights {
            ambient: Some(AmbientLight {
                color: self.color.clone(),
                level: Level {
                    intensity: self.intensity.clone(),
                    offset: 1.0,
                    scale: -1.0,
                },
            }),
            directional: Some(DirectionalLight {
                color: self.color,
                level: Level {
                    intensity: self.intensity,
                    offset: 0.0,
                    scale: 2.0,
                },
                anchor: self.anchor,
                direction: Direction::Position(self.position),
                // v8's light casts no shadows.
                shadow: None,
            }),
            terrain: false,
            materials: Vec::new(),
        }
    }
}

/// v8's default `light`, lowered: every property at its spec default.
pub fn default_lights() -> Lights {
    parse(&mut Cx::default(), None).lower()
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
        ..Prop::SCENE
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
