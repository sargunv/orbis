//! Paint properties by their v8 names (decision 0031): the typed IR's
//! interpolatable paint, which transitions animate and which documents
//! can address by name. Other paint (patterns, dash arrays, enums, ramps)
//! is not interpolated.

use crate::{Color, LayerKind, Property};

/// How a paint change animates (v8's `transition`), in milliseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transition {
    pub duration: f64,
    pub delay: f64,
}

/// GL JS's default: 300 ms, at once.
impl Default for Transition {
    fn default() -> Self {
        Self {
            duration: 300.0,
            delay: 0.0,
        }
    }
}

/// A paint property's own `*-transition`: the members it sets over the
/// style's transition, as GL JS merges them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TransitionOverride {
    pub duration: Option<f64>,
    pub delay: Option<f64>,
}

impl TransitionOverride {
    pub fn over(self, style: Transition) -> Transition {
        Transition {
            duration: self.duration.unwrap_or(style.duration),
            delay: self.delay.unwrap_or(style.delay),
        }
    }
}

/// A `{duration, delay}` object's members, or `None` where it isn't one.
pub(crate) fn parse_transition(json: &orb_expr::json::Json) -> Option<TransitionOverride> {
    let member = |key: &str| match json.get(key) {
        None => Some(None),
        Some(v) => v.as_f64().filter(|v| *v >= 0.0).map(Some),
    };
    json.as_object()?;
    Some(TransitionOverride {
        duration: member("duration")?,
        delay: member("delay")?,
    })
}

/// A paint property's value, by its kind.
#[derive(Debug)]
pub enum PaintRef<'a> {
    Number(&'a Property<f32>),
    Color(&'a Property<Color>),
    /// A colour that may be unset (`fill-outline-color`).
    OptionalColor(&'a Property<Option<Color>>),
    /// A pair of numbers (`*-translate`).
    Pair(&'a Property<[f32; 2]>),
}

/// A paint property to replace, by its kind.
#[derive(Debug)]
pub enum PaintMut<'a> {
    Number(&'a mut Property<f32>),
    Color(&'a mut Property<Color>),
    OptionalColor(&'a mut Property<Option<Color>>),
    Pair(&'a mut Property<[f32; 2]>),
}

/// An owned paint property, as a transition holds its prior value.
#[derive(Clone, Debug)]
pub enum PaintValue {
    Number(Property<f32>),
    Color(Property<Color>),
    OptionalColor(Property<Option<Color>>),
    Pair(Property<[f32; 2]>),
}

impl PaintValue {
    /// What the property varies with.
    pub fn kind(&self) -> orb_expr::Kind {
        match self {
            Self::Number(p) => p.kind(),
            Self::Color(p) => p.kind(),
            Self::OptionalColor(p) => p.kind(),
            Self::Pair(p) => p.kind(),
        }
    }

    pub fn as_mut(&mut self) -> PaintMut<'_> {
        match self {
            Self::Number(p) => PaintMut::Number(p),
            Self::Color(p) => PaintMut::Color(p),
            Self::OptionalColor(p) => PaintMut::OptionalColor(p),
            Self::Pair(p) => PaintMut::Pair(p),
        }
    }
}

impl PaintRef<'_> {
    pub fn to_owned(&self) -> PaintValue {
        match self {
            Self::Number(p) => PaintValue::Number((*p).clone()),
            Self::Color(p) => PaintValue::Color((*p).clone()),
            Self::OptionalColor(p) => PaintValue::OptionalColor((*p).clone()),
            Self::Pair(p) => PaintValue::Pair((*p).clone()),
        }
    }
}

// One table for both accessors: (layer kind, binding, [(name, kind, field)]).
macro_rules! paint_table {
    ($self:ident, $name:ident, $kind:ident, $($ref:tt)*) => {
        match $self {
            LayerKind::Background(l) => match $name {
                "background-color" => Some($kind::Color($($ref)* l.paint.color)),
                "background-opacity" => Some($kind::Number($($ref)* l.paint.opacity)),
                _ => None,
            },
            LayerKind::Fill(l) => match $name {
                "fill-opacity" => Some($kind::Number($($ref)* l.paint.opacity)),
                "fill-layer-opacity" => Some($kind::Number($($ref)* l.paint.layer_opacity)),
                "fill-color" => Some($kind::Color($($ref)* l.paint.color)),
                "fill-outline-color" => Some($kind::OptionalColor($($ref)* l.paint.outline_color)),
                "fill-translate" => Some($kind::Pair($($ref)* l.paint.translate)),
                _ => None,
            },
            LayerKind::FillExtrusion(l) => match $name {
                "fill-extrusion-color" => Some($kind::Color($($ref)* l.paint.color)),
                "fill-extrusion-opacity" => Some($kind::Number($($ref)* l.paint.opacity)),
                "fill-extrusion-height" => Some($kind::Number($($ref)* l.paint.height)),
                "fill-extrusion-base" => Some($kind::Number($($ref)* l.paint.base)),
                "fill-extrusion-translate" => Some($kind::Pair($($ref)* l.paint.translate)),
                _ => None,
            },
            LayerKind::Line(l) => match $name {
                "line-opacity" => Some($kind::Number($($ref)* l.paint.opacity)),
                "line-layer-opacity" => Some($kind::Number($($ref)* l.paint.layer_opacity)),
                "line-color" => Some($kind::Color($($ref)* l.paint.color)),
                "line-translate" => Some($kind::Pair($($ref)* l.paint.translate)),
                "line-width" => Some($kind::Number($($ref)* l.paint.width)),
                "line-gap-width" => Some($kind::Number($($ref)* l.paint.gap_width)),
                "line-offset" => Some($kind::Number($($ref)* l.paint.offset)),
                "line-blur" => Some($kind::Number($($ref)* l.paint.blur)),
                _ => None,
            },
            LayerKind::Circle(l) => match $name {
                "circle-radius" => Some($kind::Number($($ref)* l.paint.radius)),
                "circle-color" => Some($kind::Color($($ref)* l.paint.color)),
                "circle-blur" => Some($kind::Number($($ref)* l.paint.blur)),
                "circle-opacity" => Some($kind::Number($($ref)* l.paint.opacity)),
                "circle-translate" => Some($kind::Pair($($ref)* l.paint.translate)),
                "circle-stroke-width" => Some($kind::Number($($ref)* l.paint.stroke_width)),
                "circle-stroke-color" => Some($kind::Color($($ref)* l.paint.stroke_color)),
                "circle-stroke-opacity" => Some($kind::Number($($ref)* l.paint.stroke_opacity)),
                _ => None,
            },
            LayerKind::Heatmap(l) => match $name {
                "heatmap-radius" => Some($kind::Number($($ref)* l.paint.radius)),
                "heatmap-weight" => Some($kind::Number($($ref)* l.paint.weight)),
                "heatmap-intensity" => Some($kind::Number($($ref)* l.paint.intensity)),
                "heatmap-opacity" => Some($kind::Number($($ref)* l.paint.opacity)),
                _ => None,
            },
            LayerKind::Raster(l) => match $name {
                "raster-opacity" => Some($kind::Number($($ref)* l.paint.opacity)),
                "raster-hue-rotate" => Some($kind::Number($($ref)* l.paint.hue_rotate)),
                "raster-brightness-min" => Some($kind::Number($($ref)* l.paint.brightness_min)),
                "raster-brightness-max" => Some($kind::Number($($ref)* l.paint.brightness_max)),
                "raster-saturation" => Some($kind::Number($($ref)* l.paint.saturation)),
                "raster-contrast" => Some($kind::Number($($ref)* l.paint.contrast)),
                _ => None,
            },
            LayerKind::Hillshade(l) => match $name {
                "hillshade-exaggeration" => Some($kind::Number($($ref)* l.paint.exaggeration)),
                "hillshade-accent-color" => Some($kind::Color($($ref)* l.paint.accent_color)),
                _ => None,
            },
            LayerKind::ColorRelief(l) => match $name {
                "color-relief-opacity" => Some($kind::Number($($ref)* l.paint.opacity)),
                _ => None,
            },
            LayerKind::Symbol(l) => {
                let (part, rest) = if let Some(rest) = $name.strip_prefix("icon-") {
                    ($($ref)* l.paint.icon, rest)
                } else {
                    ($($ref)* l.paint.text, $name.strip_prefix("text-")?)
                };
                match rest {
                    "opacity" => Some($kind::Number($($ref)* part.opacity)),
                    "color" => Some($kind::Color($($ref)* part.color)),
                    "halo-color" => Some($kind::Color($($ref)* part.halo_color)),
                    "halo-width" => Some($kind::Number($($ref)* part.halo_width)),
                    "halo-blur" => Some($kind::Number($($ref)* part.halo_blur)),
                    "translate" => Some($kind::Pair($($ref)* part.translate)),
                    _ => None,
                }
            }
            LayerKind::Unsupported { .. } => None,
        }
    };
}

impl LayerKind {
    /// An interpolatable paint property by its v8 name.
    pub fn paint(&self, name: &str) -> Option<PaintRef<'_>> {
        paint_table!(self, name, PaintRef, &)
    }

    /// An interpolatable paint property by its v8 name, to replace.
    pub fn paint_mut(&mut self, name: &str) -> Option<PaintMut<'_>> {
        paint_table!(self, name, PaintMut, &mut)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paint_reads_and_replaces_by_name() {
        let mut style = crate::parse(
            r#"{"version":8,"sources":{"s":{"type":"geojson","data":{"type":"FeatureCollection","features":[]}}},
            "layers":[{"id":"f","type":"fill","source":"s","paint":{"fill-opacity":0.5}},
                      {"id":"t","type":"symbol","source":"s","paint":{"text-halo-width":2}}]}"#,
        )
        .unwrap();
        let Some(PaintRef::Number(opacity)) = style.layers[0].kind.paint("fill-opacity") else {
            panic!("a number");
        };
        assert_eq!(opacity.evaluate(0.0), 0.5);
        let Some(PaintRef::Number(halo)) = style.layers[1].kind.paint("text-halo-width") else {
            panic!("a number");
        };
        assert_eq!(halo.evaluate(0.0), 2.0);
        assert!(style.layers[0].kind.paint("fill-pattern").is_none());
        if let Some(PaintMut::Number(p)) = style.layers[0].kind.paint_mut("fill-opacity") {
            *p = Property::constant(0.25);
        }
        let Some(PaintRef::Number(opacity)) = style.layers[0].kind.paint("fill-opacity") else {
            panic!("a number");
        };
        assert_eq!(opacity.evaluate(0.0), 0.25);
    }
}
