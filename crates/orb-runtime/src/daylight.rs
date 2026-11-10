//! Label colours over `["dev.sargunv.orbis:daylight"]`, how lit the
//! ground is where a label stands, drawn per label every frame: the
//! symbol shader finds the daylight at the label's anchor and mixes the
//! colour between its curve's two stops. Each stop's colour is a property
//! of its own, so it comes from the node, the feature's row or its state
//! as any colour does; the night stops take two slots after the family's
//! five (`symbol.wgsl`'s `SymbolNode`).

use orb_expr::{Interpolation, Kind, PropertyExpression};
use orb_style::{Color, Key, LayerType, PropertyRef, Slot, SymbolPart, Value};

/// A symbol node's colours over the daylight.
#[derive(Debug)]
pub(crate) struct LabelDaylight {
    /// The colour's and the halo colour's, each `None` where it doesn't
    /// read the daylight.
    parts: [Option<DaylightColor>; 2],
}

/// One colour over the daylight.
#[derive(Debug)]
struct DaylightColor {
    /// Its night and day stops: at the curve's first and last.
    stops: [Slot; 2],
    /// The curve as the shader takes it, `None` to draw the day stop
    /// alone: the first stop's daylight, the last's, 0 for a step or 1 for
    /// an exponential curve, and its base.
    curve: Option<[f32; 4]>,
}

impl LabelDaylight {
    /// The colours of `layer`'s `part` over the daylight, when either
    /// reads it, and what can't draw per label, which draws at full
    /// daylight.
    pub(crate) fn of(layer: &orb_style::Layer, part: SymbolPart) -> (Option<Self>, Vec<String>) {
        let mut warnings = Vec::new();
        if layer.ty != LayerType::Symbol {
            return (None, warnings);
        }
        let parts = [part.color, part.halo_color].map(|key| {
            let expression = layer.get(key).as_expression()?.clone();
            expression.reads_daylight().then(|| {
                DaylightColor::of(&expression).unwrap_or_else(|why| {
                    warnings.push(format!(
                        "{}: {why}; it draws the day side's",
                        key.id().name()
                    ));
                    let day = slot(expression.at_daylight(1.0));
                    DaylightColor {
                        stops: [day.clone(), day],
                        curve: None,
                    }
                })
            })
        });
        let daylight = parts.iter().any(Option::is_some).then_some(Self { parts });
        (daylight, warnings)
    }

    /// The day stop of colour `key` (part slot 1 or 2): the layer's own
    /// where it doesn't read the daylight.
    pub(crate) fn day<'a>(
        &'a self,
        layer: &'a orb_style::Layer,
        index: usize,
        key: Key<Color>,
    ) -> PropertyRef<'a, Color> {
        match &self.parts[index] {
            Some(c) => PropertyRef::of(&c.stops[1], key.id()),
            None => layer.get(key),
        }
    }

    /// The night stop of colour `key`: the day's where it doesn't read the
    /// daylight, which the shader then doesn't mix.
    pub(crate) fn night<'a>(
        &'a self,
        layer: &'a orb_style::Layer,
        index: usize,
        key: Key<Color>,
    ) -> PropertyRef<'a, Color> {
        match &self.parts[index] {
            Some(c) => PropertyRef::of(&c.stops[0], key.id()),
            None => layer.get(key),
        }
    }

    /// The shader's curves of the colour and the halo colour, `None` where
    /// it draws the day stop alone.
    pub(crate) fn curves(&self) -> [Option<[f32; 4]>; 2] {
        self.parts
            .each_ref()
            .map(|p| p.as_ref().and_then(|c| c.curve))
    }
}

impl DaylightColor {
    fn of(expression: &PropertyExpression) -> Result<Self, &'static str> {
        let curve = expression.daylight_curve()?;
        let last = curve.labels.len() - 1;
        let stops = [slot(curve.stop(0)), slot(curve.stop(last))];
        let (lo, hi) = (curve.labels[0], curve.labels[last]);
        let shader = match curve.interpolation {
            // One stop: the same value by night and day.
            _ if last == 0 => [0.0, f32::MAX, 0.0, 1.0],
            None => [0.0, hi as f32, 0.0, 1.0],
            Some(Interpolation::Exponential(base)) => [lo as f32, hi as f32, 1.0, base as f32],
            Some(_) => [lo as f32, hi as f32, 1.0, 1.0],
        };
        Ok(Self {
            stops,
            curve: Some(shader),
        })
    }
}

fn slot(value: PropertyExpression) -> Slot {
    match value.kind() {
        Kind::Constant if !value.expression().reads_images() => {
            let value = value.expression().evaluate(0.0).map(Value::into_owned);
            Slot::constant(value.unwrap_or(Value::Null), true)
        }
        _ => Slot::expression(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(paint: &str) -> std::sync::Arc<orb_style::Layer> {
        let style = format!(
            r#"{{"version": 8, "sources": {{"s": {{"type": "geojson",
                "data": {{"type": "FeatureCollection", "features": []}}}}}},
                "layers": [{{"id": "l", "type": "symbol", "source": "s",
                "paint": {{{paint}}}}}]}}"#
        );
        let parsed = orb_style::v8::parse_with(&style, orb_style::Previews::ALL, &[]).unwrap();
        parsed.layers[0].clone()
    }

    #[test]
    fn a_colour_over_the_daylight_draws_between_its_stops() {
        let l = layer(
            r##""text-halo-color": ["interpolate", ["exponential", 2], ["dev.sargunv.orbis:daylight"], 0.25, "#000", 0.75, ["get", "c"]], "text-color": "#123""##,
        );
        let (daylight, warnings) = LabelDaylight::of(&l, orb_style::TEXT_PART);
        assert!(warnings.is_empty());
        let daylight = daylight.unwrap();
        assert_eq!(daylight.curves(), [None, Some([0.25, 0.75, 1.0, 2.0])]);
        let halo = orb_style::TEXT_HALO_COLOR;
        assert_eq!(daylight.night(&l, 1, halo).kind(), Kind::Constant);
        assert_eq!(daylight.day(&l, 1, halo).kind(), Kind::Source);
        // The colour that doesn't read it is the layer's, by night and day.
        let color = orb_style::TEXT_COLOR;
        assert_eq!(
            daylight.night(&l, 0, color).constant(),
            daylight.day(&l, 0, color).constant()
        );
        // A label layer that doesn't read it has none.
        assert!(
            LabelDaylight::of(&layer(r#""text-color": "red""#), orb_style::TEXT_PART)
                .0
                .is_none()
        );
    }

    #[test]
    fn other_shapes_draw_the_day_side() {
        let l = layer(
            r##""icon-color": ["interpolate", ["linear"], ["dev.sargunv.orbis:daylight"], 0, "#fff", 0.5, "#f00", 1, "#000"]"##,
        );
        let (daylight, warnings) = LabelDaylight::of(&l, orb_style::ICON_PART);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        let daylight = daylight.unwrap();
        assert_eq!(daylight.curves(), [None, None]);
        let day = daylight.day(&l, 0, orb_style::ICON_COLOR).evaluate(0.0);
        assert_eq!(day.premultiplied(), [0.0, 0.0, 0.0, 1.0]);
    }
}
