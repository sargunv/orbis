//! `fill-extrusion-color` over `["elevation"]`, the height above the
//! ground, drawn per fragment: the extrusion shader takes the curve's
//! input through its arithmetic and finds the colour between its stops.
//! Each stop's colour is a property of its own, so it comes from the node,
//! the feature's row or its state as any colour does.

use orb_expr::{ElevationOp, Interpolation, Kind, PropertyExpression};
use orb_render::{Curve, ElevationPaint, pack_color};
use orb_shaders::layout::{MAX_ELEVATION_OPS, MAX_ELEVATION_STOPS};
use orb_style::{LayerType, Slot, Value};

use crate::style::PaintProperty;

/// Pieces a stop interval of a Lab or HCL curve is drawn as: the shader
/// mixes in RGB, so between them the colour is the curve's own.
const PIECES: usize = 8;

/// An extrusion layer's colour as a curve over the elevation.
#[derive(Debug)]
pub(crate) struct ElevationColor {
    input: Vec<ElevationOp>,
    labels: Vec<f64>,
    /// `None` for a `step`.
    interpolation: Option<Interpolation>,
    /// `fill-extrusion-color` at each stop.
    pub(crate) colors: Vec<Slot>,
}

impl ElevationColor {
    /// The layer's colour as a curve, when it reads the elevation and its
    /// colour draws: with a pattern it doesn't. An error says why it
    /// can't be drawn per fragment.
    pub(crate) fn of(layer: &orb_style::Layer) -> Option<Result<Self, String>> {
        if layer.ty != LayerType::FillExtrusion
            || layer.get(orb_style::EXTRUSION_PATTERN).constant() != Some(None)
        {
            return None;
        }
        let color = layer.get(orb_style::EXTRUSION_COLOR);
        let expression = color.as_expression().filter(|e| e.reads_elevation())?;
        Some(Self::from_expression(expression))
    }

    fn from_expression(expression: &PropertyExpression) -> Result<Self, String> {
        let curve = expression.elevation_curve().map_err(str::to_string)?;
        if curve.input.len() > MAX_ELEVATION_OPS {
            return Err(format!(
                "[\"elevation\"] draws per fragment through at most {MAX_ELEVATION_OPS} operators"
            ));
        }
        let n = curve.labels.len();
        let (labels, colors, interpolation) = if curve.rgb || n < 2 {
            let colors = (0..n).map(|i| curve.stop(i)).collect();
            (curve.labels.clone(), colors, curve.interpolation)
        } else {
            let pieces = ((MAX_ELEVATION_STOPS - 1) / (n - 1)).clamp(1, PIECES);
            let mut labels = Vec::new();
            let mut colors = Vec::new();
            for (i, pair) in curve.labels.windows(2).enumerate() {
                for k in 0..pieces {
                    let u = pair[0] + (pair[1] - pair[0]) * k as f64 / pieces as f64;
                    labels.push(u);
                    colors.push(if k == 0 { curve.stop(i) } else { curve.at(u) });
                }
            }
            labels.push(curve.labels[n - 1]);
            colors.push(curve.stop(n - 1));
            (labels, colors, Some(Interpolation::Linear))
        };
        if labels.len() > MAX_ELEVATION_STOPS {
            return Err(format!(
                "[\"elevation\"] draws per fragment over at most {MAX_ELEVATION_STOPS} stops"
            ));
        }
        Ok(Self {
            input: curve.input,
            labels,
            interpolation,
            colors: colors.into_iter().map(slot).collect(),
        })
    }

    /// The property row's words beyond the family's: two per colour after
    /// the first, when any of them comes from the row.
    pub(crate) fn row_words(&self) -> usize {
        // What reads feature state alone comes from the state table.
        let rows = self.colors[1..].iter().any(|s| {
            s.as_expression()
                .is_some_and(PropertyExpression::reads_feature_properties)
        });
        if rows { 2 * (self.colors.len() - 1) } else { 0 }
    }

    /// The node's curve at `zoom`; `properties` are the node's, whose
    /// colour slots ([`colour_slot`]) give each stop's.
    pub(crate) fn paint(&self, properties: &[PaintProperty<'_>], zoom: f64) -> ElevationPaint {
        let mut paint = ElevationPaint {
            input: [(0, 0.0); MAX_ELEVATION_OPS],
            ops: self.input.len(),
            // Past the stops, never at or below an input.
            labels: [f32::MAX; MAX_ELEVATION_STOPS],
            colors: [0; MAX_ELEVATION_STOPS],
            stops: self.colors.len(),
            rows: 0,
            curve: match self.interpolation {
                None => Curve::Step,
                Some(Interpolation::Linear) => Curve::Exponential(1.0),
                Some(Interpolation::Exponential(base)) => Curve::Exponential(base as f32),
                Some(Interpolation::CubicBezier(p)) => Curve::CubicBezier(p.map(|v| v as f32)),
            },
        };
        for (to, &op) in paint.input.iter_mut().zip(&self.input) {
            *to = encode(op);
        }
        for (i, &label) in self.labels.iter().enumerate() {
            // A step's first is -inf, which the shader never compares.
            paint.labels[i] = label.max(f64::from(f32::MIN)) as f32;
            let property = &properties[colour_slot(i)];
            paint.colors[i] = if property.per_feature() {
                paint.rows |= 1 << i;
                0
            } else if property.is_state_lookup() {
                pack_color(property.default_color(zoom))
            } else {
                pack_color(property.color(zoom))
            };
        }
        paint
    }
}

/// The node property slot of stop `i`'s colour: the layer's colour slot,
/// then after height and base.
pub(crate) fn colour_slot(i: usize) -> usize {
    if i == 0 { 0 } else { 2 + i }
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

/// The shader's code, less one, and constant for an operator
/// (`extrusion.wgsl`'s `elevation_op`).
fn encode(op: ElevationOp) -> (u32, f32) {
    use ElevationOp as E;
    let (code, c) = match op {
        E::Add(c) => (0, c),
        E::Mul(c) => (1, c),
        E::Sub(c) => (2, c),
        E::SubFrom(c) => (3, c),
        E::Div(c) => (4, c),
        E::Rem(c) => (5, c),
        E::Min(c) => (6, c),
        E::Max(c) => (7, c),
        E::Abs => (8, 0.0),
        E::Floor => (9, 0.0),
        E::Ceil => (10, 0.0),
        E::Round => (11, 0.0),
    };
    (code, c as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(text: &str) -> Result<ElevationColor, String> {
        let spec = orb_style::EXTRUSION_COLOR.id().spec();
        let json = orb_expr::json::parse(text).unwrap();
        ElevationColor::from_expression(&PropertyExpression::parse(&json, &spec).unwrap())
    }

    /// The shader mixes in RGB, so a Lab or HCL curve draws in pieces,
    /// exact at their ends, as many as fit.
    #[test]
    fn other_colour_spaces_draw_in_pieces() {
        let c = curve(r#"["interpolate-hcl", ["linear"], ["elevation"], 0, "red", 80, "blue"]"#)
            .unwrap();
        let tens: Vec<f64> = (0..=8).map(|i| f64::from(i) * 10.0).collect();
        assert_eq!(c.labels, tens);
        assert_eq!(c.interpolation, Some(Interpolation::Linear));
        let c = curve(
            r#"["interpolate-lab", ["linear"], ["elevation"], 0, "red", 1, "lime", 2, "blue", 3, "white"]"#,
        )
        .unwrap();
        assert_eq!(c.labels.len(), 16);
        let c =
            curve(r#"["interpolate", ["linear"], ["elevation"], 0, "red", 80, "blue"]"#).unwrap();
        assert_eq!(c.labels, [0.0, 80.0]);
    }

    #[test]
    fn a_curve_fits_the_shader() {
        let stops: Vec<String> = (0..MAX_ELEVATION_STOPS)
            .map(|i| format!("{i}, \"red\""))
            .collect();
        let text = format!(r#"["step", ["elevation"], "blue", {}]"#, stops.join(", "));
        assert!(curve(&text).unwrap_err().contains("stops"));
        let mut input = r#"["elevation"]"#.to_string();
        for _ in 0..=MAX_ELEVATION_OPS {
            input = format!(r#"["abs", {input}]"#);
        }
        let text = format!(r#"["step", {input}, "blue", 3, "red"]"#);
        assert!(curve(&text).unwrap_err().contains("operators"));
    }
}
