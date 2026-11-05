//! Paint transitions (v8 `transition`, `*-transition`; decision 0031): a
//! paint edit eases from the property's prior value to its new one. Each
//! frame a transition runs, the plan's style is overlaid with the eased
//! values as constants, through the IR's by-name paint
//! ([`orb_v8::LayerKind::paint_mut`]), and the paint writes from that.
//! Values interpolate as GL JS's do: numbers, colours (premultiplied) and
//! pairs, eased cubic in-out. Per-feature values (data or feature state)
//! change at once, which is also what GL JS does with feature state.

use std::sync::Arc;

use orb_expr::Kind;
use orb_v8::{Color, PaintMut, PaintRef, PaintValue, Property, Transition};

use crate::style::StylePlan;

#[derive(Debug, Default)]
pub(crate) struct Transitions {
    running: Vec<Running>,
    /// A transition ran in the last frame: the next writes final values.
    pub(crate) ran: bool,
}

#[derive(Debug)]
struct Running {
    layer: String,
    name: String,
    prior: PaintValue,
    begin: f64,
    end: f64,
}

/// GL JS's `easeCubicInOut`.
fn ease(t: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    let (t2, t3) = (t * t, t * t * t);
    4.0 * if t < 0.5 {
        t3
    } else {
        3.0 * (t - t2) + t3 - 0.75
    }
}

fn per_feature(kind: Kind) -> bool {
    !matches!(kind, Kind::Constant | Kind::Camera)
}

impl Transitions {
    /// A paint change of `layer`'s `name` from `prior` (the property
    /// before the edit), at `now` and `zoom`.
    pub(crate) fn start(
        &mut self,
        layer: &str,
        name: &str,
        prior: Option<PaintRef<'_>>,
        transition: Transition,
        (now, zoom): (f64, f64),
    ) {
        let Some(prior) = prior.map(|p| p.to_owned()) else {
            return;
        };
        // A transition under way restarts from where it stands, as GL JS's
        // chain of priors does: its prior eased toward what it was heading
        // for, the property before this edit.
        let prior = match self
            .running
            .iter()
            .position(|r| r.layer == layer && r.name == name)
        {
            Some(i) => {
                let r = self.running.remove(i);
                let t = ease((now - r.begin) / (r.end - r.begin));
                let mut current = prior.clone();
                mix(current.as_mut(), &r.prior, zoom, t);
                current
            }
            None => prior,
        };
        if transition.duration + transition.delay <= 0.0 || per_feature(prior.kind()) {
            return;
        }
        self.running.push(Running {
            layer: layer.to_string(),
            name: name.to_string(),
            prior,
            begin: now + transition.delay,
            end: now + transition.delay + transition.duration,
        });
    }

    pub(crate) fn running(&self) -> bool {
        !self.running.is_empty()
    }

    /// The style overlaid with the eased values at `zoom` and `now`, while
    /// a transition runs; finished ones end.
    pub(crate) fn overlay(&mut self, plan: &StylePlan, zoom: f64, now: f64) -> Option<StylePlan> {
        self.running.retain(|r| r.end > now);
        self.ran |= !self.running.is_empty();
        if self.running.is_empty() {
            return None;
        }
        let mut style = (*plan.style).clone();
        for r in &self.running {
            let Some(layer) = style.layers.iter_mut().find(|l| l.id == r.layer) else {
                continue;
            };
            let t = ease((now - r.begin) / (r.end - r.begin));
            if let Some(value) = layer.kind.paint_mut(&r.name) {
                mix(value, &r.prior, zoom, t);
            }
        }
        Some(StylePlan {
            style: Arc::new(style),
            ..plan.clone()
        })
    }
}

/// Replaces `value`, the new property, with its value at `zoom` eased from
/// `prior` by `t`; per-feature values are left as they are.
fn mix(value: PaintMut<'_>, prior: &PaintValue, zoom: f64, t: f64) {
    let t32 = t as f32;
    match (value, prior) {
        (PaintMut::Number(to), PaintValue::Number(from)) if !per_feature(to.kind()) => {
            let (a, b) = (from.evaluate(zoom), to.evaluate(zoom));
            *to = Property::constant(a + (b - a) * t32);
        }
        (PaintMut::Color(to), PaintValue::Color(from)) if !per_feature(to.kind()) => {
            let (a, b) = (from.evaluate(zoom), to.evaluate(zoom));
            *to = Property::constant(a.mix(&b, t));
        }
        (PaintMut::OptionalColor(to), PaintValue::OptionalColor(from))
            if !per_feature(to.kind()) =>
        {
            if let (Some(a), Some(b)) = (from.evaluate(zoom), to.evaluate(zoom)) {
                *to = Property::constant(Some::<Color>(a.mix(&b, t)));
            }
        }
        (PaintMut::Pair(to), PaintValue::Pair(from)) if !per_feature(to.kind()) => {
            let (a, b) = (from.evaluate(zoom), to.evaluate(zoom));
            *to = Property::constant([a[0] + (b[0] - a[0]) * t32, a[1] + (b[1] - a[1]) * t32]);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn easing_is_gl_js_cubic_in_out() {
        assert_eq!(ease(0.0), 0.0);
        assert_eq!(ease(1.0), 1.0);
        assert!((ease(0.5) - 0.5).abs() < 1e-12);
        assert!(ease(0.25) < 0.25 && ease(0.75) > 0.75);
    }
}
