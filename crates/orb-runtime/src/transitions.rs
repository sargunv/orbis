//! Paint transitions (v8 `transition`, `*-transition`; decision 0031): a
//! paint edit eases from the property's prior value to its new one. Each
//! frame a transition runs, the plan's style is overlaid with the eased
//! values as constants, by [`PropertyId`]: the overlay copies only the
//! eased layers, and the paint writes from it. A new style ends them.
//! Values interpolate as GL JS's do: numbers, colours (premultiplied) and
//! pairs, eased cubic in-out. Per-feature values (data or feature state)
//! change at once, which is also what GL JS does with feature state.

use std::sync::Arc;

use orb_expr::Kind;
use orb_style::{PropertyId, Slot, Transition, Value};

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
    id: PropertyId,
    prior: Slot,
    /// When it eases, in frame time; `None` until the first frame after
    /// the edit, when its delay starts.
    span: Option<(f64, f64)>,
    transition: Transition,
}

impl Running {
    /// How far it has eased at `now`; a transition not started yet hasn't.
    fn at(&self, now: f64) -> f64 {
        self.span
            .map_or(0.0, |(begin, end)| ease((now - begin) / (end - begin)))
    }
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
    /// A paint change of `layer`'s property `id` from `prior` (its slot
    /// before the edit), made after the frame at `shown` and `zoom`; it
    /// starts with the next frame.
    pub(crate) fn start(
        &mut self,
        layer: &str,
        id: PropertyId,
        prior: &Slot,
        transition: Transition,
        (shown, zoom): (f64, f64),
    ) {
        if !id.transitions() {
            return;
        }
        // A transition under way restarts from where it stands, as GL JS's
        // chain of priors does: its prior eased toward what it was heading
        // for, the property before this edit.
        let prior = match self
            .running
            .iter()
            .position(|r| r.layer == layer && r.id == id)
        {
            Some(i) => {
                let r = self.running.remove(i);
                let t = r.at(shown);
                let mut current = prior.clone();
                mix(id, &mut current, &r.prior, zoom, t);
                current
            }
            None => prior.clone(),
        };
        if transition.duration + transition.delay <= 0.0 || per_feature(prior.kind()) {
            return;
        }
        self.running.push(Running {
            layer: layer.to_string(),
            id,
            prior,
            span: None,
            transition,
        });
    }

    pub(crate) fn running(&self) -> bool {
        !self.running.is_empty()
    }

    /// The style overlaid with the eased values at `zoom` and `now`, while
    /// a transition runs; finished ones end.
    pub(crate) fn overlay(&mut self, plan: &StylePlan, zoom: f64, now: f64) -> Option<StylePlan> {
        for r in &mut self.running {
            let t = r.transition;
            r.span
                .get_or_insert((now + t.delay, now + t.delay + t.duration));
        }
        self.running
            .retain(|r| r.span.is_some_and(|(_, end)| end > now));
        self.ran |= !self.running.is_empty();
        if self.running.is_empty() {
            return None;
        }
        let mut style = (*plan.style).clone();
        for r in &self.running {
            // A replaced style may reuse the id for another layer type.
            let Some(layer) = style
                .layers
                .iter_mut()
                .find(|l| l.id == r.layer && l.ty == r.id.layer_type())
            else {
                continue;
            };
            let t = r.at(now);
            mix(r.id, Arc::make_mut(layer).slot_mut(r.id), &r.prior, zoom, t);
        }
        Some(StylePlan {
            style: Arc::new(style),
            ..plan.clone()
        })
    }
}

/// Replaces `value`, the new property, with its value at `zoom` eased from
/// `prior` by `t`; per-feature values are left as they are.
fn mix(id: PropertyId, value: &mut Slot, prior: &Slot, zoom: f64, t: f64) {
    if per_feature(value.kind()) || per_feature(prior.kind()) {
        return;
    }
    let lerp = |a: f64, b: f64| a + (b - a) * t;
    let eased = match (id.value(prior, zoom), id.value(value, zoom)) {
        (Value::Number(a), Value::Number(b)) => Value::Number(lerp(a, b)),
        (Value::Color(a), Value::Color(b)) => Value::Color(a.mix(&b, t)),
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => Value::Array(
            a.iter()
                .zip(b.iter())
                .map(|(a, b)| match (a.as_f64(), b.as_f64()) {
                    (Some(a), Some(b)) => Value::Number(lerp(a, b)),
                    _ => b.clone(),
                })
                .collect(),
        ),
        _ => return,
    };
    value.set_constant(eased);
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
