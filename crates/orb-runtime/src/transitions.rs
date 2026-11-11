//! Paint transitions (v8 `transition`, `*-transition`): a
//! paint edit eases from the property's prior value to its new one. Each
//! frame a transition runs, the plan's style is overlaid with the eased
//! values as constants, by [`PropertyId`]: the overlay copies only the
//! eased layers, and the paint writes from it, draped paint too. A new
//! style ends them. Values interpolate as GL JS's do: numbers, colours
//! (straight components) and pairs, eased cubic in-out. Per-feature
//! values change at once (feature-state changes fade in the shaders,
//! [`crate::state`]).
//!
//! Beyond GL JS and Native, which snap, a layer shown or hidden fades in
//! or out by the style's `transition`: the overlay keeps a hidden layer
//! drawing while it fades, and scales its opacity, the layer opacity of a
//! fill or line (which composites the layer once) and the per-feature
//! opacities of the rest, where those are the same for every feature. A
//! layer whose opacity varies by feature shows or hides at once. A fill or
//! line starts fading once the composite's pipeline compiled: a frame
//! waiting for it would draw nothing.

use std::sync::Arc;

use orb_expr::Kind;
use orb_style::{LayerType, PropertyId, Slot, Transition, Value, Visibility};

use crate::style::StylePlan;

#[derive(Debug, Default)]
pub(crate) struct Transitions {
    running: Vec<Running>,
    fades: Vec<Fade>,
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

/// A layer fading in or out.
#[derive(Debug)]
struct Fade {
    layer: String,
    ty: LayerType,
    /// How much of it shows at the start and end, 0 to 1.
    from: f64,
    to: f64,
    span: Option<(f64, f64)>,
    transition: Transition,
}

impl Fade {
    /// Whether it fades through its layer's composite (the layer opacity
    /// of a fill or line), which must compile before it starts.
    fn composited(&self) -> bool {
        matches!(self.ty, LayerType::Fill | LayerType::Line)
    }

    fn at(&self, now: f64) -> f64 {
        let t = self
            .span
            .map_or(0.0, |(begin, end)| ease((now - begin) / (end - begin)));
        self.from + (self.to - self.from) * t
    }
}

/// The properties a fade scales: what makes a layer of `ty` show.
fn fade_properties(ty: LayerType) -> &'static [PropertyId] {
    use orb_style as s;
    macro_rules! ids {
        ($($key:ident),*) => {{
            const IDS: &[PropertyId] = &[$(s::$key.id()),*];
            IDS
        }};
    }
    match ty {
        LayerType::Fill => ids!(FILL_LAYER_OPACITY),
        LayerType::Line => ids!(LINE_LAYER_OPACITY),
        LayerType::Circle => ids!(CIRCLE_OPACITY, CIRCLE_STROKE_OPACITY),
        LayerType::Symbol => ids!(TEXT_OPACITY, ICON_OPACITY),
        LayerType::FillExtrusion => ids!(EXTRUSION_OPACITY),
        LayerType::Raster => ids!(RASTER_OPACITY),
        LayerType::Heatmap => ids!(HEATMAP_OPACITY),
        LayerType::Background => ids!(BACKGROUND_OPACITY),
        LayerType::ColorRelief => ids!(COLOR_RELIEF_OPACITY),
        // A hillshade without exaggeration shades nothing.
        LayerType::Hillshade => ids!(HILLSHADE_EXAGGERATION),
    }
}

/// The transition a paint change of `layer`'s property `id` eases by: its
/// own `*-transition` over the style's `transition`.
pub(crate) fn transition_of(
    style: &orb_style::Style,
    layer: &orb_style::Layer,
    id: PropertyId,
) -> Transition {
    layer
        .transitions
        .iter()
        .find(|(n, _)| n == id.name())
        .map_or(style.transition, |(_, t)| t.over(style.transition))
}

/// GL JS's `easeCubicInOut`.
pub(crate) fn ease(t: f64) -> f64 {
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
        // A zero transition is kept too: the one in force at the next
        // frame decides, and a zero one then ends it at once.
        if per_feature(prior.kind()) {
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

    /// Layer `layer` (of `ty`, whose slots are `slot`) was shown or hidden
    /// after the frame at `shown`: it fades by `transition` from the next
    /// frame, unless its opacity varies by feature.
    pub(crate) fn fade(
        &mut self,
        (layer, ty): (&str, LayerType),
        slot: impl Fn(PropertyId) -> Slot,
        shown: bool,
        transition: Transition,
        now: f64,
    ) {
        let to = f64::from(u8::from(shown));
        let from = match self
            .fades
            .iter()
            .position(|f| f.layer == layer && f.ty == ty)
        {
            Some(i) => self.fades.remove(i).at(now),
            None => 1.0 - to,
        };
        let uniform = fade_properties(ty)
            .iter()
            .all(|&id| !per_feature(slot(id).kind()));
        if transition.duration + transition.delay <= 0.0 || !uniform || from == to {
            return;
        }
        self.fades.push(Fade {
            layer: layer.to_string(),
            ty,
            from,
            to,
            span: None,
            transition,
        });
    }

    /// Whether layer `layer` still shows, fading out, though hidden.
    pub(crate) fn fading_out(&self, layer: &str) -> bool {
        self.fades.iter().any(|f| f.layer == layer && f.to == 0.0)
    }

    /// Whether a fade waits to start for the layer composite.
    pub(crate) fn awaits_composite(&self) -> bool {
        self.fades
            .iter()
            .any(|f| f.span.is_none() && f.composited())
    }

    /// Whether a paint transition or a fade runs.
    pub(crate) fn running(&self) -> bool {
        !self.running.is_empty() || !self.fades.is_empty()
    }

    /// Whether a paint transition runs, as GL JS would run it.
    pub(crate) fn easing_paint(&self) -> bool {
        !self.running.is_empty()
    }

    /// The style overlaid with the eased values at `zoom` and `now`, while
    /// a transition runs; finished ones end. Fades through the layer
    /// composite start once it is `composite_ready`, showing what they
    /// showed until then.
    pub(crate) fn overlay(
        &mut self,
        plan: &StylePlan,
        zoom: f64,
        now: f64,
        composite_ready: bool,
    ) -> Option<StylePlan> {
        for r in &mut self.running {
            if r.span.is_none() {
                // It eases by the transition in force when it starts, as
                // GL JS and Native resolve it at their next update: one
                // changed after the edit, before this frame, holds.
                if let Some(layer) = plan.style.layers.iter().find(|l| l.id == r.layer) {
                    r.transition = transition_of(&plan.style, layer, r.id);
                }
                let t = r.transition;
                r.span = Some((now + t.delay, now + t.delay + t.duration));
            }
        }
        self.running
            .retain(|r| r.span.is_some_and(|(_, end)| end > now));
        for f in &mut self.fades {
            let t = f.transition;
            if f.span.is_none() && (composite_ready || !f.composited()) {
                f.span = Some((now + t.delay, now + t.delay + t.duration));
            }
        }
        self.fades
            .retain(|f| f.span.is_none_or(|(_, end)| end > now));
        self.ran |= self.running();
        if !self.running() {
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
        for f in &self.fades {
            let Some(layer) = style
                .layers
                .iter_mut()
                .find(|l| l.id == f.layer && l.ty == f.ty)
            else {
                continue;
            };
            let layer = Arc::make_mut(layer);
            let shown = f.at(now);
            // At its ends it shows or hides as such, needing no composite.
            if shown == 0.0 {
                layer.visibility = Visibility::None;
                continue;
            }
            layer.visibility = Visibility::Visible;
            if shown == 1.0 {
                continue;
            }
            for &id in fade_properties(f.ty) {
                let slot = layer.slot_mut(id);
                if let Value::Number(n) = id.value(slot, zoom) {
                    slot.set_constant(Value::Number(n * shown));
                }
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
