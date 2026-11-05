//! How a paint change animates (decision 0031).

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
