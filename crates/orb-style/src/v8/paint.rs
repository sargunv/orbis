//! Paint transitions: how a paint change animates.

use crate::*;
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
