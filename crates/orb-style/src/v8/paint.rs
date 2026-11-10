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

/// The `*-transition` members of a root object at `path` (`sky`, `light`)
/// whose property is one of `names`, warning about malformed ones.
pub(crate) fn root_transitions(
    cx: &mut crate::v8::Cx,
    json: Option<&orb_expr::json::Json>,
    path: &str,
    names: &[&str],
) -> Vec<(String, TransitionOverride)> {
    let Some(members) = json.and_then(|j| j.as_object()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for m in members {
        let Some(name) = m.key.strip_suffix("-transition") else {
            continue;
        };
        if !names.contains(&name) {
            continue;
        }
        match parse_transition(&m.value) {
            Some(t) => out.push((name.to_string(), t)),
            None => cx.warn(
                format!("{path}.{}", m.key),
                m.key_span,
                "expected {\"duration\", \"delay\"} in milliseconds",
            ),
        }
    }
    out
}
