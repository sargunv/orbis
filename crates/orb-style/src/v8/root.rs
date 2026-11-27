//! Root objects (`light`, `sky`): their members checked against a spec
//! table and their properties read typed.

use crate::v8::{Cx, layer::typed_property_at, paint::parse_transition};
use crate::*;
use orb_expr::json::Json;

/// A root object (`light`, `sky`) checked against its spec table: its
/// properties read typed, and the `*-transition` of each interpolated
/// one.
pub(crate) struct RootObject<'a> {
    json: Option<&'a Json>,
    path: &'static str,
    props: &'static [Prop],
    /// Each property's own `*-transition`, by property name.
    pub transitions: Vec<(String, TransitionOverride)>,
}

impl<'a> RootObject<'a> {
    /// The object at `path` (`None`: absent), warning of members that
    /// are neither one of `props` nor an interpolated one's transition,
    /// and of malformed transitions.
    pub(crate) fn parse(
        cx: &mut Cx,
        json: Option<&'a Json>,
        path: &'static str,
        props: &'static [Prop],
    ) -> Self {
        let mut transitions: Vec<(String, TransitionOverride)> = Vec::new();
        let members = json.and_then(|j| cx.object(j, path)).unwrap_or_default();
        for m in members {
            let find = |name: &str| props.iter().find(|p| p.name == name);
            let transition = (m.key.strip_suffix("-transition"))
                .and_then(find)
                .filter(|p| p.interpolated);
            if let Some(prop) = transition {
                transitions.retain(|(name, _)| name != prop.name);
                match parse_transition(&m.value) {
                    Some(t) => transitions.push((prop.name.to_string(), t)),
                    None => cx.warn(
                        format!("{path}.{}", m.key),
                        m.key_span,
                        "expected {\"duration\", \"delay\"} in milliseconds",
                    ),
                }
            } else if find(&m.key).is_none() {
                cx.warn_unknown(path, m);
            }
        }
        Self {
            json,
            path,
            props,
            transitions,
        }
    }

    /// `prop`, one of the object's, at its value or its default.
    pub(crate) fn read<T: PropertyValue>(&self, cx: &mut Cx, prop: &Prop) -> Property<T> {
        debug_assert!(self.props.iter().any(|p| p.name == prop.name));
        let value = self.json.and_then(|j| j.get(prop.name));
        typed_property_at(cx, value, &format!("{}.{}", self.path, prop.name), prop)
    }
}
