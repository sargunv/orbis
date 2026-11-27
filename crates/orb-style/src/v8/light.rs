//! v8's `light`, parsed into the style's [`Light`].
use crate::spec::light::{ANCHOR, COLOR, INTENSITY, POSITION, PROPS};
use crate::v8::{Cx, root::RootObject};
use crate::*;
use orb_expr::json::Json;

/// v8's default `light`: every property at its spec default.
pub fn default_light() -> Light {
    parse(&mut Cx::default(), None)
}

pub(crate) fn parse(cx: &mut Cx, json: Option<&Json>) -> Light {
    let light = RootObject::parse(cx, json, "light", PROPS);
    Light {
        anchor: light.read(cx, &ANCHOR),
        position: light.read(cx, &POSITION),
        color: light.read(cx, &COLOR),
        intensity: light.read(cx, &INTENSITY),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn interpolated_light_properties_take_transitions() {
        let style = crate::v8::parse(
            r#"{"version":8,"light":{"color":"red","color-transition":{"duration":300},
              "intensity-transition":{"delay":-1},"anchor-transition":{"duration":1},"shadow":1},
              "sources":{},"layers":[]}"#,
        )
        .unwrap();
        let light = &style.light;
        assert_eq!(light.color.transition.and_then(|t| t.duration), Some(300.0));
        assert!(light.intensity.transition.is_none() && light.position.transition.is_none());
        // A malformed transition; the anchor, which doesn't interpolate,
        // has none to take; a key the spec doesn't define.
        let paths: Vec<_> = style.warnings.iter().map(|w| w.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "light.intensity-transition",
                "light.anchor-transition",
                "light.shadow"
            ]
        );
    }
}
