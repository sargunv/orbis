//! v8's `sky`, lowered into the style's [`Atmosphere`]: one sky and fog
//! model.
use crate::spec::sky::{
    ATMOSPHERE_BLEND, FOG_COLOR, FOG_GROUND_BLEND, HORIZON_COLOR, HORIZON_FOG_BLEND, PROPS,
    SKY_COLOR, SKY_HORIZON_BLEND,
};
use crate::v8::{Cx, root::RootObject};
use crate::*;
use orb_expr::json::Json;

pub(crate) fn parse(cx: &mut Cx, json: &Json) -> Atmosphere {
    let sky = RootObject::parse(cx, Some(json), "sky", PROPS);
    Atmosphere {
        sky_color: sky.read(cx, &SKY_COLOR),
        horizon_color: sky.read(cx, &HORIZON_COLOR),
        fog_color: sky.read(cx, &FOG_COLOR),
        fog_ground_blend: sky.read(cx, &FOG_GROUND_BLEND),
        horizon_fog_blend: sky.read(cx, &HORIZON_FOG_BLEND),
        sky_horizon_blend: sky.read(cx, &SKY_HORIZON_BLEND),
        atmosphere_blend: sky.read(cx, &ATMOSPHERE_BLEND),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn absent_sky_is_transparent_but_empty_sky_uses_spec_defaults() {
        let absent = crate::v8::parse(r#"{"version":8,"sources":{},"layers":[]}"#).unwrap();
        assert!(absent.atmosphere.is_none());
        let defaults =
            crate::v8::parse(r#"{"version":8,"sky":{},"sources":{},"layers":[]}"#).unwrap();
        let sky = defaults.atmosphere.unwrap();
        assert_eq!(
            sky.sky_color.evaluate(0.0).straight(),
            [136.0 / 255.0, 198.0 / 255.0, 252.0 / 255.0, 1.0]
        );
        assert_eq!(sky.sky_horizon_blend.evaluate(0.0), 0.8);
    }

    #[test]
    fn sky_zoom_expressions_and_ranges_follow_the_spec() {
        let style = crate::v8::parse(r#"{"version":8,"sky":{"sky-horizon-blend":["interpolate",["linear"],["zoom"],0,0,10,1]},"sources":{},"layers":[]}"#).unwrap();
        assert_eq!(
            style.atmosphere.unwrap().sky_horizon_blend.evaluate(5.0),
            0.5
        );
        // Out of range: the default, with a warning.
        let style = crate::v8::parse(
            r#"{"version":8,"sky":{"sky-horizon-blend":2},"sources":{},"layers":[]}"#,
        )
        .unwrap();
        assert_eq!(style.warnings[0].path, "sky.sky-horizon-blend");
        assert_eq!(
            style.atmosphere.unwrap().sky_horizon_blend.evaluate(0.0),
            0.8
        );
    }

    #[test]
    fn sky_has_v8_properties_only() {
        let style = crate::v8::parse(
            r#"{"version":8,"sky":{"star-intensity":1,"star-opacity":1},"sources":{},"layers":[]}"#,
        )
        .unwrap();
        assert_eq!(style.warnings.len(), 2);
    }
}
