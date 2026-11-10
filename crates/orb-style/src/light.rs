//! v8's `light`: the one light, which shades extrusions.

use orb_expr::Color;

use crate::{Property, TranslateAnchor};

/// v8's `light`. GL JS shades extrusion faces under it with an ambient
/// level `1 − intensity` and a directional one `2 · intensity` of its
/// colour (`lighting.wgsl`).
#[derive(Clone, Debug)]
pub struct Light {
    /// A viewport-anchored light turns with the map's bearing.
    pub anchor: Property<TranslateAnchor>,
    /// Radius, azimuth clockwise from north and polar angle from the
    /// zenith, in degrees; the radius scales the directional term.
    pub position: Property<[f32; 3]>,
    pub color: Property<Color>,
    pub intensity: Property<f32>,
    /// style-spec#1797: how much extrusions' shadows darken the ground,
    /// 0 to 1 (0 without the preview).
    pub shadow_intensity: Property<f32>,
    /// `dev.sargunv.orbis:ground-lighting`: how much the light shades the
    /// ground beneath the labels, 0 (off) to 1: by `color` where the sun
    /// is up and by the ambient light where it's down.
    pub ground_lighting: Property<f32>,
    /// `dev.sargunv.orbis:ambient-color` and `-intensity`: the light the
    /// ground keeps where the sun is down.
    pub ambient_color: Property<Color>,
    pub ambient_intensity: Property<f32>,
    /// `dev.sargunv.orbis:twilight`: the sun elevations, degrees, centred
    /// on the horizon, over which the ground passes from ambient to sunlit.
    pub twilight: Property<f32>,
    /// Each property's own `*-transition`, by property name.
    pub transitions: Vec<(String, crate::TransitionOverride)>,
}
