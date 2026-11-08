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
}
