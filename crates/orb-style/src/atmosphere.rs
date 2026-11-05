//! The atmosphere: what v8's `sky` lowers into and the
//! scene document's `atmosphere` sets.

use orb_expr::Color;

use crate::Property;

#[derive(Clone, Debug)]
/// The style's atmosphere: the sky above the horizon and the fog over
/// terrain.
pub struct Atmosphere {
    pub sky_color: Property<Color>,
    pub horizon_color: Property<Color>,
    pub fog_color: Property<Color>,
    pub fog_ground_blend: Property<f32>,
    pub horizon_fog_blend: Property<f32>,
    pub sky_horizon_blend: Property<f32>,
    pub atmosphere_blend: Property<f32>,
    /// The stars' brightness, 0 (none) to 1, behind the sky.
    pub stars: Property<f32>,
    /// The colour behind everything, stars before it: where neither the
    /// map nor the sky covers it. Transparent shows the host's backing.
    pub backdrop: Property<Color>,
}
