//! The atmosphere: what v8's `sky` lowers into.

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
    /// style-spec#1889: the colour behind the stars and the sky, and the
    /// stars' opacity; transparent and 0 unless its preview is on.
    pub backdrop_color: Property<Color>,
    pub star_opacity: Property<f32>,
}
