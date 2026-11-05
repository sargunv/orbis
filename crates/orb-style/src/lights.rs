//! Lights (decision 0026): what v8's `light` lowers into and what the
//! scene document's `lights` replace it with.

use orb_expr::Color;

use crate::{Property, TranslateAnchor};

/// The style's lights: what v8's `light` lowers into (an ambient and a
/// directional term of GL JS's flat extrusion response) and what the scene
/// document's `lights` replace it with. Shadows and glTF's materials extend
/// this model; they don't add another.
#[derive(Clone, Debug)]
pub struct Lights {
    pub ambient: Option<AmbientLight>,
    pub directional: Option<DirectionalLight>,
    /// Whether the lights shade the whole map, its layers by their
    /// materials, and the terrain (a scene's), or only extrusions (v8's
    /// `light`, as in GL JS).
    pub terrain: bool,
    /// Layers' materials by layer id; the rest are [`LayerMaterial`]'s
    /// default.
    pub materials: Vec<(String, LayerMaterial)>,
}

/// What a layer is made of, under the scene's lights: its v8 colours are
/// its albedo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerMaterial {
    /// How much of its own colour shows whatever the lights: 0 lit fully,
    /// 1 not at all.
    pub emissive: f64,
    /// 0 mirror-smooth to 1 matte.
    pub roughness: f64,
    /// 0 a dielectric to 1 a metal.
    pub metalness: f64,
    /// An extrusion's lit windows, the share of them that glow: 0 none.
    pub windows: f64,
    /// A fill's or line's waves, as water: their RMS slope, 0 flat. They
    /// move with time, so a map showing them draws frames continuously.
    pub waves: f64,
    /// A layer that is light where the sun is down: its colour added to
    /// the map beneath, this much, from dusk to dawn, and nothing by day.
    pub night_light: f64,
}

impl Default for LayerMaterial {
    /// Matte paint: lit as a Lambertian surface.
    fn default() -> Self {
        Self {
            emissive: 0.0,
            roughness: 1.0,
            metalness: 0.0,
            windows: 0.0,
            waves: 0.0,
            night_light: 0.0,
        }
    }
}

impl Lights {
    /// A layer's material.
    pub fn material(&self, layer: &str) -> LayerMaterial {
        self.materials
            .iter()
            .find(|(id, _)| id == layer)
            .map_or_else(LayerMaterial::default, |&(_, m)| m)
    }

    /// A layer's emissive strength; 0 unless the scene sets it.
    pub fn emissive(&self, layer: &str) -> f64 {
        self.material(layer).emissive
    }
}
#[derive(Clone, Debug)]
pub struct AmbientLight {
    pub color: Property<Color>,
    pub level: Level,
}
#[derive(Clone, Debug)]
pub struct DirectionalLight {
    pub color: Property<Color>,
    pub level: Level,
    /// Viewport-anchored lights turn with the map's bearing.
    pub anchor: Property<TranslateAnchor>,
    pub direction: Direction,
    /// When it casts shadows, how much of its light they take away, 0 to
    /// 1 (the scene's `cast-shadows` and `shadow-intensity`).
    pub shadow: Option<f32>,
}
#[derive(Clone, Debug)]
pub enum Direction {
    /// Radius, azimuth clockwise from north and polar angle from the
    /// zenith, in degrees: v8's `position`, whose radius scales the light.
    Position(Property<[f32; 3]>),
    /// The sun over the camera's target at a UTC time (Unix seconds).
    Sun { time: f64 },
}
/// A light's level, `offset + scale · intensity`: lowering maps v8's
/// single intensity onto both terms (ambient `1 − i`, directional `2i`);
/// scene lights take theirs as declared.
#[derive(Clone, Debug)]
pub struct Level {
    pub intensity: Property<f32>,
    pub offset: f32,
    pub scale: f32,
}
impl Level {
    pub fn declared(intensity: f32) -> Self {
        Self {
            intensity: Property::constant(intensity),
            offset: 0.0,
            scale: 1.0,
        }
    }

    pub fn evaluate(&self, zoom: f64) -> f32 {
        self.offset + self.scale * self.intensity.evaluate(zoom)
    }
}
