// Hillshades: a node's paint (`HillshadeNode`), and its table, the colour
// it shades every slope (`hillshade_table.wgsl` draws it from the paint).
// What draws a hillshade reads its slopes from the DEM prepare pass
// (`prepare_slope.wgsl`: (dim + 2)² texels a tile, a one-texel border
// around it, as 8-bit derivatives about 128/255) and looks their colour up
// in the table: a sample there in place of GL JS's per-pixel `hillshade`
// fragment shader, whose arctangents, powers and sines over every pixel
// cost a 5120×2880 frame of the Globe over Nepal 0.7 ms of 7 for its two
// full-screen hillshades (M6).

const STANDARD: u32 = 0u;
const COMBINED: u32 = 1u;
const IGOR: u32 = 2u;
const MULTIDIRECTIONAL: u32 = 3u;
const BASIC: u32 = 4u;

struct HillshadeNode {
    accent: vec4<f32>,
    shadows: array<vec4<f32>, 4>,
    highlights: array<vec4<f32>, 4>,
    // Radians, per light.
    altitudes: vec4<f32>,
    // Radians, per light, with the map's bearing for viewport anchoring.
    azimuths: vec4<f32>,
    exaggeration: f32,
    method: u32,
    lights: u32,
    index: u32,
    nearest: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// The table's texels a side (orb-render's `HILLSHADE_TABLE_SIDE`).
const HILLSHADE_TABLE_SIDE: f32 = 256.0;

// Where the table holds a slope's colour: each derivative `d` squeezed into
// (−1, 1) as d / (1 + |d|), finest at the gentle slopes most of a map is
// (a texel every 1/128 there, four times as fine as the 8-bit slopes) and
// reaching the steepest; texel i at (2i + 1) / side − 1, so that sampling
// between texels interpolates their colours, within a level of the colour
// computed where it falls.
fn hillshade_table_uv(deriv: vec2<f32>) -> vec2<f32> {
    return deriv / (1.0 + abs(deriv)) * 0.5 + 0.5;
}

// The slope at table position `uv` (`hillshade_table_uv`'s inverse).
fn hillshade_table_slope(uv: vec2<f32>) -> vec2<f32> {
    let u = uv * 2.0 - 1.0;
    return u / (1.0 - abs(u));
}

// The slope of prepared texel `pixel` where Mercator stretches it by
// `stretch`: slopes shrink by the cosine of the latitude, so `stretch` is
// its inverse (`mercator_stretch`, GL JS #4807).
fn hillshade_slope(pixel: vec4<f32>, stretch: f32) -> vec2<f32> {
    return (pixel.rg - 128.0 / 255.0) * 8.0 * stretch;
}

// Mercator's slope stretch at `latitude` degrees: the inverse of its cosine.
fn mercator_stretch(latitude: f32) -> f32 {
    return 1.0 / cos(radians(latitude));
}
