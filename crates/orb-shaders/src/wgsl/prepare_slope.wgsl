@group(0) @binding(0) var padded: texture_2d_array<f32>;

// The log2 of the Mercator world's width in metres, as GL JS's
// `getLog2WorldWidth` gives for wrapping worlds.
const LOG2_WORLD_WIDTH: f32 = 25.2562;

// Texel `id` of a job's slope layer, (dim + 2)² texels.
fn slope_texel(job: DemJob, id: vec2<u32>) -> vec4<f32> {
    let size = textureDimensions(padded).x - 2u;
    let pos = vec2<i32>(id) + 1;
    let tile_size = f32(size - 2u);

    let a = elevation_at(job, pos + vec2<i32>(-1, -1));
    let b = elevation_at(job, pos + vec2<i32>(0, -1));
    let c = elevation_at(job, pos + vec2<i32>(1, -1));
    let d = elevation_at(job, pos + vec2<i32>(-1, 0));
    let f = elevation_at(job, pos + vec2<i32>(1, 0));
    let g = elevation_at(job, pos + vec2<i32>(-1, 1));
    let h = elevation_at(job, pos + vec2<i32>(0, 1));
    let i = elevation_at(job, pos + vec2<i32>(1, 1));

    // Below zoom 15, slopes are exaggerated so hillshading shows at low
    // zooms (GL JS's `hillshade_prepare`).
    let zoom = job.zoom;
    var exaggeration_factor = 0.3;
    if zoom < 2.0 {
        exaggeration_factor = 0.4;
    } else if zoom < 4.5 {
        exaggeration_factor = 0.35;
    }
    var exaggeration = 0.0;
    if zoom < 15.0 {
        exaggeration = (zoom - 15.0) * exaggeration_factor;
    }
    let deriv = vec2<f32>((c + f + f + i) - (a + d + d + g), (g + h + h + i) - (a + b + b + c))
        * tile_size / pow(2.0, exaggeration + (3.0 + LOG2_WORLD_WIDTH - zoom));

    // Flat ground is 128/255, which 8 bits hold exactly.
    let color = clamp(vec4<f32>(deriv.x / 8.0 + 128.0 / 255.0, deriv.y / 8.0 + 128.0 / 255.0, 1.0, 1.0),
        vec4<f32>(0.0), vec4<f32>(1.0));
    return color;
}

fn elevation_at(job: DemJob, p: vec2<i32>) -> f32 {
    return dem_elevation(textureLoad(padded, p, job.layer, 0), job.unpack);
}
