// What an extrusion layer darkens on the ground beneath its buildings, on
// terrain (orb-render's terrain decal): the terrain's mesh drawn again into
// the layer's target before its buildings, depth-tested against the
// terrain's own depth, as black whose alpha is the darkness the ground
// mask holds where the point falls on the mask's plane
// (`ground_mask_clip`, extrusion_ground.wgsl). It lies on the surface the
// terrain draws, wherever the DEM bends.

@group(1) @binding(1) var ground_mask: texture_2d<f32>;
@group(1) @binding(2) var ground_mask_sampler: sampler;

@fragment
fn fs_terrain_decal(in: TerrainOut) -> @location(0) vec4<f32> {
    let tile = terrain_tile_at(in.tile.x);
    let world = tile.origin + vec2<f32>(in.uv.x, -in.uv.y) * TERRAIN_EXTENT * tile.scale;
    let clip = ground_mask_clip(world);
    // Off the mask, nothing: its edge texels don't reach on.
    if clip.z < 0.0 || any(abs(clip.xy) > vec2<f32>(clip.w)) {
        return vec4<f32>(0.0);
    }
    let uv = clip.xy / clip.w * vec2<f32>(0.5, -0.5) + 0.5;
    return vec4<f32>(0.0, 0.0, 0.0, textureSampleLevel(ground_mask, ground_mask_sampler, uv, 0.0).r);
}
