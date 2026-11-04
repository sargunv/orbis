// Drape atlas upkeep: clearing the slots a frame redraws and building
// their mip levels from the level above. One instanced quad per slot in
// `slots`.

struct DrapeLevel {
    // The level's size in texels.
    size: vec2<f32>,
    // A slot's side at this level, and slots per atlas row.
    slot_px: f32,
    columns: u32,
};

@group(0) @binding(0) var<uniform> level: DrapeLevel;
@group(0) @binding(1) var<storage, read> slots: array<u32>;
// The level above, for `fs_downsample`.
@group(0) @binding(2) var source: texture_2d<f32>;
@group(0) @binding(3) var source_sampler: sampler;

struct SlotOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_slot(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> SlotOut {
    let slot = slots[instance];
    let corner = vec2<f32>(f32(vertex & 1u), f32((vertex >> 1u) & 1u));
    let origin = vec2<f32>(f32(slot % level.columns), f32(slot / level.columns)) * level.slot_px;
    let texel = origin + corner * level.slot_px;
    var out: SlotOut;
    out.uv = texel / level.size;
    out.position = vec4<f32>(out.uv.x * 2.0 - 1.0, 1.0 - out.uv.y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_clear(in: SlotOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}

// A 2×2 box filter: at an exact halving, the bilinear sample at the
// texel's centre averages its four sources.
@fragment
fn fs_downsample(in: SlotOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(source, source_sampler, in.uv, 0.0);
}
