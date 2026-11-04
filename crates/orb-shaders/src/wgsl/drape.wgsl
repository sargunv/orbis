// Drape atlas upkeep: clearing the slots a frame redraws (one instanced
// quad per slot in `slots`), and building their mip levels from the level
// above (`downsample`, a compute dispatch per level).

struct DrapeLevel {
    // The level's size in texels.
    size: vec2<f32>,
    // A slot's side at this level, and slots per atlas row.
    slot_px: f32,
    columns: u32,
};

@group(0) @binding(0) var<uniform> level: DrapeLevel;
@group(0) @binding(1) var<storage, read> slots: array<u32>;
// The level above, for `downsample`.
@group(0) @binding(2) var source: texture_2d<f32>;

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

// A 2×2 box filter from the level above into this one, over the slots in
// `slots`: one invocation per texel, `z` picks the slot.
@group(0) @binding(3) var destination: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = u32(level.slot_px);
    if any(id.xy >= vec2<u32>(n)) {
        return;
    }
    let slot = slots[id.z];
    let p = vec2<u32>(slot % level.columns, slot / level.columns) * n + id.xy;
    let s = vec2<i32>(p * 2u);
    let c = textureLoad(source, s, 0) + textureLoad(source, s + vec2<i32>(1, 0), 0)
        + textureLoad(source, s + vec2<i32>(0, 1), 0) + textureLoad(source, s + vec2<i32>(1, 1), 0);
    textureStore(destination, vec2<i32>(p), c * 0.25);
}
