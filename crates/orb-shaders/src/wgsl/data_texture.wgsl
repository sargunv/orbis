// Tables in data textures, for devices without storage buffers in the
// vertex stage (WebGL 2): each an R32Uint texture DATA_WIDTH words wide,
// word i at texel (i % DATA_WIDTH, i / DATA_WIDTH). Rows are read by
// functions the build script reflects from their structs
// (`rows_texture.wgsl`, `terrain_rows_texture.wgsl`).

// orb-render's DATA_WIDTH_LOG2.
const DATA_WIDTH_LOG2: u32 = 12u;

fn data_word(t: texture_2d<u32>, i: u32) -> u32 {
    return textureLoad(t, vec2<u32>(i & ((1u << DATA_WIDTH_LOG2) - 1u), i >> DATA_WIDTH_LOG2), 0).x;
}
