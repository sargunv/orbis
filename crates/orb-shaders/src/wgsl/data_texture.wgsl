// Tables in data textures, for devices without storage buffers in the
// vertex stage (WebGL 2): each an R32Uint texture 2^data_width_log2 words
// wide, word i at texel (i % width, i / width). Rows are read by functions
// the build script reflects from their structs (`rows_texture.wgsl`,
// `terrain_rows_texture.wgsl`).

// orb-render's `data_width`: 4096 words, or 2048 on a device whose side is
// shorter (orb_shaders::DATA_WIDTH_LOG2).
override data_width_log2: u32 = 12u;

fn data_word(t: texture_2d<u32>, i: u32) -> u32 {
    return textureLoad(t, vec2<u32>(i & ((1u << data_width_log2) - 1u), i >> data_width_log2), 0).x;
}
