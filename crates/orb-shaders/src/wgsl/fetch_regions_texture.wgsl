// `fetch_regions.wgsl` over a data texture (`data_texture.wgsl`).

@group(0) @binding(2) var regions: texture_2d<u32>;

fn region_at(i: u32) -> vec4<u32> {
    return vec4<u32>(data_word(regions, 4u * i), data_word(regions, 4u * i + 1u),
        data_word(regions, 4u * i + 2u), data_word(regions, 4u * i + 3u));
}
