// A map layer's fragment lit by a scene's lights (#91, decision 0048): its
// colour is the albedo of its node's material (the scene document's
// `roughness`, `metalness` and `emissive-strength`), shaded in place with
// `pbr_light`. Map layers face up; blending lit colours equals lighting
// blended albedo for their diffuse light, which shares one normal and
// point per pixel. Shadows fall on them as on the unlit ground: the
// ground pass takes the directional light's share away where they fall
// (`shadow_ground.wgsl`), which for matte ground is exact.
struct NodeMaterial {
    roughness: f32,
    metalness: f32,
    // The share of its albedo that shows whatever the lights.
    self_lit: f32,
    // An extrusion's windows: the share of them lit.
    windows: f32,
    // A fill's or line's waves: the water's slope.
    waves: f32,
    // How much of its colour it gives off where the sun is down.
    night_light: f32,
};
@group(2) @binding(0) var<storage, read> node_materials: array<NodeMaterial>;

struct MapOut {
    @location(0) color: vec4<f32>,
};

// A flat layer's colour shaded with its node's material where the
// Earth's up is `up`, facing `normal` (east, north and up there), its
// glint widened by a slope variance `spread`. A night light's colour is
// light instead, added to the map beneath where the sun is down.
fn map_lit(color: vec4<f32>, node: u32, up: vec3<f32>, normal: vec3<f32>, spread: f32) -> MapOut {
    let a = color.a;
    if a <= 0.0 {
        return MapOut(vec4<f32>(0.0));
    }
    let n = node_materials[node];
    let lights = scene_lights_at(up);
    if n.night_light > 0.0 {
        return MapOut(vec4<f32>(color.rgb * n.night_light * darkness(lights, up), 0.0));
    }
    var m: Material;
    m.albedo = srgb_to_linear(color.rgb / a);
    m.alpha = a;
    // GGX's α² takes the slopes' variance twice (as Beckmann's does).
    let r = clamp(n.roughness, 0.04, 1.0);
    m.roughness = min(sqrt(sqrt(r * r * r * r + 2.0 * spread)), 1.0);
    m.metallic = n.metalness;
    m.self_lit = n.self_lit;
    m.occlusion = 1.0;
    m.emitted = vec3<f32>(0.0);
    m.normal = turn_to_up(normal, up);
    return MapOut(vec4<f32>(linear_to_srgb(pbr_light(m, lights)) * a, a));
}

fn map_out(color: vec4<f32>, node: u32, up: vec3<f32>) -> MapOut {
    return map_lit(color, node, up, vec3<f32>(0.0, 0.0, 1.0), 0.0);
}

fn map_out_at(color: vec4<f32>, node: u32, ground: GroundPoint, up: vec3<f32>) -> MapOut {
    let steepness = node_materials[node].waves;
    if steepness <= 0.0 {
        return map_out(color, node, up);
    }
    let w = waves(ground, steepness);
    return map_lit(color, node, up, w.normal, w.spread);
}

// Water's surface at a ground point: its normal (east, north, up) and the
// variance of the slopes too fine to show.
struct Waves {
    normal: vec3<f32>,
    spread: f32,
};

// Wave vectors in whole cycles per ground cell, so the waves tile it:
// wavelengths from under half to a fifteenth of it, every way.
const WAVE_VECTORS = array<vec2<f32>, 6>(
    vec2<f32>(2.0, 1.0), vec2<f32>(-1.0, 3.0), vec2<f32>(4.0, -3.0),
    vec2<f32>(-5.0, -6.0), vec2<f32>(9.0, 4.0), vec2<f32>(-8.0, 13.0));

// Sines whose slopes add to an RMS `steepness`, running with time alone at
// deep water's speed (taking the cell as 128 m, which the quantum makes it
// on Earth in any unit to within a factor of two). A wave shorter than a
// few pixels fades out and its slope widens the glint instead, so far
// water doesn't alias.
fn waves(g: GroundPoint, steepness: f32) -> Waves {
    let cell = f32(GROUND_CELL) * frame.quantum;
    let px = max(max(g.width.x, g.width.y), 1e-12);
    let seconds = frame.time * 0.001;
    let amplitude = steepness * sqrt(2.0 / 6.0);
    var slope = vec2<f32>(0.0);
    var spread = 0.0;
    for (var i = 0u; i < 6u; i++) {
        let k = WAVE_VECTORS[i];
        let cycles = length(k);
        let shown = smoothstep(2.0, 6.0, cell / cycles / px);
        let speed = sqrt(9.81 * 2.0 * PBR_PI * cycles / 128.0);
        let phase = 2.0 * PBR_PI * dot(k, g.at) / cell - speed * seconds + f32(i) * 1.7;
        slope += k / cycles * amplitude * shown * cos(phase);
        spread += 0.5 * amplitude * amplitude * (1.0 - shown * shown);
    }
    return Waves(normalize(vec3<f32>(-slope, 1.0)), spread);
}

// A surface that isn't the flat map (an extrusion's): its colour as the
// unlit map shades it, its albedo (premultiplied), its normal (east,
// north, up), the share of the directional light its shadows leave, the
// ambient light reaching it, its node, and on a wall where along it and how high (metres), that
// position's change across a pixel (taken where control flow is uniform)
// and a value per building, for windows.
struct SurfaceIn {
    color: vec4<f32>,
    albedo: vec4<f32>,
    normal: vec3<f32>,
    shadow: f32,
    occlusion: f32,
    node: u32,
    wall: vec2<f32>,
    wall_width: vec2<f32>,
    seed: u32,
    up: vec3<f32>,
};

// How much of a lit window `wall` (metres along and up a wall) is on: a
// floor every 3.5 m from the second, a bay every 3 m, each window 60% of
// its bay and half its floor, lit when its hash (with the building's
// `seed`) falls under `share`. Edges are `width` (a pixel) wide, so far walls blend
// to the share rather than alias.
fn window_light(wall: vec2<f32>, width: vec2<f32>, seed: u32, share: f32) -> f32 {
    let size = vec2<f32>(3.0, 3.5);
    let cell = floor(wall / size);
    if cell.y < 1.0 {
        return 0.0;
    }
    let f = wall / size - cell;
    let w = max(width / size, vec2<f32>(1e-4));
    let inside = smoothstep(vec2<f32>(0.2) - w, vec2<f32>(0.2) + w, f)
        * (1.0 - smoothstep(vec2<f32>(0.8, 0.7) - w, vec2<f32>(0.8, 0.7) + w, f));
    let c = bitcast<vec2<u32>>(vec2<i32>(cell));
    var h = (c.x * 73856093u) ^ (c.y * 19349663u) ^ (seed * 83492791u);
    h = (h ^ (h >> 13u)) * 1274126177u;
    let on = f32(h & 0xffffu) / 65535.0 < share;
    return select(0.0, inside.x * inside.y, on);
}

// The surface shaded by the scene's lights with its node's material; lit
// windows on walls give off a warm light once the sun is down there.
fn surface_out(s: SurfaceIn) -> vec4<f32> {
    let a = s.albedo.a;
    if a <= 0.0 {
        return vec4<f32>(0.0);
    }
    let n = node_materials[s.node];
    var m: Material;
    m.albedo = srgb_to_linear(s.albedo.rgb / a);
    m.alpha = a;
    m.roughness = clamp(n.roughness, 0.04, 1.0);
    m.metallic = n.metalness;
    m.self_lit = n.self_lit;
    m.occlusion = s.occlusion;
    let normal = normalize(s.normal);
    m.normal = turn_to_up(normal, s.up);
    var lights = scene_lights_at(s.up);
    lights.shadow = s.shadow;
    m.emitted = vec3<f32>(0.0);
    let dark = darkness(lights, s.up);
    if n.windows > 0.0 && dark > 0.0 && abs(normal.z) < 0.5 {
        m.emitted = vec3<f32>(1.0, 0.72, 0.42) * window_light(s.wall, s.wall_width, s.seed, n.windows) * dark;
    }
    return vec4<f32>(linear_to_srgb(pbr_light(m, lights)) * a, a);
}
