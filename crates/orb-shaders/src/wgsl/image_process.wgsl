// Image processing (the `dev.sargunv.orbis:targets` preview): one pass of
// an image-process layer's operations, drawn by a full-screen triangle
// over a texture the size of the view. Texels are premultiplied; coverage
// is their alpha.
//
// A pass samples its input through a kernel: a copy, one axis of a
// Gaussian blur, or a square convolution. The blur's second axis may weigh
// the pass's original against the blur (`sharpen`, `high-pass`). Then it
// applies its pointwise operations in order: `select`, `noise`,
// `threshold` and `invert`.

struct Process {
    // x: the kernel (0 copy, 1 blur, 2 convolve); y: a blur's linear taps
    // each side, or a convolution's side; z: how a blur combines with the
    // original (0 the blur alone, 1 sharpen, 2 high-pass); w: sharpen's
    // amount.
    kernel: vec4<f32>,
    // xy: a blur's step between taps, or a convolution's between weights,
    // in uv.
    step: vec4<f32>,
    // A blur's centre weight, then (offset, weight) pairs of linear taps,
    // in steps; or a convolution's weights, row by row.
    weights: array<vec4<f32>, 16>,
    // x: how many pointwise operations follow.
    count: vec4<f32>,
    // Each pointwise operation as two vectors (`pointwise`).
    ops: array<vec4<f32>, 16>,
    // Normalized device coordinates on the view to the camera-relative
    // world on the plane, homogeneous: noise is fixed to the map.
    unproject: array<vec4<f32>, 3>,
};

@group(0) @binding(0) var<uniform> process: Process;
@group(0) @binding(1) var input: texture_2d<f32>;
// What a blur's second axis weighs itself against: the first's input.
@group(0) @binding(2) var original: texture_2d<f32>;
@group(0) @binding(3) var linear: sampler;

struct ProcessOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_process(@builtin(vertex_index) index: u32) -> ProcessOut {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u)) * 2.0 - 1.0;
    var out: ProcessOut;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    return out;
}

fn sample_at(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(input, linear, uv, 0.0);
}

// A premultiplied texel's colour, white where it has no coverage.
fn unpremultiplied(c: vec4<f32>) -> vec3<f32> {
    if c.a <= 0.0 {
        return vec3<f32>(1.0);
    }
    return clamp(c.rgb / c.a, vec3<f32>(0.0), vec3<f32>(1.0));
}

// `c`'s colour at coverage `a`.
fn with_coverage(c: vec4<f32>, a: f32) -> vec4<f32> {
    let coverage = clamp(a, 0.0, 1.0);
    return vec4<f32>(unpremultiplied(c) * coverage, coverage);
}

fn weight(i: u32) -> f32 {
    return process.weights[i / 4u][i % 4u];
}

fn blur(uv: vec2<f32>) -> vec4<f32> {
    let step = process.step.xy;
    var sum = sample_at(uv) * weight(0u);
    let taps = u32(process.kernel.y);
    for (var i = 0u; i < taps; i++) {
        let offset = weight(1u + 2u * i);
        let w = weight(2u + 2u * i);
        sum += (sample_at(uv + step * offset) + sample_at(uv - step * offset)) * w;
    }
    switch u32(process.kernel.z) {
        case 1u: {
            let o = textureSampleLevel(original, linear, uv, 0.0);
            let s = clamp(o + (o - sum) * process.kernel.w, vec4<f32>(0.0), vec4<f32>(1.0));
            return vec4<f32>(min(s.rgb, vec3<f32>(s.a)), s.a);
        }
        case 2u: {
            let o = textureSampleLevel(original, linear, uv, 0.0);
            return with_coverage(o, o.a - sum.a);
        }
        default: {
            return sum;
        }
    }
}

fn convolve(uv: vec2<f32>) -> vec4<f32> {
    let side = i32(process.kernel.y);
    let half = side / 2;
    var sum = vec4<f32>(0.0);
    for (var y = 0; y < side; y++) {
        for (var x = 0; x < side; x++) {
            let offset = vec2<f32>(f32(x - half), f32(y - half)) * process.step.xy;
            sum += sample_at(uv + offset) * weight(u32(y * side + x));
        }
    }
    let s = clamp(sum, vec4<f32>(0.0), vec4<f32>(1.0));
    return vec4<f32>(min(s.rgb, vec3<f32>(s.a)), s.a);
}

// A hash of a lattice point, repeating every 256 points.
fn lattice_hash(p: vec2<f32>) -> f32 {
    let q = vec2<u32>(vec2<i32>(p) & vec2<i32>(255));
    var h = q.x * 374761393u + q.y * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 65535u) / 65535.0;
}

fn corner(i: vec2<f32>, f: vec2<f32>, o: vec2<f32>) -> f32 {
    let angle = lattice_hash(i + o) * 6.2831853;
    return dot(vec2<f32>(cos(angle), sin(angle)), f - o);
}

// Gradient noise in about -1 to 1.
fn gradient_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let bottom = mix(corner(i, f, vec2<f32>(0.0, 0.0)), corner(i, f, vec2<f32>(1.0, 0.0)), u.x);
    let top = mix(corner(i, f, vec2<f32>(0.0, 1.0)), corner(i, f, vec2<f32>(1.0, 1.0)), u.x);
    return mix(bottom, top, u.y) * 1.4142135;
}

// The camera-relative world position on the plane under `uv`.
fn world_at(uv: vec2<f32>) -> vec2<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0);
    let h = vec3<f32>(
        dot(process.unproject[0].xyz, ndc),
        dot(process.unproject[1].xyz, ndc),
        dot(process.unproject[2].xyz, ndc),
    );
    return h.xy / h.z;
}

// A pointwise operation, as two vectors (`a`, `b`): `a.x` names it.
// - select (1): `a.y` the tolerance (0–1), `b.rgb` the colour;
// - noise (2): `a.y` the amount, `a.z` how far the finer of its two
//   octaves has faded in, `a.w` the coarser's lattice points per world
//   unit, `b` each octave's lattice offset at the camera;
// - threshold (3): `a.y` the cutoff (0–1);
// - invert (4).
fn pointwise(c: vec4<f32>, a: vec4<f32>, b: vec4<f32>, world: vec2<f32>) -> vec4<f32> {
    switch u32(a.x) {
        case 1u: {
            let rgb = unpremultiplied(c);
            let d = max(max(abs(rgb.r - b.r), abs(rgb.g - b.g)), abs(rgb.b - b.b));
            return c * select(0.0, 1.0, d <= a.y);
        }
        case 2u: {
            let coarse = gradient_noise(world * a.w + b.xy);
            let fine = gradient_noise(world * a.w * 2.0 + b.zw);
            return with_coverage(c, c.a + a.y * mix(coarse, fine, a.z));
        }
        case 3u: {
            let w = max(fwidth(c.a) * 0.5, 0.001);
            return with_coverage(c, smoothstep(a.y - w, a.y + w, c.a));
        }
        case 4u: {
            return with_coverage(c, 1.0 - c.a);
        }
        default: {
            return c;
        }
    }
}

@fragment
fn fs_process(in: ProcessOut) -> @location(0) vec4<f32> {
    var c: vec4<f32>;
    switch u32(process.kernel.x) {
        case 1u: {
            c = blur(in.uv);
        }
        case 2u: {
            c = convolve(in.uv);
        }
        default: {
            c = textureLoad(input, vec2<i32>(in.position.xy), 0);
        }
    }
    let world = world_at(in.uv);
    let count = u32(process.count.x);
    for (var i = 0u; i < count; i++) {
        c = pointwise(c, process.ops[2u * i], process.ops[2u * i + 1u], world);
    }
    return c;
}
