// Image processing (the `dev.sargunv.orbis:targets` preview): one pass of
// an image-process layer's operations, drawn by a full-screen triangle
// over a texture the size of the view. Texels are premultiplied; coverage
// is their alpha.
//
// A pass samples its input through a kernel: a copy, one axis of a
// Gaussian blur, a square convolution, one axis of a dilation or erosion,
// a move, or a move of each colour channel apart. The blur's second axis
// may weigh the blur's input against the blur (`sharpen`, `high-pass`). A
// blur too wide for its taps runs on the image halved: a move by nothing
// halves it (a texel at its size between four), and a blur of no taps
// brings it back, weighing it as the second axis would. Then it applies
// its pointwise
// operations in order (`pointwise`), which may read one colour ramp and
// one other target.

struct Process {
    // x: the kernel (0 copy, 1 blur, 2 convolve, 3 dilate or erode, 4
    // move, 5 split); y: a blur's linear taps each side, a convolution's side, or a
    // dilation's reach in texels; z: how a blur combines with the original
    // (0 the blur alone, 1 sharpen, 2 high-pass), or 1 to erode; w:
    // sharpen's amount.
    kernel: vec4<f32>,
    // xy: a blur's step between taps or a convolution's between weights,
    // in uv, or a dilation's axis in texels; zw: where a move takes each
    // pixel from, in uv.
    step: vec4<f32>,
    // A blur's centre weight, then (offset, weight) pairs of linear taps,
    // in steps; or a convolution's weights, row by row; or where a split
    // takes red, green and blue from, in uv, then 1 to scale those with
    // the distance from the view's centre.
    weights: array<vec4<f32>, 16>,
    // x: how many pointwise operations follow; y: the ramp's stops; zw:
    // the view's centre in pixels (the source may be a blank of another
    // size).
    count: vec4<f32>,
    // Each pointwise operation as three vectors (`pointwise`).
    ops: array<vec4<f32>, 24>,
    // Normalized device coordinates on the view to the camera-relative
    // world on the plane, homogeneous: noise is fixed to the map.
    unproject: array<vec4<f32>, 3>,
    // The ramp's stops: coverages, and colours premultiplied.
    ramp_at: array<vec4<f32>, 2>,
    ramp_colors: array<vec4<f32>, 8>,
};

@group(0) @binding(0) var<uniform> process: Process;
@group(0) @binding(1) var input: texture_2d<f32>;
// What a blur's second axis weighs itself against: the first's input.
@group(0) @binding(2) var original: texture_2d<f32>;
@group(0) @binding(3) var linear: sampler;
// The target a pass's `combine` reads.
@group(0) @binding(4) var other: texture_2d<f32>;

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

// The most (or, eroding, least) of the texels within the reach along the
// pass's axis, the view's edges extended.
fn morph(position: vec2<f32>) -> vec4<f32> {
    let reach = i32(process.kernel.y);
    let erode = process.kernel.z > 0.5;
    let axis = vec2<i32>(process.step.xy);
    let last = vec2<i32>(textureDimensions(input)) - vec2<i32>(1);
    let at = vec2<i32>(position);
    var c = textureLoad(input, at, 0);
    for (var i = -reach; i <= reach; i++) {
        let t = textureLoad(input, clamp(at + axis * i, vec2<i32>(0), last), 0);
        c = select(max(c, t), min(c, t), erode);
    }
    return c;
}

// Each channel from its own place, coverage the most of the three: a
// premultiplied channel is within its own sample's coverage, so within
// that.
fn split(uv: vec2<f32>, centred: vec2<f32>) -> vec4<f32> {
    var scale = 1.0;
    if weight(6u) > 0.5 {
        scale = length(centred) / max(length(process.count.zw), 1.0);
    }
    let r = sample_at(uv + vec2<f32>(weight(0u), weight(1u)) * scale);
    let g = sample_at(uv + vec2<f32>(weight(2u), weight(3u)) * scale);
    let b = sample_at(uv + vec2<f32>(weight(4u), weight(5u)) * scale);
    return vec4<f32>(r.r, g.g, b.b, max(max(r.a, g.a), b.a));
}

// Stripes of a band pattern a period apart, each `width` of one, box
// filtered over `fw` periods about `x`: what of a pixel they cover.
fn bands(x: f32, width: f32, fw: f32) -> f32 {
    let w = max(fw, 0.0001);
    let lo = x - w * 0.5;
    let hi = x + w * 0.5;
    let below_hi = floor(hi) * width + min(fract(hi), width);
    let below_lo = floor(lo) * width + min(fract(lo), width);
    return clamp((below_hi - below_lo) / w, 0.0, 1.0);
}

fn ramp(t: f32) -> vec4<f32> {
    let stops = u32(process.count.y);
    var c = process.ramp_colors[0];
    for (var i = 1u; i < stops; i++) {
        let a = process.ramp_at[(i - 1u) / 4u][(i - 1u) % 4u];
        let b = process.ramp_at[i / 4u][i % 4u];
        let f = clamp((t - a) / max(b - a, 0.00001), 0.0, 1.0);
        c = select(c, mix(process.ramp_colors[i - 1u], process.ramp_colors[i], f), t > a);
    }
    return c;
}

fn combine(c: vec4<f32>, mode: u32, uv: vec2<f32>) -> vec4<f32> {
    let o = textureSampleLevel(other, linear, uv, 0.0);
    switch mode {
        case 0u: {
            return c * o.a;
        }
        case 1u: {
            return c * (1.0 - o.a);
        }
        case 2u: {
            return max(c, o);
        }
        case 3u: {
            return min(c, o);
        }
        case 4u: {
            return o + c * (1.0 - o.a);
        }
        default: {
            return c + o * (1.0 - c.a);
        }
    }
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

// A pointwise operation on `c`, as three vectors (`a`, `b`, `d`): `a.x`
// names it.
// - select (1): `a.y` the tolerance (0–1), `b.rgb` the colour;
// - noise (2): `a.y` the amount, `a.z` how far the finer of its two
//   octaves has faded in, `a.w` the coarser's lattice points per world
//   unit, `b` each octave's lattice offset at the camera;
// - threshold (3): `a.y` the cutoff (0–1);
// - invert (4);
// - opacity (5): `a.y` the factor;
// - color (6): `b` the colour, straight;
// - luminance (7);
// - ramp (8): coverage through the pass's ramp;
// - combine (9): `a.y` the mode (`combine`) with the pass's other target;
// - moving noise (10): as noise, each octave the mean of two lattices
//   sliding apart, `b` and `d` their offsets;
// - gradient (11): `a.y` 1 for radial; `b.xy` a linear one's direction
//   over its length in pixels, `b.z` a radial one's over the corners';
// - bands (12): `a.y` the count, `a.z` the width.
// `centred` is the pixel's place from the view's centre, in pixels.
fn pointwise(
    c: vec4<f32>,
    a: vec4<f32>,
    b: vec4<f32>,
    d: vec4<f32>,
    world: vec2<f32>,
    uv: vec2<f32>,
    centred: vec2<f32>,
) -> vec4<f32> {
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
        case 5u: {
            return c * a.y;
        }
        case 6u: {
            return vec4<f32>(b.rgb, 1.0) * b.a * c.a;
        }
        case 7u: {
            let rgb = unpremultiplied(c);
            let l = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
            return vec4<f32>(l * c.a);
        }
        case 8u: {
            return ramp(c.a);
        }
        case 9u: {
            return combine(c, u32(a.y), uv);
        }
        case 10u: {
            let p = world * a.w;
            let coarse = gradient_noise(p + b.xy) + gradient_noise(p + d.xy);
            let fine = gradient_noise(p * 2.0 + b.zw) + gradient_noise(p * 2.0 + d.zw);
            return with_coverage(c, c.a + a.y * mix(coarse, fine, a.z) * 0.70710678);
        }
        case 11u: {
            let linear = dot(centred, b.xy) + 0.5;
            let radial = length(centred) * b.z;
            return with_coverage(c, select(linear, radial, a.y > 0.5));
        }
        case 12u: {
            let x = c.a * a.y;
            let covered = bands(x, a.z, fwidth(x));
            return with_coverage(c, select(0.0, covered, c.a > 0.0));
        }
        default: {
            return c;
        }
    }
}

@fragment
fn fs_process(in: ProcessOut) -> @location(0) vec4<f32> {
    let centred = in.position.xy - process.count.zw;
    var c: vec4<f32>;
    switch u32(process.kernel.x) {
        case 1u: {
            c = blur(in.uv);
        }
        case 2u: {
            c = convolve(in.uv);
        }
        case 3u: {
            c = morph(in.position.xy);
        }
        case 4u: {
            c = sample_at(in.uv + process.step.zw);
        }
        case 5u: {
            c = split(in.uv, centred);
        }
        default: {
            c = textureLoad(input, vec2<i32>(in.position.xy), 0);
        }
    }
    let world = world_at(in.uv);
    let count = u32(process.count.x);
    for (var i = 0u; i < count; i++) {
        let at = 3u * i;
        let a = process.ops[at];
        let b = process.ops[at + 1u];
        let d = process.ops[at + 2u];
        c = pointwise(c, a, b, d, world, in.uv, centred);
    }
    return c;
}
