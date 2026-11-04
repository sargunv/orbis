// The GPU paint interpreter (plan §8.4's `expr`), composed only into
// bytecode-class pipelines. It runs the programs `orb-expr`'s `gpu` module
// encodes into the property arena; see there for the format and the
// precision rule.
//
// A value is `vec4<u32>(tag, x, y, z)`: a number is its f64 bits in x, y
// and its f32 in z; a colour two unorm16 pairs of its straight RGBA; a
// string its dictionary id; a boolean 0 or 1. Branches compare numbers by
// their f64 bits; arithmetic runs in f32.

const EXPR_NULL: u32 = 0u;
const EXPR_NUMBER: u32 = 1u;
const EXPR_BOOL: u32 = 2u;
const EXPR_STRING: u32 = 3u;
const EXPR_COLOR: u32 = 4u;
const EXPR_ERROR: u32 = 5u;
const EXPR_ABSENT: u32 = 8u;
const EXPR_NO_STATE: u32 = 0xffffffffu;

fn expr_error() -> vec4<u32> {
    return vec4<u32>(EXPR_ERROR, 0u, 0u, 0u);
}

fn expr_bool(b: bool) -> vec4<u32> {
    return vec4<u32>(EXPR_BOOL, select(0u, 1u, b), 0u, 0u);
}

fn expr_nan() -> vec4<u32> {
    return vec4<u32>(EXPR_NUMBER, 0u, 0x7ff80000u, 0x7fc00000u);
}

fn expr_infinity(negative: bool) -> vec4<u32> {
    let sign = select(0u, 0x80000000u, negative);
    return vec4<u32>(EXPR_NUMBER, 0u, sign | 0x7ff00000u, sign | 0x7f800000u);
}

fn expr_f64_to_f32(lo: u32, hi: u32) -> f32 {
    let sign = hi & 0x80000000u;
    let e = i32((hi >> 20u) & 0x7ffu) - 1023;
    if e == 1024 {
        if ((hi & 0xfffffu) | lo) != 0u {
            return bitcast<f32>(sign | 0x7fc00000u);
        }
        return bitcast<f32>(sign | 0x7f800000u);
    }
    if e > 127 {
        return bitcast<f32>(sign | 0x7f800000u);
    }
    if e < -126 {
        return bitcast<f32>(sign);
    }
    // 1.m with the mantissa's top 41 bits, then the exponent.
    let m = 1.0 + f32(hi & 0xfffffu) * exp2(-20.0) + f32(lo >> 11u) * exp2(-41.0);
    let v = scale_exp2(m, e);
    return select(v, -v, sign != 0u);
}

fn expr_f32_to_f64(f: f32) -> vec2<u32> {
    let b = bitcast<u32>(f);
    let sign = b & 0x80000000u;
    let e = (b >> 23u) & 0xffu;
    let m = b & 0x7fffffu;
    if e == 0u {
        return vec2<u32>(0u, sign);
    }
    if e == 0xffu {
        if m != 0u {
            return vec2<u32>(0u, 0x7ff80000u);
        }
        return vec2<u32>(0u, sign | 0x7ff00000u);
    }
    return vec2<u32>(m << 29u, sign | ((e + 896u) << 20u) | (m >> 3u));
}

fn expr_number(f: f32) -> vec4<u32> {
    let d = expr_f32_to_f64(f);
    return vec4<u32>(EXPR_NUMBER, d.x, d.y, bitcast<u32>(f));
}

fn expr_f32(v: vec4<u32>) -> f32 {
    return bitcast<f32>(v.w);
}

fn expr_is_nan(v: vec4<u32>) -> bool {
    let h = v.z & 0x7fffffffu;
    return h > 0x7ff00000u || (h == 0x7ff00000u && v.y != 0u);
}

fn expr_f32_nan(f: f32) -> bool {
    return (bitcast<u32>(f) & 0x7fffffffu) > 0x7f800000u;
}

// An unsigned key in the f64 total order, with -0 as +0.
fn expr_key(v: vec4<u32>) -> vec2<u32> {
    var hi = v.z;
    if (hi & 0x7fffffffu) == 0u && v.y == 0u {
        hi = 0u;
    }
    if (hi & 0x80000000u) != 0u {
        return vec2<u32>(~v.y, ~hi);
    }
    return vec2<u32>(v.y, hi | 0x80000000u);
}

// -1, 0 or 1 for two numbers that are not NaN, exactly as f64s compare.
fn expr_order(a: vec4<u32>, b: vec4<u32>) -> i32 {
    let x = expr_key(a);
    let y = expr_key(b);
    if x.y != y.y {
        return select(1, -1, x.y < y.y);
    }
    if x.x != y.x {
        return select(1, -1, x.x < y.x);
    }
    return 0;
}

// `==` of the VM (JavaScript's `===` on null, booleans, numbers, strings).
fn expr_equal(a: vec4<u32>, b: vec4<u32>) -> bool {
    if a.x != b.x {
        return false;
    }
    switch a.x {
        case 0u: {
            return true;
        }
        case 1u: {
            return !expr_is_nan(a) && !expr_is_nan(b) && expr_order(a, b) == 0;
        }
        case 2u, 3u: {
            return a.y == b.y;
        }
        default: {
            return false;
        }
    }
}

fn expr_color(c: vec4<f32>) -> vec4<u32> {
    return vec4<u32>(EXPR_COLOR, pack2x16unorm(c.xy), pack2x16unorm(c.zw), 0u);
}

fn expr_rgba(v: vec4<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16unorm(v.y), unpack2x16unorm(v.z));
}

// Input `j` of program `p`: from the feature's inputs or its state values.
fn expr_input(p: u32, j: u32, inputs: u32, state: u32) -> vec4<u32> {
    let entry = prop_at(p + prop_at(p + 3u) + j);
    let offset = entry & 0x0fffffffu;
    let color = (entry & 0x40000000u) != 0u;
    var base = inputs + offset;
    if (entry & 0x80000000u) != 0u {
        if state == EXPR_NO_STATE {
            return vec4<u32>(EXPR_ABSENT, 0u, 0u, 0u);
        }
        base = state + offset;
    }
    if color {
        return vec4<u32>(prop_at(base), prop_at(base + 1u), prop_at(base + 2u), 0u);
    }
    let lo = prop_at(base);
    let hi = prop_at(base + 1u);
    if (hi & 0xfff80000u) == 0xfff80000u {
        return vec4<u32>(hi & 0xffu, lo, 0u, 0u);
    }
    return vec4<u32>(EXPR_NUMBER, lo, hi, bitcast<u32>(expr_f64_to_f32(lo, hi)));
}

fn expr_arg(r: ptr<function, array<vec4<u32>, 16>>, consts: u32, o: u32) -> vec4<u32> {
    if o >= 0x80000000u {
        let at = consts + (o & 0xffffu) * 4u;
        return vec4<u32>(prop_at(at), prop_at(at + 1u), prop_at(at + 2u), prop_at(at + 3u));
    }
    return (*r)[o & 15u];
}

// How far `x` is between stops `lo` and `hi` (GL JS `interpolationFactor`).
fn expr_factor(bezier: bool, params: vec4<u32>, x: f32, lo: f32, hi: f32) -> f32 {
    let d = hi - lo;
    let p = x - lo;
    if d == 0.0 {
        return 0.0;
    }
    if bezier {
        return unit_bezier(bitcast<vec4<f32>>(params), p / d);
    }
    let base = bitcast<f32>(params.x);
    if base == 1.0 {
        return p / d;
    }
    return (pow(base, p) - 1.0) / (pow(base, d) - 1.0);
}

fn expr_binary(f: u32, a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    let x = expr_f32(a);
    let y = expr_f32(b);
    if f == 3u && y == 0.0 {
        return expr_number(1.0);
    }
    if expr_is_nan(a) || expr_is_nan(b) {
        return expr_nan();
    }
    switch f {
        case 0u: {
            return expr_number(x - y);
        }
        case 1u: {
            if y == 0.0 {
                if x == 0.0 {
                    return expr_nan();
                }
                return expr_infinity((x < 0.0) != ((b.z & 0x80000000u) != 0u));
            }
            return expr_number(x / y);
        }
        case 2u: {
            if y == 0.0 {
                return expr_nan();
            }
            return expr_number(x - y * trunc(x / y));
        }
        default: {
            if abs(x) == 1.0 && (b.z & 0x7fffffffu) == 0x7ff00000u {
                return expr_nan();
            }
            if x == 0.0 {
                if y < 0.0 {
                    return expr_infinity(false);
                }
                return expr_number(0.0);
            }
            if x < 0.0 {
                if y != trunc(y) {
                    return expr_nan();
                }
                let v = pow(-x, y);
                return expr_number(select(v, -v, trunc(y * 0.5) != y * 0.5));
            }
            return expr_number(pow(x, y));
        }
    }
}

fn expr_unary(f: u32, a: vec4<u32>) -> vec4<u32> {
    let x = expr_f32(a);
    if expr_is_nan(a) {
        return expr_nan();
    }
    switch f {
        case 0u: {
            return expr_number(-x);
        }
        case 1u: {
            return expr_number(abs(x));
        }
        case 2u: {
            return expr_number(floor(x));
        }
        case 3u: {
            return expr_number(ceil(x));
        }
        case 4u: {
            // Halves away from zero, as GL JS rounds.
            return expr_number(sign(x) * floor(abs(x) + 0.5));
        }
        case 5u: {
            if x < 0.0 {
                return expr_nan();
            }
            return expr_number(sqrt(x));
        }
        case 6u, 7u, 8u: {
            if x < 0.0 {
                return expr_nan();
            }
            if x == 0.0 {
                return expr_infinity(true);
            }
            var scale = 1.0;
            if f == 7u {
                scale = 1.4426950408889634;
            }
            if f == 8u {
                scale = 0.4342944819032518;
            }
            return expr_number(log(x) * scale);
        }
        case 9u: {
            return expr_number(sin(x));
        }
        case 10u: {
            return expr_number(cos(x));
        }
        case 11u: {
            return expr_number(tan(x));
        }
        case 12u, 13u: {
            if abs(x) > 1.0 {
                return expr_nan();
            }
            return expr_number(select(acos(x), asin(x), f == 12u));
        }
        default: {
            return expr_number(atan(x));
        }
    }
}

// Runs program `p` for a feature whose inputs start at word `inputs` and
// whose state values start at `state` (EXPR_NO_STATE: no state), at `zoom`
// (from `expr_zoom`).
fn expr_eval(p: u32, inputs: u32, state: u32, zoom: vec4<u32>) -> vec4<u32> {
    var r: array<vec4<u32>, 16>;
    let n = prop_at(p);
    let consts = p + prop_at(p + 2u);
    for (var pc = 0u; pc < n; pc++) {
        let at = p + 8u + pc * 4u;
        let w0 = prop_at(at);
        let op = w0 & 0xffu;
        let aux = w0 >> 16u;
        let count = aux & 0xffu;
        let flags = aux >> 8u;
        let w1 = prop_at(at + 1u);
        let w2 = prop_at(at + 2u);
        let w3 = prop_at(at + 3u);
        let table = p + w1;
        var out = expr_error();
        switch op {
            case 1u: {
                let v = expr_input(p, w1, inputs, state);
                if (aux & 1u) != 0u {
                    out = expr_bool(v.x != EXPR_ABSENT);
                } else if v.x == EXPR_ABSENT {
                    out = vec4<u32>(EXPR_NULL, 0u, 0u, 0u);
                } else {
                    out = v;
                }
            }
            case 2u: {
                out = zoom;
            }
            case 3u: {
                let a = expr_arg(&r, consts, w1);
                if a.x == EXPR_BOOL {
                    out = expr_bool(a.y == 0u);
                }
            }
            case 4u: {
                let a = expr_arg(&r, consts, w1);
                let b = expr_arg(&r, consts, w2);
                let cmp = aux & 15u;
                if a.x != EXPR_ERROR && b.x != EXPR_ERROR {
                    if cmp < 2u {
                        out = expr_bool(expr_equal(a, b) != (cmp == 1u));
                    } else if a.x == EXPR_NUMBER && b.x == EXPR_NUMBER {
                        if expr_is_nan(a) || expr_is_nan(b) {
                            out = expr_bool(false);
                        } else {
                            let o = expr_order(a, b);
                            switch cmp {
                                case 2u: {
                                    out = expr_bool(o < 0);
                                }
                                case 3u: {
                                    out = expr_bool(o <= 0);
                                }
                                case 4u: {
                                    out = expr_bool(o > 0);
                                }
                                default: {
                                    out = expr_bool(o >= 0);
                                }
                            }
                        }
                    } else if (aux & 16u) == 0u {
                        out = expr_bool(false);
                    }
                }
            }
            case 5u: {
                let any = flags != 0u;
                out = expr_bool(!any);
                for (var i = 0u; i < count; i++) {
                    let v = expr_arg(&r, consts, prop_at(table + i));
                    if v.x != EXPR_BOOL {
                        out = expr_error();
                        break;
                    }
                    if (v.y != 0u) == any {
                        out = v;
                        break;
                    }
                }
            }
            case 6u: {
                out = expr_arg(&r, consts, w2);
                for (var i = 0u; i < count; i++) {
                    let t = expr_arg(&r, consts, prop_at(table + 2u * i));
                    if t.x != EXPR_BOOL {
                        out = expr_error();
                        break;
                    }
                    if t.y != 0u {
                        out = expr_arg(&r, consts, prop_at(table + 2u * i + 1u));
                        break;
                    }
                }
            }
            case 7u: {
                let v = expr_arg(&r, consts, w2);
                if v.x != EXPR_ERROR {
                    out = expr_arg(&r, consts, w3);
                    for (var i = 0u; i < count; i++) {
                        let label = expr_arg(&r, consts, prop_at(table + 2u * i));
                        if label.x == v.x && ((v.x == EXPR_STRING && label.y == v.y) || (v.x == EXPR_NUMBER && !expr_is_nan(v) && expr_order(label, v) == 0)) {
                            out = expr_arg(&r, consts, prop_at(table + 2u * i + 1u));
                            break;
                        }
                    }
                }
            }
            case 8u: {
                out = vec4<u32>(EXPR_NULL, 0u, 0u, 0u);
                for (var i = 0u; i < count; i++) {
                    out = expr_arg(&r, consts, prop_at(table + i));
                    if out.x != EXPR_NULL {
                        break;
                    }
                }
            }
            case 9u: {
                if count == 0u {
                    out = expr_arg(&r, consts, prop_at(table));
                } else {
                    let x = expr_arg(&r, consts, w2);
                    if x.x == EXPR_NUMBER && !expr_is_nan(x) {
                        var pick = prop_at(table);
                        for (var i = 0u; i < count; i++) {
                            if expr_order(x, expr_arg(&r, consts, prop_at(table + 1u + 2u * i))) < 0 {
                                break;
                            }
                            pick = prop_at(table + 2u + 2u * i);
                        }
                        out = expr_arg(&r, consts, pick);
                    }
                }
            }
            case 10u: {
                let last = 2u * (count - 1u);
                let x = expr_arg(&r, consts, w2);
                if count == 1u {
                    out = expr_arg(&r, consts, prop_at(table + 1u));
                } else if x.x == EXPR_NUMBER && !expr_is_nan(x) {
                    if expr_order(x, expr_arg(&r, consts, prop_at(table))) <= 0 {
                        out = expr_arg(&r, consts, prop_at(table + 1u));
                    } else if expr_order(x, expr_arg(&r, consts, prop_at(table + last))) >= 0 {
                        out = expr_arg(&r, consts, prop_at(table + last + 1u));
                    } else {
                        var i = 0u;
                        for (var j = 1u; j < count; j++) {
                            if expr_order(x, expr_arg(&r, consts, prop_at(table + 2u * j))) < 0 {
                                break;
                            }
                            i = j;
                        }
                        let lo = expr_f32(expr_arg(&r, consts, prop_at(table + 2u * i)));
                        let hi = expr_f32(expr_arg(&r, consts, prop_at(table + 2u * i + 2u)));
                        let a = expr_arg(&r, consts, prop_at(table + 2u * i + 1u));
                        let b = expr_arg(&r, consts, prop_at(table + 2u * i + 3u));
                        let t = expr_factor(flags == 1u, expr_arg(&r, consts, w3), expr_f32(x), lo, hi);
                        if a.x == EXPR_NUMBER && b.x == EXPR_NUMBER {
                            let fa = expr_f32(a);
                            out = expr_number(fa + t * (expr_f32(b) - fa));
                        } else if a.x == EXPR_COLOR && b.x == EXPR_COLOR {
                            let ca = expr_rgba(a);
                            out = expr_color(ca + t * (expr_rgba(b) - ca));
                        }
                    }
                }
            }
            case 11u: {
                var acc = select(0.0, 1.0, flags == 1u);
                var nan = false;
                var ok = true;
                for (var i = 0u; i < count; i++) {
                    let v = expr_arg(&r, consts, prop_at(table + i));
                    if v.x != EXPR_NUMBER {
                        ok = false;
                        break;
                    }
                    let f = expr_f32(v);
                    nan = nan || expr_is_nan(v);
                    switch flags {
                        case 0u: {
                            acc += f;
                        }
                        case 1u: {
                            acc *= f;
                        }
                        case 2u: {
                            acc = select(min(acc, f), f, i == 0u);
                        }
                        default: {
                            acc = select(max(acc, f), f, i == 0u);
                        }
                    }
                }
                if ok {
                    out = select(expr_number(acc), expr_nan(), nan);
                }
            }
            case 12u: {
                let a = expr_arg(&r, consts, w1);
                let b = expr_arg(&r, consts, w2);
                if a.x == EXPR_NUMBER && b.x == EXPR_NUMBER {
                    out = expr_binary(aux, a, b);
                }
            }
            case 13u: {
                let a = expr_arg(&r, consts, w1);
                if a.x == EXPR_NUMBER {
                    out = expr_unary(aux, a);
                }
            }
            case 14u: {
                for (var i = 0u; i < count; i++) {
                    let v = expr_arg(&r, consts, prop_at(table + i));
                    if v.x == EXPR_ERROR || v.x == flags {
                        out = v;
                        break;
                    }
                }
            }
            case 15u: {
                // Arguments come converted, nulls kept: `to-number` makes
                // the first null 0; anything else unconverted is skipped.
                for (var i = 0u; i < count; i++) {
                    let v = expr_arg(&r, consts, prop_at(table + i));
                    if v.x == EXPR_ERROR || v.x == flags {
                        out = v;
                        break;
                    }
                    if v.x == EXPR_NULL && flags == EXPR_NUMBER {
                        out = expr_number(0.0);
                        break;
                    }
                }
            }
            case 16u: {
                let v = expr_arg(&r, consts, w2);
                if v.x <= EXPR_STRING {
                    out = expr_bool(false);
                    for (var i = 0u; i < count; i++) {
                        if expr_equal(v, expr_arg(&r, consts, prop_at(table + i))) {
                            out = expr_bool(true);
                            break;
                        }
                    }
                }
            }
            case 17u: {
                let v = expr_arg(&r, consts, w1);
                if v.x != EXPR_ERROR {
                    out = expr_bool((v.x == (aux & 0xffu)) != (flags != 0u));
                }
            }
            default: {}
        }
        r[(w0 >> 8u) & 15u] = out;
    }
    return expr_arg(&r, consts, prop_at(p + 1u));
}

fn expr_zoom(zoom: f32, bits: vec2<u32>) -> vec4<u32> {
    return vec4<u32>(EXPR_NUMBER, bits.x, bits.y, bitcast<u32>(zoom));
}

fn expr_default(p: u32) -> vec4<u32> {
    return vec4<u32>(prop_at(p + 4u), prop_at(p + 5u), prop_at(p + 6u), prop_at(p + 7u));
}

// A property's value: its default where the program's result is an error,
// null, NaN or of the wrong type, as GL JS falls back. A number in x, or a
// premultiplied colour.
fn expr_value(p: u32, inputs: u32, state: u32, zoom: vec4<u32>, color: bool) -> vec4<f32> {
    var v = expr_eval(p, inputs, state, zoom);
    if color {
        if v.x != EXPR_COLOR {
            v = expr_default(p);
        }
        let c = expr_rgba(v);
        return vec4<f32>(c.rgb * c.a, c.a);
    }
    if v.x != EXPR_NUMBER || expr_is_nan(v) {
        v = expr_default(p);
    }
    return vec4<f32>(expr_f32(v), 0.0, 0.0, 0.0);
}
