// GL JS's `UnitBezier.solve`.
fn unit_bezier(p: vec4<f32>, x: f32) -> f32 {
    let cx = 3.0 * p.x;
    let bx = 3.0 * (p.z - p.x) - cx;
    let ax = 1.0 - cx - bx;
    let cy = 3.0 * p.y;
    let by = 3.0 * (p.w - p.y) - cy;
    let ay = 1.0 - cy - by;
    var t = x;
    var solved = false;
    for (var i = 0; i < 8; i++) {
        let err = ((ax * t + bx) * t + cx) * t - x;
        if abs(err) < 1e-6 {
            solved = true;
            break;
        }
        let d = (3.0 * ax * t + 2.0 * bx) * t + cx;
        if abs(d) < 1e-6 {
            break;
        }
        t -= err / d;
    }
    if !solved {
        var lo = 0.0;
        var hi = 1.0;
        t = x;
        for (var i = 0; i < 32; i++) {
            let fx = ((ax * t + bx) * t + cx) * t;
            if abs(fx - x) < 1e-6 {
                break;
            }
            if x > fx {
                lo = t;
            } else {
                hi = t;
            }
            t = (hi - lo) * 0.5 + lo;
        }
    }
    return ((ay * t + by) * t + cy) * t;
}
