// A node's interpreted paint (plan §11.3 GpuBytecode, `expr.wgsl`): bit i
// of `mask` runs slot i's program, at `programs + props[programs + i]`. A
// feature's input columns follow its family's words in its property row;
// its raw state values start `state` words into its state block.
struct GpuPaint {
    mask: u32,
    programs: u32,
    state: u32,
    _pad: u32,
};

fn gpu_slot(paint: GpuPaint, slot: u32) -> bool {
    return (paint.mask & (1u << slot)) != 0u;
}

fn gpu_program(paint: GpuPaint, slot: u32) -> u32 {
    return paint.programs + prop_at(paint.programs + slot);
}

fn gpu_state(paint: GpuPaint, block: u32) -> u32 {
    if block == NO_STATE {
        return NO_STATE;
    }
    return block + paint.state;
}

// Every interpreted slot's value for the feature whose inputs start at
// `inputs`: a number in x, or a premultiplied colour for the slots in
// `colors`. One call site in a loop with a data-dependent trip count, so
// drivers that inline compile the interpreter once per pipeline, not once
// per slot (Adreno 6xx's Vulkan compiler crashed Chrome's GPU process on
// seven inlined copies).
fn gpu_values(paint: GpuPaint, inputs: u32, block: u32, colors: u32) -> array<vec4<f32>, 7> {
    var out: array<vec4<f32>, 7>;
    let zoom = expr_zoom(frame.zoom, frame.zoom_bits);
    let state = gpu_state(paint, block);
    var mask = paint.mask;
    while mask != 0u {
        let slot = lowest_bit(mask);
        mask &= mask - 1u;
        out[slot] = expr_value(gpu_program(paint, slot), inputs, state, zoom, (colors & (1u << slot)) != 0u);
    }
    return out;
}
