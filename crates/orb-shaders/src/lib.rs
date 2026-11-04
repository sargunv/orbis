//! WGSL modules and their composition (plan §8.4).
//!
//! Each pipeline's source is the shared scene types, the bindings of its
//! kind (draw or cull) and its own module, concatenated. Shaders that read
//! paint come in two classes: the table class with stand-ins for the GPU
//! expression interpreter, and the bytecode class with the interpreter
//! (`expr.wgsl`), compiled only into pipelines of nodes that need it.

#![forbid(unsafe_code)]

const TYPES: &str = include_str!("wgsl/types.wgsl");
const DRAW: &str = concat!(
    include_str!("wgsl/bezier.wgsl"),
    include_str!("wgsl/draw.wgsl")
);
const GPU_PAINT: &str = include_str!("wgsl/gpu_paint.wgsl");
const EXPR: &str = include_str!("wgsl/expr.wgsl");
const EXPR_NONE: &str = include_str!("wgsl/expr_none.wgsl");

/// The GPU expression interpreter alone, for harnesses that bind their own
/// `props` buffer: `expr_eval`, `expr_zoom`, `expr_to_number` and
/// `expr_to_color` over `props: array<u32>`.
pub const EXPRESSION_MODULE: &str = concat!(
    include_str!("wgsl/bezier.wgsl"),
    include_str!("wgsl/expr.wgsl")
);
const PATTERN: &str = include_str!("wgsl/pattern.wgsl");
const EXTRUSION: &str = include_str!("wgsl/extrusion.wgsl");
const EXTRUSION_COMPOSITE: &str = include_str!("wgsl/extrusion_composite.wgsl");

const FILL: &str = include_str!("wgsl/fill.wgsl");
const LINE: &str = include_str!("wgsl/line.wgsl");
const POINT: &str = include_str!("wgsl/point.wgsl");
const CIRCLE: &str = include_str!("wgsl/circle.wgsl");
const SYMBOL: &str = include_str!("wgsl/symbol.wgsl");
const HEATMAP: &str = include_str!("wgsl/heatmap.wgsl");
const HEATMAP_COLOR: &str = include_str!("wgsl/heatmap_color.wgsl");
const CULL: &str = include_str!("wgsl/cull.wgsl");
const IMAGE: &str = include_str!("wgsl/image.wgsl");
const DEM: &str = include_str!("wgsl/dem.wgsl");
const RASTER: &str = include_str!("wgsl/raster.wgsl");
const HILLSHADE: &str = include_str!("wgsl/hillshade.wgsl");
const COLOR_RELIEF: &str = include_str!("wgsl/color_relief.wgsl");
const PREPARE: &str = include_str!("wgsl/prepare.wgsl");
const PREPARE_PAD: &str = include_str!("wgsl/prepare_pad.wgsl");
const PREPARE_SLOPE: &str = include_str!("wgsl/prepare_slope.wgsl");

/// A composed shader module and its entry points.
#[derive(Clone, Copy, Debug)]
pub struct Shader {
    pub label: &'static str,
    parts: &'static [&'static str],
    pub vertex: &'static str,
    pub fragment: &'static str,
    pub compute: &'static str,
    /// Reads paint through `expr_to_number` and `expr_to_color`: the
    /// shader has a bytecode class.
    pub paint: bool,
}

impl Shader {
    /// The table class: no interpreter.
    pub fn source(&self) -> String {
        self.compose(EXPR_NONE)
    }

    /// The bytecode class, with the GPU expression interpreter.
    pub fn bytecode_source(&self) -> String {
        assert!(self.paint, "{} reads no paint", self.label);
        self.compose(EXPR)
    }

    fn compose(&self, paint: &str) -> String {
        let mut source = self.parts.concat();
        if self.paint {
            source.push_str(GPU_PAINT);
            source.push_str(paint);
        }
        source
    }
}

pub const TERRAIN_SHADER: Shader = Shader {
    label: "terrain",
    parts: &[TYPES, DEM, include_str!("wgsl/terrain.wgsl")],
    vertex: "vs_terrain",
    fragment: "fs_terrain",
    compute: "",
    paint: false,
};

/// Clears drape slots (`fs_clear`) or builds a mip level of them
/// (`fs_downsample`).
pub const DRAPE_SHADER: Shader = Shader {
    label: "drape",
    parts: &[include_str!("wgsl/drape.wgsl")],
    vertex: "vs_slot",
    fragment: "fs_downsample",
    compute: "",
    paint: false,
};

pub const TERRAIN_STACK_SHADER: Shader = Shader {
    label: "terrain stack",
    parts: &[include_str!("wgsl/terrain_stack.wgsl")],
    vertex: "vs_stack",
    fragment: "fs_stack",
    compute: "",
    paint: false,
};

pub const SKY_SHADER: Shader = Shader {
    label: "sky",
    parts: &[TYPES, include_str!("wgsl/sky.wgsl")],
    vertex: "vs_sky",
    fragment: "fs_sky",
    compute: "",
    paint: false,
};

pub const FILL_SHADER: Shader = Shader {
    label: "fill",
    parts: &[TYPES, DRAW, PATTERN, FILL],
    vertex: "vs_fill",
    fragment: "fs_fill",
    compute: "",
    paint: true,
};

pub const FILL_OUTLINE_SHADER: Shader = Shader {
    label: "fill outline",
    parts: &[TYPES, DRAW, PATTERN, FILL],
    vertex: "vs_fill_outline",
    fragment: "fs_fill_outline",
    compute: "",
    paint: true,
};

pub const LINE_SHADER: Shader = Shader {
    label: "line",
    parts: &[TYPES, DRAW, PATTERN, LINE],
    vertex: "vs_line",
    fragment: "fs_line",
    compute: "",
    paint: true,
};

pub const CIRCLE_SHADER: Shader = Shader {
    label: "circle",
    parts: &[TYPES, DRAW, POINT, CIRCLE],
    vertex: "vs_circle",
    fragment: "fs_circle",
    compute: "",
    paint: true,
};
/// Symbol quads: glyphs and icons, with their halos (decision 0009).
pub const SYMBOL_SHADER: Shader = Shader {
    label: "symbol",
    parts: &[TYPES, DRAW, POINT, SYMBOL],
    vertex: "vs_symbol",
    fragment: "fs_symbol",
    compute: "",
    paint: false,
};
pub const HEATMAP_SHADER: Shader = Shader {
    label: "heatmap density",
    parts: &[TYPES, DRAW, POINT, HEATMAP],
    vertex: "vs_heatmap",
    fragment: "fs_heatmap",
    compute: "",
    paint: false,
};
pub const HEATMAP_COLOR_SHADER: Shader = Shader {
    label: "heatmap color",
    parts: &[TYPES, DRAW, POINT, HEATMAP, HEATMAP_COLOR],
    vertex: "vs_heatmap_color",
    fragment: "fs_heatmap_color",
    compute: "",
    paint: false,
};

pub const RASTER_SHADER: Shader = Shader {
    label: "raster",
    parts: &[TYPES, DRAW, IMAGE, RASTER],
    vertex: "vs_raster",
    fragment: "fs_raster",
    compute: "",
    paint: false,
};

pub const HILLSHADE_SHADER: Shader = Shader {
    label: "hillshade",
    parts: &[TYPES, DRAW, IMAGE, HILLSHADE],
    vertex: "vs_hillshade",
    fragment: "fs_hillshade",
    compute: "",
    paint: false,
};

pub const COLOR_RELIEF_SHADER: Shader = Shader {
    label: "color relief",
    parts: &[TYPES, DRAW, IMAGE, DEM, COLOR_RELIEF],
    vertex: "vs_color_relief",
    fragment: "fs_color_relief",
    compute: "",
    paint: false,
};

/// The DEM prepare pass's border fill, from raw tiles into padded ones.
pub const PREPARE_PAD_SHADER: Shader = Shader {
    label: "prepare pad",
    parts: &[PREPARE, PREPARE_PAD],
    vertex: "",
    fragment: "",
    compute: "pad",
    paint: false,
};

/// The DEM prepare pass's hillshade slopes, from padded tiles.
pub const PREPARE_SLOPE_SHADER: Shader = Shader {
    label: "prepare slope",
    parts: &[DEM, PREPARE, PREPARE_SLOPE],
    vertex: "",
    fragment: "",
    compute: "slope",
    paint: false,
};

pub const CULL_SHADER: Shader = Shader {
    label: "cull",
    parts: &[TYPES, CULL],
    vertex: "",
    fragment: "",
    compute: "cull",
    paint: false,
};

/// Lit roofs and walls with reversed-Z depth.
pub const EXTRUSION_SHADER: Shader = Shader {
    label: "extrusion",
    parts: &[TYPES, DRAW, PATTERN, EXTRUSION],
    vertex: "vs_extrusion",
    fragment: "fs_extrusion",
    compute: "",
    paint: false,
};
pub const EXTRUSION_COMPOSITE_SHADER: Shader = Shader {
    label: "extrusion composite",
    parts: &[TYPES, DRAW, PATTERN, EXTRUSION, EXTRUSION_COMPOSITE],
    vertex: "vs_extrusion_composite",
    fragment: "fs_extrusion_composite",
    compute: "",
    paint: false,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(shader: &Shader) -> naga::Module {
        if shader.paint {
            parse(shader.label, &shader.bytecode_source());
        }
        parse(shader.label, &shader.source())
    }

    fn parse(label: &str, source: &str) -> naga::Module {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(source)));
        module
    }

    #[test]
    fn shaders_validate() {
        for shader in [
            SKY_SHADER,
            TERRAIN_SHADER,
            DRAPE_SHADER,
            TERRAIN_STACK_SHADER,
            EXTRUSION_SHADER,
            EXTRUSION_COMPOSITE_SHADER,
            FILL_SHADER,
            FILL_OUTLINE_SHADER,
            LINE_SHADER,
            CIRCLE_SHADER,
            SYMBOL_SHADER,
            HEATMAP_SHADER,
            HEATMAP_COLOR_SHADER,
            RASTER_SHADER,
            HILLSHADE_SHADER,
            COLOR_RELIEF_SHADER,
            PREPARE_PAD_SHADER,
            PREPARE_SLOPE_SHADER,
            CULL_SHADER,
        ] {
            let module = validate(&shader);
            for entry in [shader.vertex, shader.fragment, shader.compute] {
                if !entry.is_empty() {
                    assert!(
                        module.entry_points.iter().any(|e| e.name == entry),
                        "{}: no entry point {entry}",
                        shader.label
                    );
                }
            }
        }
    }
}
