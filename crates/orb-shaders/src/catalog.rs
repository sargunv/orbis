// The shader catalog: composition of WGSL modules into pipeline sources.
// Shared with the build script, which translates the Compat compositions
// to GLSL ES 3.00.
pub(crate) const TYPES: &str = include_str!("wgsl/types.wgsl");
const PROJECT: &str = include_str!("wgsl/project.wgsl");
const DRAW: &str = concat!(
    include_str!("wgsl/bezier.wgsl"),
    include_str!("wgsl/draw.wgsl"),
    include_str!("wgsl/project.wgsl")
);
const FETCH: &str = include_str!("wgsl/fetch.wgsl");
const FETCH_COMPAT: &str = include_str!("wgsl/fetch_compat.wgsl");
const BUILTINS: &str = include_str!("wgsl/builtins.wgsl");
const BUILTINS_COMPAT: &str = include_str!("wgsl/builtins_compat.wgsl");
/// glTF's metallic-roughness shading, functions without bindings (plan
/// §9.4): for the model pipelines to compose.
pub const PBR_MODULE: &str = include_str!("wgsl/pbr.wgsl");

/// Cascaded shadows from the directional light, functions without
/// bindings: for casters and receivers to compose.
pub const SHADOW_MODULE: &str = include_str!("wgsl/shadow.wgsl");
const PATTERN: &str = include_str!("wgsl/pattern.wgsl");
const EXTRUSION: &str = include_str!("wgsl/extrusion.wgsl");
const EXTRUSION_COMPOSITE: &str = include_str!("wgsl/extrusion_composite.wgsl");

const FILL: &str = include_str!("wgsl/fill.wgsl");
const LINE: &str = include_str!("wgsl/line.wgsl");
const POINT: &str = include_str!("wgsl/point.wgsl");
const CIRCLE: &str = include_str!("wgsl/circle.wgsl");
const MODEL: &str = include_str!("wgsl/model.wgsl");
const SYMBOL: &str = include_str!("wgsl/symbol.wgsl");
const HEATMAP: &str = include_str!("wgsl/heatmap.wgsl");
const HEATMAP_COLOR: &str = include_str!("wgsl/heatmap_color.wgsl");
pub(crate) const CULL: &str = include_str!("wgsl/cull.wgsl");
const IMAGE: &str = include_str!("wgsl/image.wgsl");
const DEM: &str = include_str!("wgsl/dem.wgsl");
/// The atmosphere's sky and fog (decision 0025).
const ATMOSPHERE: &str = include_str!("wgsl/atmosphere.wgsl");
/// The style's lights on a surface (decision 0026).
const LIGHTING: &str = include_str!("wgsl/lighting.wgsl");
const RASTER: &str = include_str!("wgsl/raster.wgsl");
const HILLSHADE: &str = include_str!("wgsl/hillshade.wgsl");
const COLOR_RELIEF: &str = include_str!("wgsl/color_relief.wgsl");
const PREPARE: &str = include_str!("wgsl/prepare.wgsl");
const PREPARE_PAD: &str = include_str!("wgsl/prepare_pad.wgsl");
const PREPARE_SLOPE: &str = include_str!("wgsl/prepare_slope.wgsl");
const PREPARE_FS: &str = include_str!("wgsl/prepare_fs.wgsl");

/// A composed shader module and its entry points.
#[derive(Clone, Copy, Debug)]
pub struct Shader {
    pub label: &'static str,
    parts: &'static [&'static str],
    pub vertex: &'static str,
    pub fragment: &'static str,
    pub compute: &'static str,
}

impl Shader {
    pub fn source(&self) -> String {
        self.compose(false)
    }

    /// The Compat composition (plan §8.7): scene tables from data
    /// textures, no builtins GLSL ES 3.00 lacks. The `glsl` feature has
    /// these translated offline; see [`glsl`](crate::glsl).
    pub fn compat_source(&self) -> String {
        self.compose(true)
    }

    fn compose(&self, compat: bool) -> String {
        let mut source = self.parts.concat();
        if self.parts.contains(&DRAW) {
            source.push_str(if compat { FETCH_COMPAT } else { FETCH });
            if compat {
                source.push_str(crate::rows_compat());
            }
            source.push_str(if compat { BUILTINS_COMPAT } else { BUILTINS });
        }
        source
    }
}

pub const TERRAIN_SHADER: Shader = Shader {
    label: "terrain",
    parts: &[
        TYPES,
        DEM,
        ATMOSPHERE,
        LIGHTING,
        PROJECT,
        include_str!("wgsl/terrain.wgsl"),
    ],
    vertex: "vs_terrain",
    fragment: "fs_terrain",
    compute: "",
};

/// Clears drape slots (`vs_slot`, `fs_clear`) and builds their mip levels
/// (`downsample`).
pub const DRAPE_SHADER: Shader = Shader {
    label: "drape",
    parts: &[include_str!("wgsl/drape.wgsl")],
    vertex: "vs_slot",
    fragment: "fs_clear",
    compute: "downsample",
};

pub const TERRAIN_STACK_SHADER: Shader = Shader {
    label: "terrain stack",
    parts: &[include_str!("wgsl/terrain_stack.wgsl")],
    vertex: "vs_stack",
    fragment: "fs_stack",
    compute: "",
};

pub const SKY_SHADER: Shader = Shader {
    label: "sky",
    parts: &[TYPES, ATMOSPHERE, include_str!("wgsl/sky.wgsl")],
    vertex: "vs_sky",
    fragment: "fs_sky",
    compute: "",
};

/// A layer's composite at its layer opacity, blending normally.
pub const LAYER_COMPOSITE_SHADER: Shader = Shader {
    label: "layer composite",
    parts: &[include_str!("wgsl/layer_composite.wgsl")],
    vertex: "vs_layer_composite",
    fragment: "fs_layer_composite",
    compute: "",
};

/// A layer's composite in another blend mode, over the backdrop it reads.
pub const LAYER_BLEND_SHADER: Shader = Shader {
    label: "layer blend",
    parts: &[include_str!("wgsl/layer_composite.wgsl")],
    vertex: "vs_layer_composite",
    fragment: "fs_layer_blend",
    compute: "",
};

/// The globe atmosphere's scattering over the map (decision 0025).
pub const SCATTER_SHADER: Shader = Shader {
    label: "atmosphere scattering",
    fragment: "fs_scatter",
    ..SKY_SHADER
};

pub const FILL_SHADER: Shader = Shader {
    label: "fill",
    parts: &[TYPES, DRAW, PATTERN, FILL],
    vertex: "vs_fill",
    fragment: "fs_fill",
    compute: "",
};

pub const FILL_OUTLINE_SHADER: Shader = Shader {
    label: "fill outline",
    parts: &[TYPES, DRAW, PATTERN, FILL],
    vertex: "vs_fill_outline",
    fragment: "fs_fill_outline",
    compute: "",
};

pub const LINE_SHADER: Shader = Shader {
    label: "line",
    parts: &[TYPES, DRAW, PATTERN, LINE],
    vertex: "vs_line",
    fragment: "fs_line",
    compute: "",
};

pub const CIRCLE_SHADER: Shader = Shader {
    label: "circle",
    parts: &[TYPES, DRAW, POINT, CIRCLE],
    vertex: "vs_circle",
    fragment: "fs_circle",
    compute: "",
};
/// glTF models at point features, lit by the style's lights (plan §9.4).
/// Modern only: Compat warns for the gap.
pub const MODEL_SHADER: Shader = Shader {
    label: "model",
    parts: &[TYPES, DRAW, POINT, PBR_MODULE, MODEL],
    vertex: "vs_model",
    fragment: "fs_model",
    compute: "",
};
/// Symbol quads: glyphs and icons, with their halos (decision 0009).
pub const SYMBOL_SHADER: Shader = Shader {
    label: "symbol",
    parts: &[TYPES, DRAW, POINT, SYMBOL],
    vertex: "vs_symbol",
    fragment: "fs_symbol",
    compute: "",
};
pub const HEATMAP_SHADER: Shader = Shader {
    label: "heatmap density",
    parts: &[TYPES, DRAW, POINT, HEATMAP],
    vertex: "vs_heatmap",
    fragment: "fs_heatmap",
    compute: "",
};
pub const HEATMAP_COLOR_SHADER: Shader = Shader {
    label: "heatmap color",
    parts: &[TYPES, DRAW, POINT, HEATMAP, HEATMAP_COLOR],
    vertex: "vs_heatmap_color",
    fragment: "fs_heatmap_color",
    compute: "",
};

pub const RASTER_SHADER: Shader = Shader {
    label: "raster",
    parts: &[TYPES, DRAW, IMAGE, RASTER],
    vertex: "vs_raster",
    fragment: "fs_raster",
    compute: "",
};

pub const HILLSHADE_SHADER: Shader = Shader {
    label: "hillshade",
    parts: &[TYPES, DRAW, IMAGE, HILLSHADE],
    vertex: "vs_hillshade",
    fragment: "fs_hillshade",
    compute: "",
};

pub const COLOR_RELIEF_SHADER: Shader = Shader {
    label: "color relief",
    parts: &[TYPES, DRAW, IMAGE, DEM, COLOR_RELIEF],
    vertex: "vs_color_relief",
    fragment: "fs_color_relief",
    compute: "",
};

/// The DEM prepare pass's border fill, from raw tiles into padded ones.
/// A raster page's mip levels for its new layers (`downsample`).
pub const IMAGE_MIPS_SHADER: Shader = Shader {
    label: "image mips",
    parts: &[include_str!("wgsl/image_mips.wgsl")],
    vertex: "",
    fragment: "",
    compute: "downsample",
};

pub const PREPARE_PAD_SHADER: Shader = Shader {
    label: "prepare pad",
    parts: &[
        PREPARE,
        PREPARE_PAD,
        include_str!("wgsl/prepare_pad_cs.wgsl"),
    ],
    vertex: "",
    fragment: "",
    compute: "pad",
};

/// The DEM prepare pass's hillshade slopes, from padded tiles.
pub const PREPARE_SLOPE_SHADER: Shader = Shader {
    label: "prepare slope",
    parts: &[
        DEM,
        PREPARE,
        PREPARE_SLOPE,
        include_str!("wgsl/prepare_slope_cs.wgsl"),
    ],
    vertex: "",
    fragment: "",
    compute: "slope",
};

/// The border fill drawn into one padded layer (Compat).
pub const PREPARE_PAD_DRAW_SHADER: Shader = Shader {
    label: "prepare pad draw",
    parts: &[
        PREPARE,
        PREPARE_PAD,
        PREPARE_FS,
        include_str!("wgsl/prepare_pad_fs.wgsl"),
    ],
    vertex: "vs_prepare",
    fragment: "fs_pad",
    compute: "",
};

/// The slopes drawn into one slope layer (Compat).
pub const PREPARE_SLOPE_DRAW_SHADER: Shader = Shader {
    label: "prepare slope draw",
    parts: &[
        DEM,
        PREPARE,
        PREPARE_SLOPE,
        PREPARE_FS,
        include_str!("wgsl/prepare_slope_fs.wgsl"),
    ],
    vertex: "vs_prepare",
    fragment: "fs_slope",
    compute: "",
};

pub const CULL_SHADER: Shader = Shader {
    label: "cull",
    parts: &[TYPES, CULL],
    vertex: "",
    fragment: "",
    compute: "cull",
};

/// Lit roofs and walls with reversed-Z depth.
pub const EXTRUSION_SHADER: Shader = Shader {
    label: "extrusion",
    parts: &[TYPES, DRAW, PATTERN, LIGHTING, EXTRUSION],
    vertex: "vs_extrusion",
    fragment: "fs_extrusion",
    compute: "",
};
pub const EXTRUSION_COMPOSITE_SHADER: Shader = Shader {
    label: "extrusion composite",
    parts: &[
        TYPES,
        DRAW,
        PATTERN,
        LIGHTING,
        EXTRUSION,
        EXTRUSION_COMPOSITE,
    ],
    vertex: "vs_extrusion_composite",
    fragment: "fs_extrusion_composite",
    compute: "",
};

/// The shaders Compat draws with (plan §8.7): all but terrain's, the
/// drape's and compute.
pub const COMPAT_SHADERS: &[Shader] = &[
    SKY_SHADER,
    LAYER_COMPOSITE_SHADER,
    LAYER_BLEND_SHADER,
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
    EXTRUSION_SHADER,
    EXTRUSION_COMPOSITE_SHADER,
    PREPARE_PAD_DRAW_SHADER,
    PREPARE_SLOPE_DRAW_SHADER,
];
