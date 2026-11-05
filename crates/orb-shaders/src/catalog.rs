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
const PBR_MODULE: &str = include_str!("wgsl/pbr.wgsl");

/// The shadows a surface receives (decision 0046): the cascades, their
/// maps in the scene group and `shadow_at`. Compat composes
/// `shadow_none.wgsl` in its place: no shadows (plan §8.7).
const SHADOW_RECEIVE: &str = concat!(
    include_str!("wgsl/shadow.wgsl"),
    include_str!("wgsl/shadow_receive.wgsl")
);
const SHADOW_NONE: &str = include_str!("wgsl/shadow_none.wgsl");
const SHADOW_GROUND: &str = include_str!("wgsl/shadow_ground.wgsl");
const EXTRUSION_CASTER: &str = include_str!("wgsl/extrusion_caster.wgsl");
const PATTERN: &str = include_str!("wgsl/pattern.wgsl");
const EXTRUSION: &str = include_str!("wgsl/extrusion.wgsl");

const FILL: &str = include_str!("wgsl/fill.wgsl");
const LINE: &str = include_str!("wgsl/line.wgsl");
const POINT: &str = include_str!("wgsl/point.wgsl");
const CIRCLE: &str = include_str!("wgsl/circle.wgsl");
const MODEL: &str = include_str!("wgsl/model.wgsl");
const SYMBOL: &str = include_str!("wgsl/symbol.wgsl");
const SYMBOL_POINT: &str = include_str!("wgsl/symbol_point.wgsl");
const SYMBOL_LINE: &str = include_str!("wgsl/symbol_line.wgsl");
const HEATMAP: &str = include_str!("wgsl/heatmap.wgsl");
const COMPOSITE: &str = include_str!("wgsl/composite.wgsl");
pub(crate) const CULL: &str = include_str!("wgsl/cull.wgsl");
const IMAGE: &str = include_str!("wgsl/image.wgsl");
const DEM: &str = include_str!("wgsl/dem.wgsl");
/// The atmosphere's sky and fog (decision 0025).
const ATMOSPHERE: &str = include_str!("wgsl/atmosphere.wgsl");
/// The style's lights on a surface (decision 0026).
const LIGHTING: &str = include_str!("wgsl/lighting.wgsl");
const RASTER: &str = include_str!("wgsl/raster.wgsl");
const HILLSHADE: &str = include_str!("wgsl/hillshade.wgsl");
const TERRAIN: &str = include_str!("wgsl/terrain.wgsl");
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
        let mut source: String = self
            .parts
            .iter()
            .map(|&part| match part {
                SHADOW_RECEIVE if compat => SHADOW_NONE,
                part => part,
            })
            .collect();
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
    parts: &[TYPES, DEM, ATMOSPHERE, LIGHTING, PROJECT, TERRAIN],
    vertex: "vs_terrain",
    fragment: "fs_terrain",
    compute: "",
};

/// A terrain layer that shades a hillshade between its stacks.
pub const TERRAIN_SHADE_SHADER: Shader = Shader {
    label: "terrain shade",
    parts: &[
        TYPES,
        DEM,
        ATMOSPHERE,
        LIGHTING,
        PROJECT,
        TERRAIN,
        HILLSHADE,
        include_str!("wgsl/terrain_shade.wgsl"),
    ],
    vertex: "vs_terrain",
    fragment: "fs_terrain_shaded",
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

pub const SKY_SHADER: Shader = Shader {
    label: "sky",
    parts: &[TYPES, ATMOSPHERE, include_str!("wgsl/sky.wgsl")],
    vertex: "vs_sky",
    fragment: "fs_sky",
    compute: "",
};

/// Composites (`composite.wgsl`): over the map by premultiplied blending,
/// a source at an opacity or a density through a ramp.
pub const COMPOSITE_OVER_SHADER: Shader = Shader {
    label: "composite over",
    parts: &[COMPOSITE],
    vertex: "vs_composite",
    fragment: "fs_over",
    compute: "",
};

/// A layer blended into the map beneath in a W3C mode, written in place.
pub const COMPOSITE_REPLACE_SHADER: Shader = Shader {
    label: "composite replace",
    parts: &[COMPOSITE],
    vertex: "vs_composite",
    fragment: "fs_replace",
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
    parts: &[TYPES, DRAW, POINT, PBR_MODULE, SHADOW_RECEIVE, MODEL],
    vertex: "vs_model",
    fragment: "fs_model",
    compute: "",
};
/// Symbol quads: glyphs and icons, with their halos (decision 0009), of
/// point labels and of line labels (`symbol-placement: line` and
/// `line-center`): two variants, so point labels' pipeline doesn't compile
/// line placement (#69).
pub const SYMBOL_SHADERS: [Shader; 2] = [
    Shader {
        label: "symbol",
        parts: &[TYPES, DRAW, POINT, SYMBOL, SYMBOL_POINT],
        vertex: "vs_symbol",
        fragment: "fs_symbol",
        compute: "",
    },
    Shader {
        label: "symbol line",
        parts: &[TYPES, DRAW, POINT, SYMBOL, SYMBOL_LINE],
        vertex: "vs_symbol",
        fragment: "fs_symbol",
        compute: "",
    },
];
pub const HEATMAP_SHADER: Shader = Shader {
    label: "heatmap density",
    parts: &[TYPES, DRAW, POINT, HEATMAP],
    vertex: "vs_heatmap",
    fragment: "fs_heatmap",
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
    parts: &[
        TYPES,
        DRAW,
        IMAGE,
        HILLSHADE,
        include_str!("wgsl/hillshade_layer.wgsl"),
    ],
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

/// A raster page's mip levels for its new layers (`downsample`).
pub const IMAGE_MIPS_SHADER: Shader = Shader {
    label: "image mips",
    parts: &[include_str!("wgsl/image_mips.wgsl")],
    vertex: "",
    fragment: "",
    compute: "downsample",
};

/// The DEM prepare pass's border fill, from a raw tile into its padded
/// layer: a draw into the layer.
pub const PREPARE_PAD_SHADER: Shader = Shader {
    label: "prepare pad",
    parts: &[PREPARE, PREPARE_PAD],
    vertex: "vs_prepare",
    fragment: "fs_pad",
    compute: "",
};

/// The DEM prepare pass's hillshade slopes, from a padded tile into its
/// slope layer.
pub const PREPARE_SLOPE_SHADER: Shader = Shader {
    label: "prepare slope",
    parts: &[DEM, PREPARE, PREPARE_SLOPE],
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
    parts: &[TYPES, DRAW, PATTERN, LIGHTING, SHADOW_RECEIVE, EXTRUSION],
    vertex: "vs_extrusion",
    fragment: "fs_extrusion",
    compute: "",
};
/// The 3D depth resolved for the symbols it hides, from one sample and
/// from several.
pub const OCCLUSION_SHADERS: [Shader; 2] = [
    Shader {
        label: "occlusion",
        parts: &[
            include_str!("wgsl/occlusion.wgsl"),
            include_str!("wgsl/occlusion_single.wgsl"),
        ],
        vertex: "vs_occlusion",
        fragment: "fs_occlusion",
        compute: "",
    },
    Shader {
        label: "occlusion multisampled",
        parts: &[
            include_str!("wgsl/occlusion.wgsl"),
            include_str!("wgsl/occlusion_multi.wgsl"),
        ],
        vertex: "vs_occlusion",
        fragment: "fs_occlusion",
        compute: "",
    },
];
/// Extrusions' depth from the light, into a shadow cascade. Modern only.
pub const EXTRUSION_CASTER_SHADER: Shader = Shader {
    label: "extrusion caster",
    parts: &[
        TYPES,
        DRAW,
        PATTERN,
        LIGHTING,
        SHADOW_RECEIVE,
        EXTRUSION,
        EXTRUSION_CASTER,
    ],
    vertex: "vs_caster",
    fragment: "fs_caster",
    compute: "",
};
/// Models' depth from the light, into a shadow cascade. Modern only.
pub const MODEL_CASTER_SHADER: Shader = Shader {
    label: "model caster",
    parts: &[
        TYPES,
        DRAW,
        POINT,
        PBR_MODULE,
        SHADOW_RECEIVE,
        MODEL,
        include_str!("wgsl/model_caster.wgsl"),
    ],
    vertex: "vs_model_caster",
    fragment: "fs_model_caster",
    compute: "",
};
/// Shadows on the ground, multiplied over the main pass. Modern only.
pub const SHADOW_GROUND_SHADER: Shader = Shader {
    label: "shadow ground",
    parts: &[TYPES, DRAW, SHADOW_RECEIVE, SHADOW_GROUND],
    vertex: "vs_shadow_ground",
    fragment: "fs_shadow_ground",
    compute: "",
};

/// The shaders Compat draws with (plan §8.7): all but terrain's, the
/// drape's and compute.
pub const COMPAT_SHADERS: &[Shader] = &[
    SKY_SHADER,
    COMPOSITE_OVER_SHADER,
    COMPOSITE_REPLACE_SHADER,
    FILL_SHADER,
    FILL_OUTLINE_SHADER,
    LINE_SHADER,
    CIRCLE_SHADER,
    SYMBOL_SHADERS[0],
    SYMBOL_SHADERS[1],
    HEATMAP_SHADER,
    RASTER_SHADER,
    HILLSHADE_SHADER,
    COLOR_RELIEF_SHADER,
    EXTRUSION_SHADER,
    PREPARE_PAD_SHADER,
    PREPARE_SLOPE_SHADER,
];

/// Every shader: what the build script reflects layouts from, and the
/// tests validate.
pub const SHADERS: &[Shader] = &[
    SKY_SHADER,
    SCATTER_SHADER,
    COMPOSITE_OVER_SHADER,
    COMPOSITE_REPLACE_SHADER,
    TERRAIN_SHADER,
    TERRAIN_SHADE_SHADER,
    DRAPE_SHADER,
    EXTRUSION_SHADER,
    EXTRUSION_CASTER_SHADER,
    SHADOW_GROUND_SHADER,
    OCCLUSION_SHADERS[0],
    OCCLUSION_SHADERS[1],
    FILL_SHADER,
    FILL_OUTLINE_SHADER,
    LINE_SHADER,
    CIRCLE_SHADER,
    MODEL_SHADER,
    MODEL_CASTER_SHADER,
    SYMBOL_SHADERS[0],
    SYMBOL_SHADERS[1],
    HEATMAP_SHADER,
    RASTER_SHADER,
    HILLSHADE_SHADER,
    COLOR_RELIEF_SHADER,
    PREPARE_PAD_SHADER,
    PREPARE_SLOPE_SHADER,
    IMAGE_MIPS_SHADER,
    CULL_SHADER,
];
