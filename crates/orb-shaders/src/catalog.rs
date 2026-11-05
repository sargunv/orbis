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
/// The style's lights as `pbr.wgsl` takes them.
const SCENE_LIGHTS: &str = include_str!("wgsl/scene_lights.wgsl");
/// A map layer's output: its colour, or lit by a scene's lights (#91,
/// decision 0048), as [`Shader::lit`] composes.
const MAP_OUT: &str = include_str!("wgsl/map_out.wgsl");
const MAP_OUT_LIT: &str = include_str!("wgsl/map_out_lit.wgsl");

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
const EXTRUSION_FRAGMENT: &str = include_str!("wgsl/extrusion_fragment.wgsl");

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
const CONTOUR: &str = include_str!("wgsl/contour.wgsl");
const PREPARE: &str = include_str!("wgsl/prepare.wgsl");
const PREPARE_PAD: &str = include_str!("wgsl/prepare_pad.wgsl");
/// First of a vertex-only shader's parts where clip distances are (wgpu's
/// `CLIP_DISTANCES`, WebGPU's `clip-distances`): it enables them, and the
/// parts' lines marked `//planes ` become code. A fragment stage can't take
/// clip distances, so such a shader leaves its fragment stages out.
const CLIP_PLANES: &str = "enable clip_distances;\n";
const PREPARE_SLOPE: &str = include_str!("wgsl/prepare_slope.wgsl");

/// A composed shader module and its entry points.
#[derive(Clone, Copy, Debug)]
pub struct Shader {
    pub label: &'static str,
    parts: &'static [&'static str],
    pub vertex: &'static str,
    pub fragment: &'static str,
    pub compute: &'static str,
    /// Its map layers are lit by a scene's lights.
    pub lit: bool,
}

impl Shader {
    /// The same shader with its map layers lit by a scene's lights.
    pub const fn lit(self) -> Self {
        Self { lit: true, ..self }
    }

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
        if self.parts.contains(&CLIP_PLANES) {
            source = source.replace("//planes ", "");
        }
        if self.parts.contains(&DRAW) {
            source.push_str(if compat { FETCH_COMPAT } else { FETCH });
            if compat {
                source.push_str(crate::rows_compat());
            }
            source.push_str(if compat { BUILTINS_COMPAT } else { BUILTINS });
        }
        if self.lit {
            for part in [PBR_MODULE, SCENE_LIGHTS, MAP_OUT_LIT] {
                if !self.parts.contains(&part) {
                    source.push_str(part);
                }
            }
        } else if source.contains("MapOut") || source.contains("surface_out(") {
            source.push_str(MAP_OUT);
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
    lit: false,
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
    lit: false,
};

/// Clears the drape slots a frame redraws.
pub const DRAPE_SHADER: Shader = Shader {
    label: "drape",
    parts: &[include_str!("wgsl/drape.wgsl")],
    vertex: "vs_slot",
    fragment: "fs_clear",
    compute: "",
    lit: false,
};

pub const SKY_SHADER: Shader = Shader {
    label: "sky",
    parts: &[TYPES, ATMOSPHERE, include_str!("wgsl/sky.wgsl")],
    vertex: "vs_sky",
    fragment: "fs_sky",
    compute: "",
    lit: false,
};

/// Composites (`composite.wgsl`): over the map by premultiplied blending,
/// a source at an opacity or a density through a ramp.
pub const COMPOSITE_OVER_SHADER: Shader = Shader {
    label: "composite over",
    parts: &[COMPOSITE],
    vertex: "vs_composite",
    fragment: "fs_over",
    compute: "",
    lit: false,
};

/// A layer blended into the map beneath in a W3C mode, written in place.
pub const COMPOSITE_REPLACE_SHADER: Shader = Shader {
    label: "composite replace",
    parts: &[COMPOSITE],
    vertex: "vs_composite",
    fragment: "fs_replace",
    compute: "",
    lit: false,
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
    lit: false,
};

pub const FILL_OUTLINE_SHADER: Shader = Shader {
    label: "fill outline",
    parts: &[TYPES, DRAW, PATTERN, FILL],
    vertex: "vs_fill_outline",
    fragment: "fs_fill_outline",
    compute: "",
    lit: false,
};

pub const LINE_SHADER: Shader = Shader {
    label: "line",
    parts: &[TYPES, DRAW, PATTERN, LINE],
    vertex: "vs_line",
    fragment: "fs_line",
    compute: "",
    lit: false,
};

pub const CIRCLE_SHADER: Shader = Shader {
    label: "circle",
    parts: &[TYPES, DRAW, POINT, CIRCLE],
    vertex: "vs_circle",
    fragment: "fs_circle",
    compute: "",
    lit: false,
};
/// glTF models at point features, lit by the style's lights (plan §9.4).
/// Modern only: Compat warns for the gap.
pub const MODEL_SHADER: Shader = Shader {
    label: "model",
    parts: &[
        TYPES,
        DRAW,
        POINT,
        PBR_MODULE,
        SHADOW_RECEIVE,
        SCENE_LIGHTS,
        MODEL,
    ],
    vertex: "vs_model",
    fragment: "fs_model",
    compute: "",
    lit: false,
};
/// Alpha-masked and blended model materials, which discard: apart from
/// opaque ones, whose pipelines never do.
pub const MODEL_MASKED_SHADER: Shader = Shader {
    label: "model masked",
    fragment: "fs_model_masked",
    ..MODEL_SHADER
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
        lit: false,
    },
    Shader {
        label: "symbol line",
        parts: &[TYPES, DRAW, POINT, SYMBOL, SYMBOL_LINE],
        vertex: "vs_symbol",
        fragment: "fs_symbol",
        compute: "",
        lit: false,
    },
];
pub const HEATMAP_SHADER: Shader = Shader {
    label: "heatmap density",
    parts: &[TYPES, DRAW, POINT, HEATMAP],
    vertex: "vs_heatmap",
    fragment: "fs_heatmap",
    compute: "",
    lit: false,
};

pub const RASTER_SHADER: Shader = Shader {
    label: "raster",
    parts: &[TYPES, DRAW, IMAGE, RASTER],
    vertex: "vs_raster",
    fragment: "fs_raster",
    compute: "",
    lit: false,
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
    lit: false,
};

pub const COLOR_RELIEF_SHADER: Shader = Shader {
    label: "color relief",
    parts: &[TYPES, DRAW, IMAGE, DEM, COLOR_RELIEF],
    vertex: "vs_color_relief",
    fragment: "fs_color_relief",
    compute: "",
    lit: false,
};
/// Contour lines of a raster-dem source: the scene document's `contour`.
pub const CONTOUR_SHADER: Shader = Shader {
    label: "contour",
    parts: &[TYPES, DRAW, IMAGE, DEM, CONTOUR],
    vertex: "vs_contour",
    fragment: "fs_contour",
    compute: "",
    lit: false,
};

/// Mip levels over regions of array layers: drape slots, raster pages.
pub const DOWNSAMPLE_SHADER: Shader = Shader {
    label: "downsample",
    parts: &[include_str!("wgsl/downsample.wgsl")],
    vertex: "",
    fragment: "",
    compute: "downsample",
    lit: false,
};

/// The DEM prepare pass's border fill, from a raw tile into its padded
/// layer: a draw into the layer.
pub const PREPARE_PAD_SHADER: Shader = Shader {
    label: "prepare pad",
    parts: &[PREPARE, PREPARE_PAD],
    vertex: "vs_prepare",
    fragment: "fs_pad",
    compute: "",
    lit: false,
};

/// The DEM prepare pass's hillshade slopes, from a padded tile into its
/// slope layer.
pub const PREPARE_SLOPE_SHADER: Shader = Shader {
    label: "prepare slope",
    parts: &[DEM, PREPARE, PREPARE_SLOPE],
    vertex: "vs_prepare",
    fragment: "fs_slope",
    compute: "",
    lit: false,
};

pub const CULL_SHADER: Shader = Shader {
    label: "cull",
    parts: &[TYPES, CULL],
    vertex: "",
    fragment: "",
    compute: "cull",
    lit: false,
};

/// Lit roofs and walls with reversed-Z depth.
pub const EXTRUSION_SHADER: Shader = Shader {
    label: "extrusion",
    parts: &[
        TYPES,
        DRAW,
        PATTERN,
        LIGHTING,
        SHADOW_RECEIVE,
        EXTRUSION,
        EXTRUSION_FRAGMENT,
    ],
    vertex: "vs_extrusion",
    fragment: "fs_extrusion",
    compute: "",
    lit: false,
};
/// Extrusions cut at the source's buffer, clipped per fragment, apart from
/// whole ones, whose pipeline never discards: every one, or where clip
/// distances are ([`EXTRUSION_PLANES_SHADER`]) those of placements with
/// several clip rectangles (`vs_extrusion_cut_split`).
pub const EXTRUSION_CUT_SHADER: Shader = Shader {
    label: "extrusion cut",
    vertex: "vs_extrusion_cut",
    fragment: "fs_extrusion_cut",
    ..EXTRUSION_SHADER
};
/// Cut extrusions of a placement with one clip rectangle, clipped by its
/// four planes: no discard. Where clip distances are. Vertex only: its
/// pipeline draws `EXTRUSION_SHADER`'s `fs_extrusion`, as a fragment stage
/// can't take the clip distances its vertex output carries.
pub const EXTRUSION_PLANES_SHADER: Shader = Shader {
    label: "extrusion planes",
    parts: &[
        CLIP_PLANES,
        TYPES,
        DRAW,
        PATTERN,
        LIGHTING,
        SHADOW_RECEIVE,
        EXTRUSION,
    ],
    vertex: "vs_extrusion_cut_planes",
    fragment: "",
    ..EXTRUSION_SHADER
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
        lit: false,
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
        lit: false,
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
    lit: false,
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
        SCENE_LIGHTS,
        MODEL,
        include_str!("wgsl/model_caster.wgsl"),
    ],
    vertex: "vs_model_caster",
    fragment: "fs_model_caster",
    compute: "",
    lit: false,
};
/// Alpha-masked models into a shadow cascade, their texels under the
/// cutoff discarded.
pub const MODEL_CASTER_MASKED_SHADER: Shader = Shader {
    label: "model caster masked",
    fragment: "fs_model_caster_masked",
    ..MODEL_CASTER_SHADER
};
/// Shadows on the ground, multiplied over the main pass. Modern only.
pub const SHADOW_GROUND_SHADER: Shader = Shader {
    label: "shadow ground",
    parts: &[TYPES, DRAW, SHADOW_RECEIVE, SHADOW_GROUND],
    vertex: "vs_shadow_ground",
    fragment: "fs_shadow_ground",
    compute: "",
    lit: false,
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
    CONTOUR_SHADER,
    EXTRUSION_SHADER,
    EXTRUSION_CUT_SHADER,
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
    EXTRUSION_CUT_SHADER,
    EXTRUSION_PLANES_SHADER,
    EXTRUSION_CASTER_SHADER,
    SHADOW_GROUND_SHADER,
    OCCLUSION_SHADERS[0],
    OCCLUSION_SHADERS[1],
    FILL_SHADER,
    FILL_OUTLINE_SHADER,
    LINE_SHADER,
    CIRCLE_SHADER,
    MODEL_SHADER,
    MODEL_MASKED_SHADER,
    MODEL_CASTER_SHADER,
    MODEL_CASTER_MASKED_SHADER,
    FILL_SHADER.lit(),
    FILL_OUTLINE_SHADER.lit(),
    LINE_SHADER.lit(),
    CIRCLE_SHADER.lit(),
    RASTER_SHADER.lit(),
    HILLSHADE_SHADER.lit(),
    COLOR_RELIEF_SHADER.lit(),
    CONTOUR_SHADER.lit(),
    EXTRUSION_SHADER.lit(),
    SYMBOL_SHADERS[0],
    SYMBOL_SHADERS[1],
    HEATMAP_SHADER,
    RASTER_SHADER,
    HILLSHADE_SHADER,
    COLOR_RELIEF_SHADER,
    CONTOUR_SHADER,
    PREPARE_PAD_SHADER,
    PREPARE_SLOPE_SHADER,
    DOWNSAMPLE_SHADER,
    CULL_SHADER,
];
