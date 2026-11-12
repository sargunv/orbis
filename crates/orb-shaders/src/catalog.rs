// The shader catalog: composition of WGSL modules into pipeline sources.
// Shared with the build script, which reflects the shaders' uniforms.
pub(crate) const TYPES: &str = wgsl!("types");
const PROJECT: &str = wgsl!("project");
/// Draw pipelines' helpers, `builtins` and `fetch`: their reads of the
/// scene's tables.
const DRAW: &str = concat!(
    wgsl!("builtins"),
    wgsl!("bezier"),
    wgsl!("dem"),
    wgsl!("draw"),
    wgsl!("project"),
    wgsl!("fetch")
);
const PATTERN: &str = wgsl!("pattern");
const EXTRUSION: &str = wgsl!("extrusion");
const EXTRUSION_FRAGMENT: &str = wgsl!("extrusion_fragment");

const FILL: &str = wgsl!("fill");
const LINE: &str = wgsl!("line");
const POINT: &str = wgsl!("point");
const CIRCLE: &str = wgsl!("circle");
const SYMBOL: &str = wgsl!("symbol");
const SYMBOL_DRAW: &str = wgsl!("symbol_draw");
const SYMBOL_HALVES: &str = wgsl!("symbol_halves");
const SYMBOL_NODES: &str = wgsl!("symbol_nodes");
const SYMBOL_PLACE: &str = wgsl!("symbol_place");
/// The symbol draw reading the placement dispatch's target: no WebGL 2
/// stage composes it (`gl_source`).
pub(crate) const SYMBOL_PLACED: &str = wgsl!("symbol_placed");
const SYMBOL_PLACE_INLINE: &str = wgsl!("symbol_place_inline");
const SYMBOL_PLACE_COMPUTE: &str = wgsl!("symbol_place_compute");
const SYMBOL_POINT: &str = wgsl!("symbol_point");
const SYMBOL_LINE: &str = wgsl!("symbol_line");
const HEATMAP: &str = wgsl!("heatmap");
const BLEND: &str = wgsl!("blend");
const COMPOSITE: &str = concat!(wgsl!("blend"), wgsl!("composite"));
pub(crate) const CULL: &str = wgsl!("cull");
const IMAGE: &str = wgsl!("image");
const DEM: &str = wgsl!("dem");
/// The atmosphere's sky and fog.
const ATMOSPHERE: &str = wgsl!("atmosphere");
/// v8's `light` on extrusions.
const LIGHTING: &str = wgsl!("lighting");
const RASTER: &str = wgsl!("raster");
const HILLSHADE: &str = wgsl!("hillshade");
/// The terrain, and `fetch_terrain`: its reads of its tables.
const TERRAIN: &str = concat!(wgsl!("fetch_terrain"), wgsl!("terrain"));
const COLOR_RELIEF: &str = wgsl!("color_relief");
const PREPARE: &str = wgsl!("prepare");
const PREPARE_PAD: &str = wgsl!("prepare_pad");
/// First of a vertex-only shader's parts where clip distances are (wgpu's
/// `CLIP_DISTANCES`, WebGPU's `clip-distances`): it enables them, and the
/// parts' lines marked `//planes ` become code. A fragment stage can't take
/// clip distances, so such a shader leaves its fragment stages out.
pub(crate) const CLIP_PLANES: &str = "enable clip_distances;\n";
const PREPARE_SLOPE: &str = wgsl!("prepare_slope");
/// The downsampler's texels (`mip_texel`, `extend_texel`), and its reads
/// of its regions.
const DOWNSAMPLE: &str = wgsl!("downsample");
const FETCH_REGIONS: &str = wgsl!("fetch_regions");

/// A composed shader module and its entry points.
#[derive(Clone, Copy, Debug)]
pub struct Shader {
    pub label: &'static str,
    /// Its WGSL modules (`sources.rs` puts them together), which WebGL 2's
    /// web build leaves empty (`glsl_only`).
    #[allow(dead_code, reason = "WebGL 2's web build reads no WGSL")]
    pub(crate) parts: &'static [&'static str],
    pub vertex: &'static str,
    pub fragment: &'static str,
    pub compute: &'static str,
}

/// naga's line ending a GLSL vertex stage that clips depth from 0 to 1,
/// as the build writes WebGL 2's stages, and the line it writes where depth
/// is remapped into GL's -1 to 1 instead (a context without clip control).
#[cfg_attr(not(feature = "glsl"), allow(dead_code))]
pub(crate) const GLSL_ZERO_TO_ONE: &str = "gl_Position.y = -gl_Position.y;";
#[cfg_attr(not(feature = "glsl"), allow(dead_code))]
pub(crate) const GLSL_MINUS_ONE_TO_ONE: &str =
    "gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);";

/// The name of WebGL 2's vertex input at `@location(0)`
/// (`sources.rs`'s `with_attribute_zero`).
pub const ATTRIBUTE_ZERO: &str = "orb_attribute_zero";

pub const TERRAIN_SHADER: Shader = Shader {
    label: "terrain",
    parts: &[TYPES, DEM, ATMOSPHERE, PROJECT, TERRAIN],
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
        PROJECT,
        TERRAIN,
        HILLSHADE,
        BLEND,
        wgsl!("terrain_shade"),
    ],
    vertex: "vs_terrain",
    fragment: "fs_terrain_shaded",
    compute: "",
};

/// Clears the drape slots a frame redraws.
pub const DRAPE_SHADER: Shader = Shader {
    label: "drape",
    parts: &[wgsl!("fetch_drape"), wgsl!("drape")],
    vertex: "vs_slot",
    fragment: "fs_clear",
    compute: "",
};

pub const SKY_SHADER: Shader = Shader {
    label: "sky",
    parts: &[TYPES, ATMOSPHERE, wgsl!("sky")],
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

/// The globe atmosphere's scattering over the map.
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

/// Lines without dashes, a pattern or a gradient.
pub const LINE_SHADER: Shader = Shader {
    label: "line",
    parts: &[TYPES, DRAW, PATTERN, LINE],
    vertex: "vs_line",
    fragment: "fs_line",
    compute: "",
};

/// Dashed, patterned and gradient lines, whose fragments read images.
pub const LINE_STYLED_SHADER: Shader = Shader {
    label: "line styled",
    vertex: "vs_line_styled",
    fragment: "fs_line_styled",
    ..LINE_SHADER
};

pub const CIRCLE_SHADER: Shader = Shader {
    label: "circle",
    parts: &[TYPES, DRAW, POINT, CIRCLE],
    vertex: "vs_circle",
    fragment: "fs_circle",
    compute: "",
};
/// Symbol quads: glyphs and icons, with their halos, each corner where
/// the placement dispatch put its quad ([`SYMBOL_PLACE_SHADERS`]).
pub const SYMBOL_SHADER: Shader = Shader {
    label: "symbol",
    parts: &[
        TYPES,
        DRAW,
        SYMBOL_HALVES,
        SYMBOL,
        SYMBOL_PLACED,
        SYMBOL_DRAW,
    ],
    vertex: "vs_symbol",
    fragment: "fs_symbol",
    compute: "",
};
/// [`SYMBOL_LINE`] placing line labels alone: WebGL 2's line variant,
/// without variable anchors.
const LINE_LABELS_ALONE: &str = "const POINT_LABELS: bool = false;\n";
/// [`SYMBOL_LINE`] placing point labels too: the placement dispatch's.
const POINT_AND_LINE_LABELS: &str = "const POINT_LABELS: bool = true;\n";
/// The symbol placements: each quad of point labels and of line labels
/// (`symbol-placement: line` and `line-center`) placed once a frame into
/// the placement target, by one dispatch for every symbol layer: point
/// labels' alone where no layer places along lines, so their pipeline
/// doesn't compile line placement, or both.
pub const SYMBOL_PLACE_SHADERS: [Shader; 2] = [
    Shader {
        label: "symbol placement",
        parts: &[
            TYPES,
            DRAW,
            POINT,
            SYMBOL_NODES,
            SYMBOL,
            SYMBOL_PLACE,
            SYMBOL_PLACE_COMPUTE,
            SYMBOL_POINT,
        ],
        vertex: "",
        fragment: "",
        compute: "cs_place",
    },
    Shader {
        label: "symbol placement with lines",
        parts: &[
            TYPES,
            DRAW,
            POINT,
            SYMBOL_NODES,
            SYMBOL,
            SYMBOL_PLACE,
            SYMBOL_PLACE_COMPUTE,
            SYMBOL_LINE,
            POINT_AND_LINE_LABELS,
        ],
        vertex: "",
        fragment: "",
        compute: "cs_place",
    },
];
/// [`SYMBOL_SHADER`] for a device without compute (WebGL 2), each corner
/// placing its quad itself: of point labels and of line labels, so point
/// labels' pipeline doesn't compile line placement.
pub const SYMBOL_INLINE_SHADERS: [Shader; 2] = [
    Shader {
        label: "symbol inline",
        parts: &[
            TYPES,
            DRAW,
            POINT,
            SYMBOL_HALVES,
            SYMBOL,
            SYMBOL_PLACE,
            SYMBOL_PLACE_INLINE,
            SYMBOL_POINT,
            SYMBOL_DRAW,
        ],
        vertex: "vs_symbol",
        fragment: "fs_symbol",
        compute: "",
    },
    Shader {
        label: "symbol line inline",
        parts: &[
            TYPES,
            DRAW,
            POINT,
            SYMBOL_HALVES,
            SYMBOL,
            SYMBOL_PLACE,
            SYMBOL_PLACE_INLINE,
            SYMBOL_LINE,
            LINE_LABELS_ALONE,
            SYMBOL_DRAW,
        ],
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
    parts: &[TYPES, DRAW, IMAGE, HILLSHADE, wgsl!("hillshade_layer")],
    vertex: "vs_hillshade",
    fragment: "fs_hillshade",
    compute: "",
};

/// A hillshade the terrain shades, drawing its unlit slopes into its
/// drape stack.
pub const HILLSHADE_SLOPES_SHADER: Shader = Shader {
    label: "hillshade slopes",
    parts: &[TYPES, DRAW, IMAGE, HILLSHADE, wgsl!("hillshade_layer")],
    vertex: "vs_hillshade",
    fragment: "fs_hillshade_slopes",
    compute: "",
};

pub const COLOR_RELIEF_SHADER: Shader = Shader {
    label: "color relief",
    parts: &[TYPES, DRAW, IMAGE, COLOR_RELIEF],
    vertex: "vs_color_relief",
    fragment: "fs_color_relief",
    compute: "",
};

/// Mip levels over regions of array layers: drape slots, raster pages.
pub const DOWNSAMPLE_SHADER: Shader = Shader {
    label: "downsample",
    parts: &[FETCH_REGIONS, DOWNSAMPLE, wgsl!("downsample_compute")],
    vertex: "",
    fragment: "",
    compute: "downsample",
};

/// Raster tiles' texels past their area, from the nearest inside: before
/// their mip levels, in the downsampler's bindings.
pub const EXTEND_SHADER: Shader = Shader {
    label: "extend",
    parts: &[FETCH_REGIONS, DOWNSAMPLE, wgsl!("downsample_compute")],
    vertex: "",
    fragment: "",
    compute: "extend",
};

/// [`DOWNSAMPLE_SHADER`] drawn, for devices without compute: one quad per
/// region into a view of the level's layer.
pub const MIP_DRAW_SHADER: Shader = Shader {
    label: "mip draw",
    parts: &[FETCH_REGIONS, DOWNSAMPLE, wgsl!("downsample_draw")],
    vertex: "vs_mip",
    fragment: "fs_mip",
    compute: "",
};

/// [`EXTEND_SHADER`] drawn, into a view of the extended layer.
pub const EXTEND_DRAW_SHADER: Shader = Shader {
    label: "extend draw",
    vertex: "vs_extend",
    fragment: "fs_extend",
    ..MIP_DRAW_SHADER
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
    parts: &[
        TYPES,
        DRAW,
        PATTERN,
        LIGHTING,
        EXTRUSION,
        EXTRUSION_FRAGMENT,
    ],
    vertex: "vs_extrusion",
    fragment: "fs_extrusion",
    compute: "",
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
/// Where clip distances are, whole extrusions and those cut of a placement
/// with one clip rectangle, clipped by its four planes, in one pipeline
/// without discard: [`EXTRUSION_SHADER`]'s whole pipeline's place. Vertex
/// only: its pipeline draws `EXTRUSION_SHADER`'s fragment stages, as a
/// fragment stage can't take the clip distances its vertex output carries.
pub const EXTRUSION_PLANES_SHADER: Shader = Shader {
    label: "extrusion planes",
    parts: &[CLIP_PLANES, TYPES, DRAW, PATTERN, LIGHTING, EXTRUSION],
    vertex: "vs_extrusion_planes",
    fragment: "",
    ..EXTRUSION_SHADER
};
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
    FILL_SHADER,
    FILL_OUTLINE_SHADER,
    LINE_SHADER,
    LINE_STYLED_SHADER,
    CIRCLE_SHADER,
    SYMBOL_SHADER,
    SYMBOL_PLACE_SHADERS[0],
    SYMBOL_PLACE_SHADERS[1],
    SYMBOL_INLINE_SHADERS[0],
    SYMBOL_INLINE_SHADERS[1],
    HEATMAP_SHADER,
    RASTER_SHADER,
    HILLSHADE_SHADER,
    HILLSHADE_SLOPES_SHADER,
    COLOR_RELIEF_SHADER,
    PREPARE_PAD_SHADER,
    PREPARE_SLOPE_SHADER,
    DOWNSAMPLE_SHADER,
    EXTEND_SHADER,
    MIP_DRAW_SHADER,
    EXTEND_DRAW_SHADER,
    CULL_SHADER,
];
