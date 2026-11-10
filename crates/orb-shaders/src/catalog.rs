// The shader catalog: composition of WGSL modules into pipeline sources.
// Shared with the build script, which reflects the shaders' uniforms.
pub(crate) const TYPES: &str = wgsl!("types");
const PROJECT: &str = wgsl!("project");
const DRAW: &str = concat!(
    wgsl!("color"),
    wgsl!("bezier"),
    wgsl!("draw"),
    wgsl!("project")
);
const FETCH: &str = wgsl!("fetch");
const PATTERN: &str = wgsl!("pattern");
const EXTRUSION: &str = wgsl!("extrusion");
const EXTRUSION_FRAGMENT: &str = wgsl!("extrusion_fragment");

const FILL: &str = wgsl!("fill");
const LINE: &str = wgsl!("line");
const POINT: &str = wgsl!("point");
const CIRCLE: &str = wgsl!("circle");
const SYMBOL: &str = wgsl!("symbol");
const SYMBOL_POINT: &str = wgsl!("symbol_point");
const SYMBOL_LINE: &str = wgsl!("symbol_line");
const HEATMAP: &str = wgsl!("heatmap");
const COMPOSITE: &str = concat!(wgsl!("color"), wgsl!("composite"));
pub(crate) const CULL: &str = wgsl!("cull");
const IMAGE: &str = wgsl!("image");
const DEM: &str = wgsl!("dem");
/// The atmosphere's sky and fog.
const ATMOSPHERE: &str = concat!(wgsl!("color"), wgsl!("atmosphere"));
/// v8's `light` on extrusions.
const LIGHTING: &str = wgsl!("lighting");
const RASTER: &str = wgsl!("raster");
const HILLSHADE: &str = wgsl!("hillshade");
const TERRAIN: &str = wgsl!("terrain");
const COLOR_RELIEF: &str = wgsl!("color_relief");
const PREPARE: &str = wgsl!("prepare");
const PREPARE_PAD: &str = wgsl!("prepare_pad");
/// First of a vertex-only shader's parts where clip distances are (wgpu's
/// `CLIP_DISTANCES`, WebGPU's `clip-distances`): it enables them, and the
/// parts' lines marked `//planes ` become code. A fragment stage can't take
/// clip distances, so such a shader leaves its fragment stages out.
const CLIP_PLANES: &str = "enable clip_distances;\n";
const PREPARE_SLOPE: &str = wgsl!("prepare_slope");

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
        let mut source: String = self.parts.concat();
        if self.parts.contains(&CLIP_PLANES) {
            source = source.replace("//planes ", "");
        }
        if self.parts.contains(&DRAW) {
            source.push_str(FETCH);
        }
        source
    }
}

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
        wgsl!("terrain_shade"),
    ],
    vertex: "vs_terrain",
    fragment: "fs_terrain_shaded",
    compute: "",
};

/// Clears the drape slots a frame redraws.
pub const DRAPE_SHADER: Shader = Shader {
    label: "drape",
    parts: &[wgsl!("drape")],
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

/// A linear-light map, drawn into an sRGB target, encoded into the host's.
pub const COMPOSITE_ENCODE_SHADER: Shader = Shader {
    label: "composite encode",
    parts: &[COMPOSITE],
    vertex: "vs_composite",
    fragment: "fs_encode",
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
/// Symbol quads: glyphs and icons, with their halos, of
/// point labels and of line labels (`symbol-placement: line` and
/// `line-center`): two variants, so point labels' pipeline doesn't compile
/// line placement.
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
    parts: &[TYPES, DRAW, IMAGE, DEM, COLOR_RELIEF],
    vertex: "vs_color_relief",
    fragment: "fs_color_relief",
    compute: "",
};

/// Mip levels over regions of array layers: drape slots, raster pages.
pub const DOWNSAMPLE_SHADER: Shader = Shader {
    label: "downsample",
    parts: &[wgsl!("color"), wgsl!("downsample")],
    vertex: "",
    fragment: "",
    compute: "downsample",
};

/// Raster tiles' texels past their area, from the nearest inside: before
/// their mip levels, in the downsampler's bindings.
pub const EXTEND_SHADER: Shader = Shader {
    label: "extend",
    parts: &[wgsl!("color"), wgsl!("downsample")],
    vertex: "",
    fragment: "",
    compute: "extend",
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
/// Cut extrusions of a placement with one clip rectangle, clipped by its
/// four planes: no discard. Where clip distances are. Vertex only: its
/// pipeline draws `EXTRUSION_SHADER`'s `fs_extrusion`, as a fragment stage
/// can't take the clip distances its vertex output carries.
pub const EXTRUSION_PLANES_SHADER: Shader = Shader {
    label: "extrusion planes",
    parts: &[CLIP_PLANES, TYPES, DRAW, PATTERN, LIGHTING, EXTRUSION],
    vertex: "vs_extrusion_cut_planes",
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
    COMPOSITE_ENCODE_SHADER,
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
    SYMBOL_SHADERS[0],
    SYMBOL_SHADERS[1],
    HEATMAP_SHADER,
    RASTER_SHADER,
    HILLSHADE_SHADER,
    HILLSHADE_SLOPES_SHADER,
    COLOR_RELIEF_SHADER,
    PREPARE_PAD_SHADER,
    PREPARE_SLOPE_SHADER,
    DOWNSAMPLE_SHADER,
    EXTEND_SHADER,
    CULL_SHADER,
];
