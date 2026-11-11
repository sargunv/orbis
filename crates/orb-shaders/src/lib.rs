//! WGSL modules and their composition.
//!
//! Each pipeline's source is the shared scene types, the bindings of its
//! kind (draw or cull) and its own module, concatenated. With `glsl`, also
//! WebGL 2's stages as the build wrote them in GLSL ES 3.00 (`glsl`).

#![forbid(unsafe_code)]

/// A WGSL module as the build script wrote it: without comments or
/// indentation, which only the source tree needs. WebGL 2's web build
/// (`glsl_only`, build.rs) carries none: its pipelines compile the build's
/// GLSL, so its shaders have no sources.
#[cfg(not(glsl_only))]
macro_rules! wgsl {
    ($name:literal) => {
        include_str!(concat!(env!("OUT_DIR"), "/wgsl/", $name, ".wgsl"))
    };
}
#[cfg(glsl_only)]
macro_rules! wgsl {
    ($name:literal) => {
        ""
    };
}

/// A WGSL module the build script generated.
#[cfg(not(glsl_only))]
macro_rules! generated {
    ($name:literal) => {
        include_str!(concat!(env!("OUT_DIR"), "/wgsl/", $name, ".wgsl"))
    };
}

mod catalog;
pub use catalog::*;
#[cfg(not(glsl_only))]
mod sources;

#[cfg(feature = "ir")]
mod ir {
    use std::sync::OnceLock;

    /// Where each of a shader's modules lies in the inflated whole, by
    /// entry point.
    type Ranges = &'static [(&'static str, std::ops::Range<usize>)];

    include!(concat!(env!("OUT_DIR"), "/ir.rs"));

    /// A shader's modules by entry point.
    type Modules = Vec<(&'static str, &'static [u8])>;

    /// Every shader's modules by label, inflated the first time a shader
    /// asks (a fraction of a millisecond).
    pub(crate) fn modules() -> &'static [(&'static str, Modules)] {
        static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
        static MODULES: OnceLock<Vec<(&str, Modules)>> = OnceLock::new();
        MODULES.get_or_init(|| {
            let bytes = BYTES.get_or_init(|| {
                let deflated = include_bytes!(concat!(env!("OUT_DIR"), "/ir.deflate"));
                miniz_oxide::inflate::decompress_to_vec(deflated)
                    .expect("inflates the build's modules")
            });
            (IR.iter())
                .map(|&(label, entries)| {
                    let modules = entries
                        .iter()
                        .map(|(entry, at)| (*entry, &bytes[at.clone()]));
                    (label, modules.collect())
                })
                .collect()
        })
    }
}

#[cfg(feature = "glsl")]
mod glsl;
#[cfg(feature = "glsl")]
pub use glsl::glsl;

#[cfg(test)]
mod split;

/// A shader's naga modules by entry point, postcard-encoded.
pub type EntryModules = &'static [(&'static str, &'static [u8])];

/// The override constant that makes a pipeline its plane variant
/// (`types.wgsl`'s `plane`): for views of affine tiles on the plane,
/// without the globe or terrain.
pub const PLANE: &str = "plane";

/// The override constant that, set false, makes a paint pipeline its
/// variant without property rows (`draw.wgsl`'s `feature_rows`): for nodes
/// that take no property from a feature's row or state.
pub const FEATURE_ROWS: &str = "feature_rows";

/// The override constant that, set false, makes a pipeline its variant
/// without feature state (`draw.wgsl`'s `feature_state`): for nodes whose
/// paint reads no feature's state.
pub const FEATURE_STATE: &str = "feature_state";

/// The override constant that, set to 11, makes a pipeline read data
/// textures 2048 words wide (`data_texture.wgsl`'s `data_width_log2`;
/// 4096 by default): for a device whose texture side is shorter
/// (orb-render's `data_width`).
pub const DATA_WIDTH_LOG2: &str = "data_width_log2";

/// Each shader's override constants by entry point, as the build script
/// found them in each entry point's module.
type Overrides = &'static [(
    &'static str,
    &'static [(&'static str, &'static [&'static str])],
)];

include!(concat!(env!("OUT_DIR"), "/overrides.rs"));

impl Shader {
    /// Whether the stage at entry point `entry` reads override constant
    /// `name`: a pipeline sets only the constants its stages read (wgpu
    /// makes each stage from a module of what its entry point uses).
    pub fn takes(&self, entry: &str, name: &str) -> bool {
        (OVERRIDES.iter())
            .find(|(label, _)| *label == self.label)
            .and_then(|(_, entries)| entries.iter().find(|(e, _)| *e == entry))
            .is_some_and(|(_, names)| names.contains(&name))
    }

    /// Its naga modules, which the build script parsed from
    /// [`source`](Self::source), one per entry point with only what it
    /// uses, postcard-encoded (the `ir` feature): devices that compile
    /// with naga make each stage's module from these rather than parse the
    /// WGSL and split it per stage.
    pub fn ir(&self) -> Option<EntryModules> {
        #[cfg(feature = "ir")]
        {
            ir::modules()
                .iter()
                .find(|(label, _)| *label == self.label)
                .map(|(_, modules)| modules.as_slice())
        }
        #[cfg(not(feature = "ir"))]
        None
    }
}

/// The scene tables' layouts, reflected from `types.wgsl`: each struct's
/// words and each member's first word.
pub mod layout {
    include!(concat!(env!("OUT_DIR"), "/layout.rs"));
}

/// A uniform's words, written at its struct's members as the shaders
/// declare them (`orb_shaders::layout`).
#[derive(Debug)]
pub struct Block(pub Vec<u32>);

impl Block {
    pub fn new(words: usize) -> Self {
        Self(vec![0; words])
    }

    pub fn u(&mut self, at: usize, values: &[u32]) -> &mut Self {
        self.0[at..at + values.len()].copy_from_slice(values);
        self
    }

    pub fn f(&mut self, at: usize, values: &[f32]) -> &mut Self {
        for (word, v) in self.0[at..at + values.len()].iter_mut().zip(values) {
            *word = v.to_bits();
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(shader: &Shader) -> naga::Module {
        parse(shader.label, &shader.source()).0
    }

    fn parse(label: &str, source: &str) -> (naga::Module, naga::valid::ModuleInfo) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(source)));
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::CLIP_DISTANCES,
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(source)));
        (module, info)
    }

    /// The cull module's functions keep no function-scope arrays: neither
    /// array variables nor dynamic indexing into an array value, which
    /// compilers lower to the same. An Adreno 830's Vulkan driver (Chrome
    /// 156) miscompiled the cull dispatch with them, writing no draws or
    /// losing the device. Shared helpers it doesn't call are compiled
    /// out.
    #[test]
    fn cull_shader_has_no_function_arrays() {
        let (module, info) = parse(CULL_SHADER.label, &CULL_SHADER.source());
        let is_array = |inner: &naga::TypeInner| matches!(inner, naga::TypeInner::Array { .. });
        let functions = module
            .functions
            .iter()
            .map(|(handle, f)| (f, &info[handle]))
            .chain(
                module
                    .entry_points
                    .iter()
                    .enumerate()
                    .map(|(i, e)| (&e.function, info.get_entry_point(i))),
            );
        for (function, function_info) in functions {
            let name = function.name.as_deref().unwrap_or("?");
            if !CULL.contains(&format!("fn {name}(")) {
                continue;
            }
            for (_, local) in function.local_variables.iter() {
                assert!(
                    !is_array(&module.types[local.ty].inner),
                    "{name}: function-scope array `{}`",
                    local.name.as_deref().unwrap_or("?"),
                );
            }
            for (_, expression) in function.expressions.iter() {
                if let naga::Expression::Access { base, .. } = *expression {
                    let base = function_info[base].ty.inner_with(&module.types);
                    assert!(!is_array(base), "{name}: indexes an array value");
                }
            }
        }
    }

    /// FNV-1a: a hash that stays the same across Rust releases.
    fn fnv(text: &str) -> u64 {
        text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// Each pipeline stage as the modern path compiles it, its override
    /// constants at their defaults, written out by naga's Metal back end
    /// and hashed, against `src/stage-hashes.txt`: a change to what any
    /// stage computes shows as a hash to review, and changes meant to
    /// keep the modern path's code (the WebGL 2 path's shared fixes, #140)
    /// prove they do. `UPDATE_GOLDEN=1` rewrites the file.
    #[test]
    fn stages_keep_their_hashes() {
        let mut text =
            String::from("# Each pipeline stage's Metal, hashed (`stages_keep_their_hashes`).\n");
        for shader in SHADERS {
            for (entry, one) in split::entry_modules(&validate(shader)) {
                let info = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::CLIP_DISTANCES,
                )
                .validate(&one)
                .unwrap_or_else(|e| panic!("{} {entry}: {e}", shader.label));
                let (one, info) = naga::back::pipeline_constants::process_overrides(
                    &one,
                    &info,
                    None,
                    &Default::default(),
                )
                .unwrap_or_else(|e| panic!("{} {entry}: {e}", shader.label));
                let options = naga::back::msl::Options {
                    lang_version: (2, 4),
                    ..Default::default()
                };
                let (msl, _) = naga::back::msl::write_string(
                    &one,
                    &info,
                    &options,
                    &naga::back::msl::PipelineOptions::default(),
                )
                .unwrap_or_else(|e| panic!("{} {entry}: {e}", shader.label));
                text.push_str(&format!("{} {entry} {:016x}\n", shader.label, fnv(&msl)));
            }
        }
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/stage-hashes.txt");
        if std::env::var_os("UPDATE_GOLDEN").is_some() {
            std::fs::write(path, &text).unwrap();
            return;
        }
        let golden = std::fs::read_to_string(path).unwrap_or_default();
        let changed: Vec<&str> = text
            .lines()
            .filter(|l| !golden.lines().any(|g| g == *l))
            .collect();
        assert!(
            changed.is_empty() && golden.lines().count() == text.lines().count(),
            "stages changed from {path}: {changed:?}; UPDATE_GOLDEN=1 rewrites it, once the change is the one meant"
        );
    }

    /// How deep structs nest in `ty`, arrays aside: 0 for a plain type.
    fn struct_depth(types: &naga::UniqueArena<naga::Type>, ty: naga::Handle<naga::Type>) -> usize {
        match &types[ty].inner {
            naga::TypeInner::Struct { members, .. } => {
                1 + (members.iter())
                    .map(|m| struct_depth(types, m.ty))
                    .max()
                    .unwrap_or(0)
            }
            naga::TypeInner::Array { base, .. } => struct_depth(types, *base),
            _ => 0,
        }
    }

    /// Where a uniform block's std140 layout (WGSL's uniform layout, which
    /// naga keeps in the GL variant) differs from the one ANGLE's Metal
    /// back end gives the block (`blocklayoutMetal.cpp`): there a `vec3`
    /// takes 16 bytes, a struct aligns to its widest member and arrays of
    /// scalars pack. ANGLE converts such a block on the CPU at the first
    /// draw after each write to its buffer, and reads the buffer to do it:
    /// one the GPU wrote (a copy of gathered writes) makes it wait for the
    /// GPU, which stalled WebGL 2 frames on Apple devices (#140). `None`
    /// when the two agree.
    fn metal_mismatch(
        types: &naga::UniqueArena<naga::Type>,
        ty: naga::Handle<naga::Type>,
        std_size: u32,
    ) -> Option<String> {
        use naga::{TypeInner, VectorSize};
        fn align(types: &naga::UniqueArena<naga::Type>, ty: naga::Handle<naga::Type>) -> u32 {
            match &types[ty].inner {
                TypeInner::Vector {
                    size: VectorSize::Bi,
                    ..
                }
                | TypeInner::Matrix {
                    rows: VectorSize::Bi,
                    ..
                } => 8,
                TypeInner::Vector { .. } | TypeInner::Matrix { .. } => 16,
                TypeInner::Array { base, .. } => align(types, *base),
                TypeInner::Struct { members, .. } => (members.iter())
                    .map(|m| align(types, m.ty))
                    .max()
                    .unwrap_or(4),
                _ => 4,
            }
        }
        /// A leaf's size in ANGLE's Metal layout.
        fn leaf(inner: &TypeInner) -> u32 {
            match *inner {
                TypeInner::Vector {
                    size: VectorSize::Bi,
                    ..
                } => 8,
                TypeInner::Vector { .. } => 16,
                TypeInner::Matrix { columns, rows, .. } => {
                    columns as u32 * if rows == VectorSize::Bi { 8 } else { 16 }
                }
                _ => 4,
            }
        }
        fn walk(
            types: &naga::UniqueArena<naga::Type>,
            ty: naga::Handle<naga::Type>,
            at: u32,
            metal: &mut u32,
            path: &str,
        ) -> Result<(), String> {
            let a = align(types, ty);
            match &types[ty].inner {
                TypeInner::Struct { members, .. } => {
                    *metal = metal.next_multiple_of(a);
                    for m in members {
                        let name = m.name.as_deref().unwrap_or("?");
                        walk(types, m.ty, at + m.offset, metal, &format!("{path}.{name}"))?;
                    }
                    *metal = metal.next_multiple_of(a);
                }
                TypeInner::Array {
                    base,
                    size: naga::ArraySize::Constant(n),
                    stride,
                } if matches!(
                    types[*base].inner,
                    TypeInner::Struct { .. } | TypeInner::Array { .. }
                ) =>
                {
                    for i in 0..n.get() {
                        walk(
                            types,
                            *base,
                            at + i * stride,
                            metal,
                            &format!("{path}[{i}]"),
                        )?;
                    }
                }
                TypeInner::Array {
                    base,
                    size: naga::ArraySize::Constant(n),
                    stride,
                } => {
                    *metal = metal.next_multiple_of(a);
                    let size = leaf(&types[*base].inner);
                    if *metal != at || size != *stride {
                        return Err(format!(
                            "`{path}` at {at}, stride {stride} in std140; {metal}, stride {size} in Metal"
                        ));
                    }
                    *metal += size * n.get();
                }
                inner => {
                    *metal = metal.next_multiple_of(a);
                    if *metal != at {
                        return Err(format!("`{path}` at {at} in std140, {metal} in Metal"));
                    }
                    if matches!(
                        inner,
                        TypeInner::Matrix {
                            rows: VectorSize::Bi,
                            ..
                        }
                    ) {
                        return Err(format!("`{path}`'s columns 16 apart in std140, 8 in Metal"));
                    }
                    *metal += leaf(inner);
                }
            }
            Ok(())
        }
        let mut metal = 0;
        if let Err(e) = walk(types, ty, 0, &mut metal, "") {
            return Some(e);
        }
        let metal = metal.next_multiple_of(align(types, ty));
        (metal > std_size).then(|| format!("{metal} bytes in Metal, {std_size} in std140"))
    }

    /// GLSL ES 3.10 built-ins, which WebGL 2's ES 3.00 lacks.
    const ES_310_ONLY: &[&str] = &[
        "bitCount(",
        "findLSB(",
        "findMSB(",
        "bitfieldExtract(",
        "bitfieldInsert(",
        "bitfieldReverse(",
        "packUnorm4x8(",
        "unpackUnorm4x8(",
        "packSnorm4x8(",
        "unpackSnorm4x8(",
        "imageLoad(",
        "imageStore(",
        "textureGather(",
        "atomicAdd(",
        "memoryBarrier",
        "uaddCarry(",
        "umulExtended(",
        "frexp(",
        "ldexp(",
    ];

    /// A vertex entry point's flat outputs that take the first vertex's
    /// value: WGSL's `flat` alone, or `flat, first`.
    fn first_vertex_outputs(module: &naga::Module) -> Vec<String> {
        let entry = &module.entry_points[0];
        let Some(result) = &entry.function.result else {
            return Vec::new();
        };
        let first = |binding: &Option<naga::Binding>| {
            matches!(
                binding,
                Some(naga::Binding::Location {
                    interpolation: Some(naga::Interpolation::Flat),
                    sampling,
                    ..
                }) if *sampling != Some(naga::Sampling::Either)
            )
        };
        match &module.types[result.ty].inner {
            naga::TypeInner::Struct { members, .. } => (members.iter())
                .filter(|m| first(&m.binding))
                .map(|m| m.name.clone().unwrap_or_default())
                .collect(),
            _ if first(&result.binding) => vec![entry.name.clone()],
            _ => Vec::new(),
        }
    }

    /// The WebGL 2 dry run (#140): every pipeline's GL variant
    /// (`Shader::gl_source`) written as GLSL ES 3.00 by naga, one entry
    /// point at a time as wgpu's GL back end compiles them, within
    /// WebGL 2's guaranteed limits, with no flat varying that needs the
    /// first vertex's value (GL's provoking vertex is the last). A shader
    /// change that breaks the GL path fails here, on the author's machine,
    /// in the same commit.
    #[test]
    fn gl_variants_fit_webgl2() {
        use naga::back::glsl;
        // WebGL 2's minimums (`MAX_VARYING_VECTORS`,
        // `MAX_TEXTURE_IMAGE_UNITS`, `MAX_*_UNIFORM_BLOCKS`,
        // `MAX_UNIFORM_BLOCK_SIZE`).
        const VARYINGS: usize = 15;
        const TEXTURES: usize = 16;
        const UNIFORM_BLOCKS: usize = 12;
        const UNIFORM_BLOCK_BYTES: u32 = 16 << 10;
        let mut failures = Vec::new();
        let mut written = 0;
        for shader in SHADERS {
            let Some(source) = shader.gl_source() else {
                assert!(
                    !shader.compute.is_empty() || !shader.vertex.is_empty(),
                    "{}",
                    shader.label
                );
                continue;
            };
            let label = shader.label;
            let (module, _) = parse(label, &source);
            let mut textures = std::collections::BTreeSet::new();
            // Every entry point, not only the catalog's two: pipelines pair
            // others too (`fs_terrain_depth`, `vs_extrusion_occlusion`).
            for (entry, one) in split::entry_modules(&module) {
                let stage = one.entry_points[0].stage;
                assert_ne!(stage, naga::ShaderStage::Compute, "{label} {entry}");
                // The input the HAL binds so Firefox doesn't fake one (#179).
                if stage == naga::ShaderStage::Vertex
                    && !one.entry_points[0].function.arguments.iter().any(|a| {
                        a.name.as_deref() == Some(ATTRIBUTE_ZERO)
                            && matches!(
                                a.binding,
                                Some(naga::Binding::Location { location: 0, .. })
                            )
                    })
                {
                    failures.push(format!("{label} {entry}: no u32 at @location(0)"));
                }
                let info = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::empty(),
                )
                .validate(&one)
                .unwrap_or_else(|e| panic!("{label} {entry}: {e}"));
                // Overrides at their defaults, as a pipeline without
                // constants makes them.
                let (one, info) = naga::back::pipeline_constants::process_overrides(
                    &one,
                    &info,
                    None,
                    &Default::default(),
                )
                .unwrap_or_else(|e| panic!("{label} {entry}: {e}"));
                let (mut one, mut info) = (one.into_owned(), info.into_owned());
                // wgpu's GL back end splits a fragment stage's struct
                // values into their members (`flatten_structs`): the
                // Adreno 619 evaluates struct members at mediump there.
                if stage == naga::ShaderStage::Fragment {
                    one = glsl::flatten_structs(&one, &info)
                        .unwrap_or_else(|e| panic!("{label} {entry}: {e}"));
                    info = naga::valid::Validator::new(
                        naga::valid::ValidationFlags::all(),
                        naga::valid::Capabilities::all(),
                    )
                    .validate_resolved_overrides(&one)
                    .unwrap_or_else(|e| panic!("{label} {entry} flattened: {e:?}"));
                }
                let mut fail = |what: String| failures.push(format!("{label} {entry}: {what}"));
                if stage == naga::ShaderStage::Vertex {
                    for name in first_vertex_outputs(&one) {
                        fail(format!(
                            "`{name}` is flat from the first vertex, which GL can't give (it \
                             takes the last): once it is the same at every vertex of a \
                             primitive, it's `@interpolate(flat, either)`"
                        ));
                    }
                }
                for (_, var) in one.global_variables.iter() {
                    if let naga::AddressSpace::Storage { .. } = var.space {
                        fail(format!(
                            "reads storage `{}`",
                            var.name.as_deref().unwrap_or("?")
                        ));
                    }
                }
                let options = glsl::Options {
                    version: glsl::Version::Embedded {
                        version: 300,
                        is_webgl: true,
                    },
                    writer_flags: glsl::WriterFlags::ADJUST_COORDINATE_SPACE
                        | glsl::WriterFlags::FORCE_POINT_SIZE,
                    binding_map: Default::default(),
                    zero_initialize_workgroup_memory: true,
                };
                let pipeline = glsl::PipelineOptions {
                    shader_stage: stage,
                    entry_point: entry.clone(),
                    multiview: None,
                };
                let mut glsl = String::new();
                let reflection = match glsl::Writer::new(
                    &mut glsl,
                    &one,
                    &info,
                    &options,
                    &pipeline,
                    naga::proc::BoundsCheckPolicies::default(),
                )
                .and_then(|mut w| w.write())
                {
                    Ok(r) => r,
                    Err(e) => {
                        fail(format!("doesn't write as GLSL ES 3.00: {e}"));
                        continue;
                    }
                };
                written += 1;
                for builtin in ES_310_ONLY.iter().filter(|b| glsl.contains(*b)) {
                    fail(format!("calls `{builtin}`, which GLSL ES 3.00 lacks"));
                }
                if stage == naga::ShaderStage::Vertex && reflection.varying.len() > VARYINGS {
                    fail(format!(
                        "{} varyings, past WebGL 2's {VARYINGS}",
                        reflection.varying.len()
                    ));
                }
                // A texture sampled with two samplers would take one of
                // them on wgpu's GL back end.
                let mut samplers = std::collections::BTreeMap::new();
                for mapping in reflection.texture_mapping.values() {
                    let texture = one.global_variables[mapping.texture].name.clone();
                    if samplers
                        .insert(texture.clone(), mapping.sampler)
                        .is_some_and(|s| s != mapping.sampler)
                    {
                        fail(format!(
                            "samples `{}` with two samplers",
                            texture.as_deref().unwrap_or_default()
                        ));
                    }
                    textures.insert(texture);
                }
                let mut layouter = naga::proc::Layouter::default();
                layouter.update(one.to_ctx()).expect("lays out the module");
                let uniforms: Vec<_> = (one.global_variables.iter())
                    .filter(|(_, v)| v.space == naga::AddressSpace::Uniform)
                    .collect();
                if uniforms.len() > UNIFORM_BLOCKS {
                    fail(format!(
                        "{} uniform blocks, past WebGL 2's {UNIFORM_BLOCKS}",
                        uniforms.len()
                    ));
                }
                for (_, var) in uniforms {
                    let name = var.name.as_deref().unwrap_or("?");
                    let size = layouter[var.ty].size;
                    if size > UNIFORM_BLOCK_BYTES {
                        fail(format!(
                            "uniform `{name}` of {size} bytes, past WebGL 2's {UNIFORM_BLOCK_BYTES}"
                        ));
                    }
                    // ANGLE refuses a block whose structs nest past 4
                    // levels, the block's own counted.
                    let depth = struct_depth(&one.types, var.ty);
                    if depth > 3 {
                        fail(format!(
                            "uniform `{name}` nests structs {depth} deep, past the 3 ANGLE takes in a block"
                        ));
                    }
                    if let Some(what) = metal_mismatch(&one.types, var.ty, size) {
                        fail(format!(
                            "uniform `{name}` lays out otherwise in ANGLE's Metal than in std140 ({what})"
                        ));
                    }
                }
            }
            if textures.len() > TEXTURES {
                failures.push(format!(
                    "{label}: {} textures, past WebGL 2's {TEXTURES} units",
                    textures.len()
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "WebGL 2's dry run:\n  {}",
            failures.join("\n  ")
        );
        assert!(written > 90, "only {written} stages written");
    }

    /// The draw pipelines that compile a plane variant (orb-render's
    /// paint, symbol and extrusion pipelines) take [`PLANE`] in their
    /// vertex stage, the paint and symbol ones [`FEATURE_ROWS`] too, the
    /// extrusions (whose variants keep rows) [`FEATURE_STATE`], and every
    /// stage that takes either validates and writes as Metal with each set
    /// as its variants set it, alone and together.
    #[test]
    fn variants_compile() {
        let rows = [
            &FILL_SHADER,
            &FILL_OUTLINE_SHADER,
            &LINE_SHADER,
            &LINE_STYLED_SHADER,
            &CIRCLE_SHADER,
            &SYMBOL_SHADERS[0],
            &SYMBOL_SHADERS[1],
        ];
        let planes = [
            &EXTRUSION_SHADER,
            &EXTRUSION_CUT_SHADER,
            &EXTRUSION_PLANES_SHADER,
        ];
        for (shader, variant) in
            (rows.map(|s| (s, FEATURE_ROWS)).into_iter()).chain(planes.map(|s| (s, FEATURE_STATE)))
        {
            assert!(shader.takes(shader.vertex, PLANE), "{}", shader.label);
            assert!(shader.takes(shader.vertex, variant), "{}", shader.label);
            for (entry, one) in split::entry_modules(&validate(shader)) {
                let info = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::CLIP_DISTANCES,
                )
                .validate(&one)
                .unwrap_or_else(|e| panic!("{} {entry}: {e}", shader.label));
                for set in [[true, false], [false, true], [true, true]] {
                    let constants: naga::back::PipelineConstants = [(PLANE, 1.0), (variant, 0.0)]
                        .into_iter()
                        .zip(set)
                        .filter(|&((name, _), on)| on && shader.takes(&entry, name))
                        .map(|((name, value), _)| (name.to_string(), value))
                        .collect();
                    if constants.is_empty() {
                        continue;
                    }
                    let (made, info) = naga::back::pipeline_constants::process_overrides(
                        &one, &info, None, &constants,
                    )
                    .unwrap_or_else(|e| panic!("{} {entry} {set:?}: {e}", shader.label));
                    let options = naga::back::msl::Options {
                        lang_version: (2, 4),
                        ..Default::default()
                    };
                    let pipeline = naga::back::msl::PipelineOptions::default();
                    naga::back::msl::write_string(&made, &info, &options, &pipeline)
                        .unwrap_or_else(|e| panic!("{} {entry} {set:?}: {e}", shader.label));
                }
            }
        }
    }

    #[test]
    fn shaders_validate() {
        for &shader in SHADERS {
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

    /// Each build-made module is the one the shader's WGSL parses to for
    /// its entry point, every entry point has one, and each decodes to a
    /// valid module.
    #[cfg(feature = "ir")]
    #[test]
    fn modules_are_their_sources() {
        for shader in SHADERS {
            let ir = shader
                .ir()
                .unwrap_or_else(|| panic!("{}: no modules", shader.label));
            let parsed = split::entry_modules(&validate(shader));
            assert_eq!(ir.len(), parsed.len(), "{}", shader.label);
            for (&(entry, bytes), (name, one)) in ir.iter().zip(&parsed) {
                assert_eq!(entry, name, "{}", shader.label);
                assert!(
                    postcard::to_allocvec(one).unwrap() == bytes,
                    "{} {entry}: the build's module differs",
                    shader.label
                );
                let made: naga::Module = postcard::from_bytes(bytes).expect("decodes");
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::CLIP_DISTANCES,
                )
                .validate(&made)
                .unwrap_or_else(|e| panic!("{} {entry}: {e}", shader.label));
            }
        }
    }

    /// Every pipeline stage as wgpu's Vulkan back end writes it (its
    /// `naga_options`), in each variant pipelines make of it (overrides at
    /// their defaults, the plane variant, the cull's `keep_all`, the ground
    /// mask, extrusions' elevation shapes), at the SPIR-V version wgpu-hal
    /// picks for Vulkan 1.1, 1.2 and 1.3, through SPIRV-Tools' `spirv-val`
    /// as the validation layer runs it, with `Restrict` bounds checks as on
    /// a device without robust access and `Unchecked` ones as on a device
    /// with it (Vulkan 1.3 requires robust image access). Explicit layout on
    /// Function and Workgroup variables (VUID-StandaloneSpirv-None-10684)
    /// and an `i32` one clamping `u32` image coordinates failed here until
    /// naga's patch (#35). Skipped, saying so, where
    /// `spirv-val` isn't installed (`brew install spirv-tools`, or the
    /// Vulkan SDK).
    #[test]
    fn spirv_validates_for_vulkan() {
        use naga::back::spv;
        use std::io::Write as _;
        use std::process::{Command, Stdio};
        if Command::new("spirv-val").arg("--version").output().is_err() {
            eprintln!("spirv_validates_for_vulkan: skipped, no spirv-val on PATH");
            return;
        }
        // (SPIR-V, the Vulkan environment wgpu-hal picks it for).
        const TARGETS: &[((u8, u8), &str)] = &[
            ((1, 3), "vulkan1.1"),
            ((1, 5), "vulkan1.2"),
            ((1, 6), "vulkan1.3"),
        ];
        // The override constants pipelines set (orb-render's plane
        // variants, `ORB_CULL=keep-all`, the ground mask on terrain, an
        // `ExtrusionShape`), each stage taking those it reads.
        const VARIANTS: &[&[(&str, f64)]] = &[
            &[],
            &[
                (PLANE, 1.0),
                ("keep_all", 1.0),
                ("ground_mask", 1.0),
                ("elevation_stops", 2.0),
                ("elevation_input", 1.0),
                ("elevation_curve", 1.0),
                ("elevation_dynamic", 1.0),
            ],
            &[
                ("elevation_stops", 16.0),
                ("elevation_input", 65.0),
                ("elevation_curve", 2.0),
                ("elevation_dynamic", 2.0),
            ],
            &[
                ("elevation_stops", 5.0),
                ("elevation_curve", 3.0),
                ("elevation_dynamic", 3.0),
            ],
        ];
        // (image and buffer bounds checks, what they're named).
        const POLICIES: &[(naga::proc::BoundsCheckPolicy, &str)] = &[
            (naga::proc::BoundsCheckPolicy::Restrict, "restrict"),
            (naga::proc::BoundsCheckPolicy::Unchecked, "robust"),
        ];
        // (what, the module's words, its Vulkan environment).
        let mut modules = Vec::new();
        let mut stages = 0;
        for shader in SHADERS {
            for (entry, one) in split::entry_modules(&validate(shader)) {
                stages += 1;
                let info = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::CLIP_DISTANCES,
                )
                .validate(&one)
                .unwrap_or_else(|e| panic!("{} {entry}: {e}", shader.label));
                let mut made = Vec::new();
                for variant in VARIANTS {
                    let constants: Vec<_> = (variant.iter())
                        .filter(|(name, _)| shader.takes(&entry, name))
                        .collect();
                    if made.contains(&constants) {
                        continue;
                    }
                    let map = (constants.iter())
                        .map(|&&(name, value)| (name.to_string(), value))
                        .collect();
                    let (one, info) =
                        naga::back::pipeline_constants::process_overrides(&one, &info, None, &map)
                            .unwrap_or_else(|e| {
                                panic!("{} {entry} {constants:?}: {e}", shader.label)
                            });
                    for &(lang_version, env) in TARGETS {
                        for &(policy, checks) in POLICIES {
                            for debug in [false, true] {
                                let mut flags = spv::WriterFlags::FORCE_POINT_SIZE;
                                flags.set(spv::WriterFlags::DEBUG, debug);
                                let options = spv::Options {
                                    lang_version,
                                    flags,
                                    bounds_check_policies: naga::proc::BoundsCheckPolicies {
                                        index: naga::proc::BoundsCheckPolicy::Restrict,
                                        buffer: policy,
                                        image_load: policy,
                                        binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
                                    },
                                    zero_initialize_workgroup_memory: if lang_version >= (1, 6) {
                                        spv::ZeroInitializeWorkgroupMemoryMode::Native
                                    } else {
                                        spv::ZeroInitializeWorkgroupMemoryMode::Polyfill
                                    },
                                    ..Default::default()
                                };
                                let words = spv::write_vec(&one, &info, &options, None)
                                    .unwrap_or_else(|e| {
                                        panic!("{} {entry} {constants:?}: {e}", shader.label)
                                    });
                                let what = format!(
                                    "{} {entry} {constants:?} (SPIR-V {}.{}, {env}, {checks}{})",
                                    shader.label,
                                    lang_version.0,
                                    lang_version.1,
                                    if debug { ", debug" } else { "" },
                                );
                                modules.push((what, words, env));
                            }
                        }
                    }
                    made.push(constants);
                }
            }
        }
        assert!(
            stages > 100 && modules.len() > 12 * stages,
            "only {} modules of {stages} stages",
            modules.len()
        );
        let validate = |(what, words, env): &(String, Vec<u32>, &str)| {
            let mut child = Command::new("spirv-val")
                .args(["--relax-block-layout", "--target-env", env, "-"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("runs spirv-val");
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            child.stdin.take().unwrap().write_all(&bytes).unwrap();
            let out = child.wait_with_output().unwrap();
            (!out.status.success()).then(|| {
                let text = String::from_utf8_lossy(&out.stderr);
                format!(
                    "{what}: {}",
                    text.lines().take(2).collect::<Vec<_>>().join(" ")
                )
            })
        };
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let failures: Vec<String> = std::thread::scope(|scope| {
            let chunks = modules.chunks(modules.len().div_ceil(threads));
            let workers: Vec<_> = chunks
                .map(|chunk| {
                    scope.spawn(move || chunk.iter().filter_map(validate).collect::<Vec<_>>())
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|w| w.join().unwrap())
                .collect()
        });
        assert!(
            failures.is_empty(),
            "spirv-val:\n  {}",
            failures.join("\n  ")
        );
    }
}
