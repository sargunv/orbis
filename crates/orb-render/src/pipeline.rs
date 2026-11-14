//! Pipelines from the shader catalog: each makes its stages' modules,
//! asks for the pipeline, and lets the modules go (pipelines keep what
//! they need of them).

use orb_hal::*;
use orb_shaders::Shader;

use crate::TargetFormat;

/// The format of the depth the map's passes have while terrain may draw
/// in them, which the 3D layers' takes then too: float depth beside a
/// stencil, which terrain layers draw each sample once by
/// ([`DepthState::once`]), or where the device has none, WebGPU's depth
/// with a stencil of at least 24 bits (GL JS's own). Without terrain the
/// 3D layers' depth is `Depth32Float`.
pub(crate) fn depth_format(caps: &Caps) -> TextureFormat {
    match caps.depth32_stencil8 {
        true => TextureFormat::Depth32FloatStencil8,
        false => TextureFormat::Depth24PlusStencil8,
    }
}

/// The depth test and write of every 3D pipeline: reversed Z, into a
/// depth attachment of `format`.
pub(crate) fn depth_test(format: TextureFormat) -> Option<DepthState> {
    Some(DepthState {
        format,
        write: true,
        test: true,
        once: false,
    })
}

/// What a pipeline draws into: a colour format at a sample count, and the
/// format of the depth attachment the map's passes carry while its style
/// has terrain ([`depth_format`]), which every pipeline drawing in them
/// declares: the terrain tests and writes it ([`depth_test`]), the rest
/// neither.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Attachments {
    pub(crate) color: TextureFormat,
    pub(crate) sample_count: u32,
    pub(crate) depth: Option<TextureFormat>,
}

impl From<TargetFormat> for Attachments {
    fn from(target: TargetFormat) -> Self {
        Self {
            color: target.color,
            sample_count: target.sample_count,
            depth: None,
        }
    }
}

impl Attachments {
    /// The colour target alone.
    pub(crate) fn format(self) -> TargetFormat {
        TargetFormat {
            color: self.color,
            sample_count: self.sample_count,
        }
    }
}

/// A render pipeline into one colour target: by default `shader`'s vertex
/// and fragment entry points, drawing triangles without culling, and
/// without a depth test or write where the target has a depth attachment.
/// Change the rest with struct update syntax over [`Draw::new`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct Draw<'a> {
    pub(crate) label: &'a str,
    pub(crate) vertex: (&'a Shader, &'a str),
    pub(crate) fragment: (&'a Shader, &'a str),
    /// The vertex and fragment stages' override constants.
    pub(crate) constants: [&'a [(&'a str, f64)]; 2],
    /// Its plane variant ([`orb_shaders::PLANE`], set in the stages that
    /// read it): for views without the globe, terrain or warped tiles.
    pub(crate) plane: bool,
    /// False for its variant without property rows
    /// ([`orb_shaders::FEATURE_ROWS`], set in the stages that read it): for
    /// nodes whose paint reads no feature's row or state.
    pub(crate) rows: bool,
    /// False for its variant without feature state
    /// ([`orb_shaders::FEATURE_STATE`], set in the stages that read it):
    /// for nodes whose paint reads no feature's state.
    pub(crate) state: bool,
    /// False for its variant without patterns ([`orb_shaders::PATTERNS`]):
    /// for nodes that draw none.
    pub(crate) patterns: bool,
    pub(crate) layouts: &'a [BindGroupLayoutId],
    pub(crate) target: ColorTargetState,
    pub(crate) sample_count: u32,
    pub(crate) depth: Option<DepthState>,
    pub(crate) topology: PrimitiveTopology,
    pub(crate) cull_back: bool,
}

impl<'a> Draw<'a> {
    pub(crate) fn new(
        shader: &'a Shader,
        layouts: &'a [BindGroupLayoutId],
        target: impl Into<Attachments>,
        blend: Option<BlendState>,
    ) -> Self {
        let target = target.into();
        Self {
            label: shader.label,
            vertex: (shader, shader.vertex),
            fragment: (shader, shader.fragment),
            constants: [&[], &[]],
            plane: false,
            rows: true,
            state: true,
            patterns: true,
            layouts,
            target: ColorTargetState {
                format: target.color,
                blend,
            },
            sample_count: target.sample_count,
            depth: target.depth.map(|format| DepthState {
                format,
                write: false,
                test: false,
                once: false,
            }),
            topology: PrimitiveTopology::TriangleList,
            cull_back: false,
        }
    }

    pub(crate) fn make(&self, gpu: &mut dyn Gpu) -> RenderPipelineId {
        let ((vertex, vertex_entry), (fragment, fragment_entry)) = (self.vertex, self.fragment);
        let variant = Variant {
            plane: self.plane,
            rows: self.rows,
            state: self.state,
            patterns: self.patterns,
        };
        let vertex_module = module(gpu, vertex, variant);
        let apart = vertex.label != fragment.label;
        let fragment_module = if apart {
            module(gpu, fragment, variant)
        } else {
            vertex_module
        };
        let [vertex_constants, fragment_constants] = [
            (vertex, vertex_entry, self.constants[0], vertex_module),
            (fragment, fragment_entry, self.constants[1], fragment_module),
        ]
        .map(|(shader, entry, constants, (_, specialised))| {
            let mut constants = constants.to_vec();
            if self.plane && shader.takes(entry, orb_shaders::PLANE) {
                constants.push((orb_shaders::PLANE, 1.0));
            }
            if !self.rows && shader.takes(entry, orb_shaders::FEATURE_ROWS) {
                constants.push((orb_shaders::FEATURE_ROWS, 0.0));
            }
            if !self.state && shader.takes(entry, orb_shaders::FEATURE_STATE) {
                constants.push((orb_shaders::FEATURE_STATE, 0.0));
            }
            if !self.patterns && shader.takes(entry, orb_shaders::PATTERNS) {
                constants.push((orb_shaders::PATTERNS, 0.0));
            }
            constants.extend(data_width(gpu, shader, entry));
            if specialised {
                constants.retain(|(name, _)| !VARIANT_CONSTANTS.contains(name));
            }
            constants
        });
        let (vertex_module, fragment_module) = (vertex_module.0, fragment_module.0);
        debug_assert!(
            !self.plane || vertex.takes(vertex_entry, orb_shaders::PLANE),
            "{}: no plane variant",
            self.label
        );
        debug_assert!(
            self.rows || vertex.takes(vertex_entry, orb_shaders::FEATURE_ROWS),
            "{}: no variant without rows",
            self.label
        );
        debug_assert!(
            self.state || vertex.takes(vertex_entry, orb_shaders::FEATURE_STATE),
            "{}: no variant without feature state",
            self.label
        );
        let pipeline = gpu.create_render_pipeline(&RenderPipelineDesc {
            label: self.label,
            bind_group_layouts: self.layouts,
            vertex: ShaderStage {
                module: vertex_module,
                entry_point: vertex_entry,
                constants: &vertex_constants,
            },
            fragment: ShaderStage {
                module: fragment_module,
                entry_point: fragment_entry,
                constants: &fragment_constants,
            },
            targets: &[self.target],
            depth: self.depth,
            topology: self.topology,
            cull_back: self.cull_back,
            sample_count: self.sample_count,
        });
        gpu.destroy_shader_module(vertex_module);
        if apart {
            gpu.destroy_shader_module(fragment_module);
        }
        pipeline
    }
}

/// A compute pipeline of `shader`'s compute entry point, with values for
/// its override constants.
pub(crate) fn compute(
    gpu: &mut dyn Gpu,
    shader: &Shader,
    layouts: &[BindGroupLayoutId],
    constants: &[(&str, f64)],
) -> ComputePipelineId {
    let set = |name: &str, default: bool| {
        (constants.iter())
            .find(|(n, _)| *n == name)
            .map_or(default, |&(_, v)| v != 0.0)
    };
    let variant = Variant {
        plane: set(orb_shaders::PLANE, false),
        rows: set(orb_shaders::FEATURE_ROWS, true),
        state: set(orb_shaders::FEATURE_STATE, true),
        patterns: set(orb_shaders::PATTERNS, true),
    };
    let (module, specialised) = module(gpu, shader, variant);
    let mut constants = constants.to_vec();
    if specialised {
        constants.retain(|(name, _)| !VARIANT_CONSTANTS.contains(name));
    }
    constants.extend(data_width(gpu, shader, shader.compute));
    let pipeline = gpu.create_compute_pipeline(&ComputePipelineDesc {
        label: shader.label,
        bind_group_layouts: layouts,
        compute: ShaderStage {
            module,
            entry_point: shader.compute,
            constants: &constants,
        },
    });
    gpu.destroy_shader_module(module);
    pipeline
}

/// The override constants that make a pipeline's variant: its module is
/// specialised to them ([`orb_shaders::specialise`]), and its stages set
/// none of them.
const VARIANT_CONSTANTS: [&str; 4] = [
    orb_shaders::PLANE,
    orb_shaders::FEATURE_ROWS,
    orb_shaders::FEATURE_STATE,
    orb_shaders::PATTERNS,
];

/// A pipeline's variant: [`Draw::plane`], [`Draw::rows`], [`Draw::state`]
/// and [`Draw::patterns`].
#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    all(cpu_driven, target_arch = "wasm32"),
    expect(dead_code, reason = "WebGL 2's build compiles the build's GLSL")
)]
struct Variant {
    plane: bool,
    rows: bool,
    state: bool,
    patterns: bool,
}

impl Variant {
    /// [`VARIANT_CONSTANTS`]' values.
    #[cfg(not(all(cpu_driven, target_arch = "wasm32")))]
    fn constants(self) -> [(&'static str, f64); 4] {
        let flags = [self.plane, self.rows, self.state, self.patterns];
        std::array::from_fn(|i| (VARIANT_CONSTANTS[i], f64::from(u8::from(flags[i]))))
    }
}

/// A module of `shader` specialised to `variant`, and whether it is:
/// Apple's compiler takes as long over code no branch reaches as over code
/// that runs, and about 30 ms over a stage with any texture in it, which
/// override constants alone left to it. The build's naga modules have the
/// device fold it into them (`ShaderModuleDesc::fold`), with no WGSL
/// parsed; the browser's WGSL is specialised as text
/// ([`orb_shaders::specialise`]). Not on a GL device, whose GLSL the build
/// wrote and whose compiler drops what a constant decides.
///
/// It reads the tables from data textures where the device's vertex
/// stage has no storage buffers (the tables' texture backing), and on a
/// device without compute (WebGL 2) every other table too, with the
/// built-ins written out ([`Shader::gl_source`]), and that written as GLSL
/// ES 3.00 by the build (`orb_shaders::glsl`), which a GL ES device
/// compiles as it is. WebGL 2's build (core-gl.wasm) carries that GLSL
/// alone, no WGSL.
fn module(gpu: &mut dyn Gpu, shader: &Shader, variant: Variant) -> (ShaderModuleId, bool) {
    #[cfg(cpu_driven)]
    let gl: Option<Glsl> = Some(orb_shaders::glsl);
    #[cfg(all(cpu_driven, target_arch = "wasm32"))]
    let (source, ir, glsl) = (String::new(), None, gl);
    #[cfg(not(target_arch = "wasm32"))]
    let (source, ir, glsl) = if gpu.caps().gl_shaders() {
        let source = (shader.gl_source())
            .unwrap_or_else(|| panic!("{} has no WebGL 2 source", shader.label));
        (source, None, gl)
    } else if !gpu.caps().vertex_storage {
        (shader.texture_tables_source(), None, None)
    } else {
        (shader.source(), shader.ir(), None)
    };
    #[cfg(all(not(cpu_driven), target_arch = "wasm32"))]
    let (source, ir, glsl) = (shader.source(), shader.ir(), None);
    #[cfg(not(all(cpu_driven, target_arch = "wasm32")))]
    let constants = variant.constants();
    #[cfg(not(all(cpu_driven, target_arch = "wasm32")))]
    let (source, fold, specialised): (_, &[_], _) = match (&glsl, ir) {
        (Some(_), _) => (source, &[], false),
        // Natively the data tables' WGSL too, which the device parses: so
        // both drivers' stages go through the same fold, and draw alike.
        (None, Some(_)) => (source, &constants, true),
        (None, None) if cfg!(not(target_arch = "wasm32")) => (source, &constants, true),
        (None, None) => match orb_shaders::specialise(&source, &constants) {
            Some(specialised) => (specialised, &[], true),
            None => (source, &[], false),
        },
    };
    #[cfg(all(cpu_driven, target_arch = "wasm32"))]
    let (fold, specialised) = {
        let _ = variant;
        (&[], false)
    };
    let module = gpu.create_shader_module(&ShaderModuleDesc {
        label: shader.label,
        source: source.into(),
        ir,
        glsl,
        fold,
    });
    (module, specialised)
}

/// The data width a stage reading data textures takes, where it isn't
/// its default (orb_shaders' `DATA_WIDTH_LOG2`).
fn data_width(gpu: &dyn Gpu, shader: &Shader, entry: &str) -> Option<(&'static str, f64)> {
    #[cfg(cpu_driven)]
    {
        use orb_shaders::DATA_WIDTH_LOG2;
        let width = crate::scene::data_width(&gpu.caps().limits);
        (!gpu.caps().vertex_storage && width != 1 << 12 && shader.takes(entry, DATA_WIDTH_LOG2))
            .then(|| (DATA_WIDTH_LOG2, f64::from(width.ilog2())))
    }
    #[cfg(not(cpu_driven))]
    {
        let _ = (gpu, shader, entry);
        None
    }
}

/// A pipeline's plane and full variants ([`Draw::plane`]), each made
/// once asked for. The plane variant draws a view of affine tiles on the
/// plane without terrain as the full one does, to within rounding: their
/// projections are written apart, and edges they antialias can differ by
/// a level or more. It compiles in about half the time on Adreno (#69).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Variants<P = RenderPipelineId>([Option<P>; 2]);

impl<P> Default for Variants<P> {
    fn default() -> Self {
        Self([None, None])
    }
}

impl Variants {
    /// Makes each of the variants `[plane, full]` asked for that isn't
    /// made.
    pub(crate) fn make(&mut self, gpu: &mut dyn Gpu, variants: [bool; 2], draw: Draw<'_>) {
        self.make_with(variants, |full| {
            Draw {
                plane: !full,
                ..draw
            }
            .make(gpu)
        });
    }
}

impl<P: Copy + Into<PipelineId>> Variants<P> {
    /// Makes each of the variants `[plane, full]` asked for that isn't
    /// made, with `make(full)`.
    pub(crate) fn make_with(&mut self, variants: [bool; 2], mut make: impl FnMut(bool) -> P) {
        for (full, made) in [false, true].into_iter().zip(&mut self.0) {
            if variants[usize::from(full)] && made.is_none() {
                *made = Some(make(full));
            }
        }
    }

    /// What a frame draws with, once compiled, and whether it is the one
    /// the frame [wants](Self::wanted): the full variant when the frame
    /// needs it (`full`), the plane one otherwise, or the full one in its
    /// place while the plane one compiles. A stand-in draws only to within
    /// rounding of what it stands for, so what recorded it records again
    /// once the wanted one compiles: a map that went on drawing with it
    /// would draw otherwise than one that never needed it. Only what it
    /// may draw with is asked after: a variant made ahead of need
    /// ([`Gpu::set_background_compiles`]) stays in the background.
    pub(crate) fn drawn(&self, gpu: &mut dyn Gpu, full: bool) -> Option<(P, bool)> {
        let [plane, whole] = self.0;
        let mut ready = |p: Option<P>| p.filter(|&p| gpu.pipeline_ready(p.into()));
        let drawn = if full {
            ready(whole)
        } else {
            ready(plane).or_else(|| ready(whole))
        };
        let wanted = self.wanted(full).map(Into::into);
        drawn.map(|p| (p, wanted == Some(p.into())))
    }

    /// The variant a frame waits for (`full` as in [`drawn`](Self::drawn)).
    pub(crate) fn wanted(&self, full: bool) -> Option<P> {
        if full {
            self.0[1]
        } else {
            self.0[0].or(self.0[1])
        }
    }
}

impl<P: crate::release::Release> crate::release::Release for Variants<P> {
    fn release(self, gpu: &mut dyn Gpu) {
        self.0.release(gpu);
    }
}
