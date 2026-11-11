//! Pipelines from the shader catalog: each makes its stages' modules,
//! asks for the pipeline, and lets the modules go (pipelines keep what
//! they need of them).

use orb_hal::*;
use orb_shaders::Shader;

use crate::TargetFormat;

/// The depth test and write of every 3D pipeline: reversed Z, into the
/// shared depth attachment.
pub(crate) const DEPTH: Option<DepthState> = Some(DepthState {
    format: TextureFormat::Depth32Float,
    write: true,
});

/// A render pipeline into one colour target: by default `shader`'s vertex
/// and fragment entry points, drawing triangles without depth or culling.
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
        target: TargetFormat,
        blend: Option<BlendState>,
    ) -> Self {
        Self {
            label: shader.label,
            vertex: (shader, shader.vertex),
            fragment: (shader, shader.fragment),
            constants: [&[], &[]],
            plane: false,
            rows: true,
            state: true,
            layouts,
            target: ColorTargetState {
                format: target.color,
                blend,
            },
            sample_count: target.sample_count,
            depth: None,
            topology: PrimitiveTopology::TriangleList,
            cull_back: false,
        }
    }

    pub(crate) fn make(&self, gpu: &mut dyn Gpu) -> RenderPipelineId {
        let ((vertex, vertex_entry), (fragment, fragment_entry)) = (self.vertex, self.fragment);
        let vertex_module = module(gpu, vertex);
        let apart = vertex.label != fragment.label;
        let fragment_module = if apart {
            module(gpu, fragment)
        } else {
            vertex_module
        };
        let [vertex_constants, fragment_constants] = [
            (vertex, vertex_entry, self.constants[0]),
            (fragment, fragment_entry, self.constants[1]),
        ]
        .map(|(shader, entry, constants)| {
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
            constants.extend(data_width(gpu, shader, entry));
            constants
        });
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
    let module = module(gpu, shader);
    let mut constants = constants.to_vec();
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

/// A module of `shader`'s WGSL: reading the tables from data textures
/// where the device's vertex stage has no storage buffers (the tables'
/// texture backing), and on a device without compute (WebGL 2) every
/// other table too, with the built-ins written out
/// ([`Shader::gl_source`]), and that written as GLSL ES 3.00 by the build
/// (`orb_shaders::glsl`), which a GL ES device compiles as it is. WebGL 2's
/// build (core-gl.wasm) carries that GLSL alone, no WGSL.
fn module(gpu: &mut dyn Gpu, shader: &Shader) -> ShaderModuleId {
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
    gpu.create_shader_module(&ShaderModuleDesc {
        label: shader.label,
        source: source.into(),
        ir,
        glsl,
    })
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
/// plane without terrain exactly as the full one does, and compiles in
/// about half the time on Adreno (#69).
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

    /// What a frame draws with, once compiled: the full variant when the
    /// frame needs it (`full`), the plane one otherwise, or the full one
    /// in its place while it isn't made or compiled. Only what it may draw
    /// with is asked after: a variant made ahead of need
    /// ([`Gpu::set_background_compiles`]) stays in the background.
    pub(crate) fn drawn(&self, gpu: &mut dyn Gpu, full: bool) -> Option<P> {
        let [plane, whole] = self.0;
        let mut ready = |p: Option<P>| p.filter(|&p| gpu.pipeline_ready(p.into()));
        if full {
            ready(whole)
        } else {
            ready(plane).or_else(|| ready(whole))
        }
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
