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
        let pipeline = gpu.create_render_pipeline(&RenderPipelineDesc {
            label: self.label,
            bind_group_layouts: self.layouts,
            vertex: ShaderStage {
                module: vertex_module,
                entry_point: vertex_entry,
                constants: self.constants[0],
            },
            fragment: ShaderStage {
                module: fragment_module,
                entry_point: fragment_entry,
                constants: self.constants[1],
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
    let pipeline = gpu.create_compute_pipeline(&ComputePipelineDesc {
        label: shader.label,
        bind_group_layouts: layouts,
        compute: ShaderStage {
            module,
            entry_point: shader.compute,
            constants,
        },
    });
    gpu.destroy_shader_module(module);
    pipeline
}

/// A module of `shader`'s WGSL.
fn module(gpu: &mut dyn Gpu, shader: &Shader) -> ShaderModuleId {
    gpu.create_shader_module(&ShaderModuleDesc {
        label: shader.label,
        source: shader.source().into(),
        ir: shader.ir(),
    })
}
