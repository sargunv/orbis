//! 3D layers (extrusions and models) draw into a target of their own,
//! resolve it, then composite it at their layer opacity once. The shared
//! depth attachment keeps occlusion between 3D layers.
use crate::offscreen::{Apart, Composites, Mode, Offscreen};
use crate::release::Release;
use orb_hal::*;
use orb_scene::SceneBuffers;
use orb_shaders::EXTRUSION_SHADER;

#[derive(Debug)]
pub(crate) struct Extrusions {
    /// At one sample and at four.
    pipelines: [Option<RenderPipelineId>; 2],
    layouts: [BindGroupLayoutId; 2],
    pub(crate) samples: u32,
    depth: Option<TextureId>,
    size: [u32; 2],
    single_layer: bool,
    /// The depth is kept and read after the 3D passes (`occlusion`).
    sampled: bool,
}
/// A 3D layer's node, drawn apart.
pub(crate) type NodeExtrusion = Apart;

impl Extrusions {
    pub(crate) fn new(scene: BindGroupLayoutId, node: BindGroupLayoutId) -> Self {
        Self {
            pipelines: [None; 2],
            layouts: [scene, node],
            samples: 4,
            depth: None,
            size: [0; 2],
            single_layer: false,
            sampled: false,
        }
    }

    /// Makes the pipeline at the current sample count, for the first
    /// extrusion node.
    pub(crate) fn make(&mut self, gpu: &mut dyn Gpu) {
        self.pipeline(gpu);
    }

    /// Whether the pipeline at the current sample count, and the
    /// composites', compiled.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu, composites: &Composites) -> bool {
        self.pipelines[usize::from(self.samples == 4)].is_some_and(|p| gpu.pipeline_ready(p.into()))
            && composites.ready(gpu, Mode::Over)
    }

    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        self.pipelines[usize::from(self.samples == 4)]
            .iter()
            .map(|&p| p.into())
            .collect()
    }

    fn pipeline(&mut self, gpu: &mut dyn Gpu) {
        let slot = usize::from(self.samples == 4);
        if self.pipelines[slot].is_some() {
            return;
        }
        let shader = EXTRUSION_SHADER;
        let module = crate::shader_module(gpu, &shader);
        let pipeline = gpu.create_render_pipeline(&RenderPipelineDesc {
            cull_mode: Some(Face::Back),
            label: shader.label,
            bind_group_layouts: &self.layouts,
            vertex: ShaderStage {
                module,
                entry_point: shader.vertex,
                constants: &[],
            },
            fragment: ShaderStage {
                module,
                entry_point: shader.fragment,
                constants: &[],
            },
            targets: &[ColorTargetState {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
            }],
            depth: Some(DepthState {
                format: TextureFormat::Depth32Float,
                write: true,
                compare: CompareFunction::GreaterEqual,
            }),
            topology: PrimitiveTopology::TriangleList,
            sample_count: self.samples,
        });
        gpu.destroy_shader_module(module);
        self.pipelines[slot] = Some(pipeline);
    }
    pub(crate) fn set_antialias(&mut self, antialias: bool) -> bool {
        let samples = if antialias { 4 } else { 1 };
        if samples == self.samples {
            return false;
        }
        self.samples = samples;
        self.size = [0; 2];
        true
    }
    /// The depth 3D passes share, once a resize made it, and how its
    /// passes store it: a single pass's is transient.
    pub(crate) fn depth(&self) -> Option<(TextureId, StoreOp)> {
        let store = if self.single_layer && !self.sampled {
            StoreOp::Discard
        } else {
            StoreOp::Store
        };
        self.depth.map(|d| (d, store))
    }

    /// Sizes the depth `count` 3D passes share: extrusion layers and
    /// terrain stacks.
    pub(crate) fn resize(&mut self, gpu: &mut dyn Gpu, size: [u32; 2], count: usize) -> bool {
        if count == 0 || (self.size == size && self.single_layer == (count == 1)) {
            return false;
        }
        self.depth.take().release(gpu);
        self.depth = Some(gpu.create_texture(&TextureDesc {
            label: "3d depth",
            width: size[0],
            height: size[1],
            array_layers: 1,
            dimension: TextureViewDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT
                | if self.sampled {
                    TextureUsages::TEXTURE_BINDING
                } else if count == 1 {
                    TextureUsages::TRANSIENT_ATTACHMENT
                } else {
                    TextureUsages::empty()
                },
            sample_count: self.samples,
            mip_levels: 1,
        }));
        self.size = size;
        self.single_layer = count == 1;
        true
    }
    /// Keeps the depth for reading after the 3D passes, or lets a single
    /// pass's be transient; whether that changed (it's made again).
    pub(crate) fn set_sampled(&mut self, sampled: bool) -> bool {
        if sampled == self.sampled {
            return false;
        }
        self.sampled = sampled;
        self.size = [0; 2];
        true
    }

    /// Gives a node its target of `size` and its composite, unless it has
    /// them at this size and sample count; whether it changed.
    pub(crate) fn resize_node(
        &self,
        gpu: &mut dyn Gpu,
        composites: &mut Composites,
        node: &mut NodeExtrusion,
        size: [u32; 2],
    ) -> bool {
        let changed = Offscreen::fit(
            &mut node.offscreen,
            gpu,
            composites,
            ("extrusion", &[TextureFormat::Rgba8Unorm]),
            (size, self.samples),
            (Mode::Over, None),
            node.opacity,
        );
        if changed {
            node.forget_bundle(gpu);
        }
        changed
    }

    pub(crate) fn prepare(
        &self,
        gpu: &mut dyn Gpu,
        node: &mut NodeExtrusion,
        (scene, group): (BindGroupId, BindGroupId),
        buffers: &SceneBuffers,
        index: u32,
        first: bool,
    ) -> RenderPass {
        let pipeline = self.pipelines[usize::from(self.samples == 4)];
        self.pass(gpu, node, first, || {
            vec![
                RenderCommand::SetPipeline(pipeline.expect("made before its nodes prepare")),
                RenderCommand::SetBindGroup {
                    index: 0,
                    group: scene,
                },
                RenderCommand::SetBindGroup { index: 1, group },
                RenderCommand::SetIndexBuffer {
                    buffer: buffers.compacted,
                    format: IndexFormat::Uint32,
                    offset: 0,
                },
                RenderCommand::DrawIndexedIndirect {
                    buffer: buffers.args,
                    offset: crate::args_offset(index),
                },
            ]
        })
    }

    /// A 3D layer's pass into its target over the shared depth, replaying
    /// its draws, recorded once from `commands`.
    pub(crate) fn pass(
        &self,
        gpu: &mut dyn Gpu,
        node: &mut NodeExtrusion,
        first: bool,
        commands: impl FnOnce() -> Vec<RenderCommand>,
    ) -> RenderPass {
        let bundle = *node.bundle.get_or_insert_with(|| {
            gpu.create_render_bundle(
                &RenderBundleDesc {
                    label: "3d layer",
                    color_formats: &[TextureFormat::Rgba8Unorm],
                    depth_format: Some(TextureFormat::Depth32Float),
                    sample_count: self.samples,
                },
                &commands(),
            )
        });
        RenderPass {
            label: "3d layer".into(),
            colors: vec![
                node.offscreen
                    .as_ref()
                    .expect("sized")
                    .target
                    .attachment(0, LoadOp::Clear([0.0; 4])),
            ],
            depth: Some(DepthAttachment {
                store: self.depth().expect("sized").1,
                target: self.depth.unwrap(),
                load: if first {
                    LoadOp::Clear(0.0)
                } else {
                    LoadOp::Load
                },
            }),
            commands: vec![RenderCommand::ExecuteBundles(vec![bundle])],
        }
    }
}

impl crate::release::Release for Extrusions {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.pipelines, self.depth).release(gpu);
    }
}
