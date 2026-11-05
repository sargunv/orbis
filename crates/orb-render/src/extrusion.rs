//! 3D layers (extrusions and models) draw into a target of their own,
//! resolve it, then composite it at their layer opacity once. The shared
//! depth attachment keeps occlusion between 3D layers.
use crate::offscreen::{Apart, Composites, Mode, Offscreen};
use crate::release::Release;
use orb_hal::*;
use orb_scene::SceneBuffers;
use orb_shaders::{EXTRUSION_CUT_SHADER, EXTRUSION_PLANES_SHADER, EXTRUSION_SHADER, Shader};

#[derive(Debug)]
pub(crate) struct Extrusions {
    /// Whole buildings; cut ones clipped per fragment, which alone discard;
    /// and where clip distances are, cut ones clipped by planes. Unlit then
    /// lit, at one sample and at four.
    pipelines: [[[Option<RenderPipelineId>; 2]; 2]; 3],
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
            pipelines: [[[None; 2]; 2]; 3],
            layouts: [scene, node],
            samples: 4,
            depth: None,
            size: [0; 2],
            single_layer: false,
            sampled: false,
        }
    }

    /// The whole, cut and planes pipelines at the current sample count, lit
    /// or not.
    fn slot(&self, lit: bool) -> [Option<RenderPipelineId>; 3] {
        self.pipelines
            .map(|p| p[usize::from(lit)][usize::from(self.samples == 4)])
    }

    /// Makes the pipeline at the current sample count, for the first
    /// extrusion node: lit with the node materials' group layout `lit`.
    pub(crate) fn make(&mut self, gpu: &mut dyn Gpu, lit: Option<BindGroupLayoutId>) {
        let samples = usize::from(self.samples == 4);
        let layouts: Vec<BindGroupLayoutId> = self.layouts.iter().copied().chain(lit).collect();
        let lit_of = |shader: Shader| if lit.is_some() { shader.lit() } else { shader };
        let (whole, cut) = (lit_of(EXTRUSION_SHADER), lit_of(EXTRUSION_CUT_SHADER));
        // With clip distances, the cut pipeline keeps the buildings of
        // placements with several clip rectangles, and the planes one draws
        // the rest with the whole pipeline's fragment stage.
        let planes = gpu.caps().clip_distances;
        let stages = [
            Some(((whole, whole.vertex), (whole, whole.fragment))),
            Some((
                (
                    cut,
                    if planes {
                        "vs_extrusion_cut_split"
                    } else {
                        cut.vertex
                    },
                ),
                (cut, cut.fragment),
            )),
            planes.then_some((
                (EXTRUSION_PLANES_SHADER, EXTRUSION_PLANES_SHADER.vertex),
                (whole, whole.fragment),
            )),
        ];
        for (kind, stages) in stages.into_iter().enumerate() {
            let at = (kind, usize::from(lit.is_some()), samples);
            if let (None, Some((vertex, fragment))) = (self.pipelines[at.0][at.1][at.2], stages) {
                let pipeline = self.pipeline(gpu, vertex, fragment, &layouts);
                self.pipelines[at.0][at.1][at.2] = Some(pipeline);
            }
        }
    }

    /// A pipeline of a vertex stage and a fragment stage, each a shader and
    /// its entry point.
    fn pipeline(
        &self,
        gpu: &mut dyn Gpu,
        (vertex, vertex_entry): (Shader, &str),
        (fragment, fragment_entry): (Shader, &str),
        layouts: &[BindGroupLayoutId],
    ) -> RenderPipelineId {
        let vertex_module = crate::shader_module(gpu, &vertex);
        let apart = vertex.label != fragment.label || vertex.lit != fragment.lit;
        let fragment_module = if apart {
            crate::shader_module(gpu, &fragment)
        } else {
            vertex_module
        };
        let pipeline = gpu.create_render_pipeline(&RenderPipelineDesc {
            cull_mode: Some(Face::Back),
            label: vertex.label,
            bind_group_layouts: layouts,
            vertex: ShaderStage {
                module: vertex_module,
                entry_point: vertex_entry,
                constants: &[],
            },
            fragment: ShaderStage {
                module: fragment_module,
                entry_point: fragment_entry,
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
        gpu.destroy_shader_module(vertex_module);
        if apart {
            gpu.destroy_shader_module(fragment_module);
        }
        pipeline
    }

    /// Whether the pipeline at the current sample count, lit or not, and
    /// the composites', compiled.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu, composites: &Composites, lit: bool) -> bool {
        let slot = self.slot(lit);
        slot[0].is_some()
            && slot.iter().flatten().all(|&p| gpu.pipeline_ready(p.into()))
            && composites.ready(gpu, Mode::Over)
    }

    pub(crate) fn pipelines(&self, lit: bool) -> Vec<PipelineId> {
        self.slot(lit).iter().flatten().map(|&p| p.into()).collect()
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

    /// An extrusion node's pass; lit with the node materials' group
    /// `materials`.
    #[expect(clippy::too_many_arguments, reason = "one recorded draw")]
    pub(crate) fn prepare(
        &self,
        gpu: &mut dyn Gpu,
        node: &mut NodeExtrusion,
        (scene, group): (BindGroupId, BindGroupId),
        materials: Option<BindGroupId>,
        buffers: &SceneBuffers,
        index: u32,
        first: bool,
    ) -> RenderPass {
        let pipelines = self.slot(materials.is_some());
        self.pass(gpu, node, first, || {
            let lit = materials.map(|group| RenderCommand::SetBindGroup { index: 2, group });
            let draw = RenderCommand::DrawIndexedIndirect {
                buffer: buffers.args,
                offset: crate::args_offset(index),
            };
            let mut commands: Vec<RenderCommand> = lit.into_iter().collect();
            commands.extend([
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
            ]);
            for &pipeline in pipelines.iter().flatten() {
                commands.extend([RenderCommand::SetPipeline(pipeline), draw.clone()]);
            }
            commands
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
