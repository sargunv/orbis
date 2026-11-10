//! Extrusion layers draw into a target of their own, resolve it, then
//! composite it at their layer opacity once. The shared depth attachment
//! keeps occlusion between 3D layers and the terrain.
use crate::TargetFormat;
use crate::offscreen::{Apart, Composites, Mode, Offscreen};
use crate::paint::ElevationShape;
use crate::pipeline::{DEPTH, Draw};
use crate::release::Release;
use crate::scene::SceneBuffers;
use orb_hal::*;
use orb_shaders::{EXTRUSION_CUT_SHADER, EXTRUSION_PLANES_SHADER, EXTRUSION_SHADER, Shader};

#[derive(Debug)]
pub(crate) struct Extrusions {
    /// Per shape of colour over the elevation: whole buildings; cut ones
    /// clipped per fragment, which alone discard; and where clip distances
    /// are, cut ones clipped by planes; at one sample and at four.
    pipelines: Vec<(ElevationShape, Pipelines)>,
    layouts: [BindGroupLayoutId; 2],
    pub(crate) samples: u32,
    /// The targets' format: the map's.
    format: TextureFormat,
    depth: Option<TextureId>,
    size: [u32; 2],
    single_layer: bool,
}
/// An extrusion layer's node, drawn apart.
pub(crate) type NodeExtrusion = Apart;

type Pipelines = [[Option<RenderPipelineId>; 2]; 3];

impl Extrusions {
    pub(crate) fn new(
        scene: BindGroupLayoutId,
        node: BindGroupLayoutId,
        format: TextureFormat,
    ) -> Self {
        Self {
            pipelines: Vec::new(),
            layouts: [scene, node],
            samples: 4,
            format,
            depth: None,
            size: [0; 2],
            single_layer: false,
        }
    }

    /// The whole, cut and planes pipelines for `shape` at the current
    /// sample count.
    fn slot(&self, shape: ElevationShape) -> [Option<RenderPipelineId>; 3] {
        let samples = usize::from(self.samples == 4);
        (self.pipelines.iter())
            .find(|(s, _)| *s == shape)
            .map_or([None; 3], |(_, p)| p.map(|p| p[samples]))
    }

    /// Makes the pipelines for `shape` at the current sample count, for
    /// the first extrusion node of that shape.
    pub(crate) fn make(&mut self, gpu: &mut dyn Gpu, shape: ElevationShape) {
        let samples = usize::from(self.samples == 4);
        let index = match self.pipelines.iter().position(|(s, _)| *s == shape) {
            Some(i) => i,
            None => {
                self.pipelines.push((shape, [[None; 2]; 3]));
                self.pipelines.len() - 1
            }
        };
        let (whole, cut) = (EXTRUSION_SHADER, EXTRUSION_CUT_SHADER);
        // With clip distances, the cut pipeline keeps the buildings of
        // placements with several clip rectangles, and the planes one draws
        // the rest with the whole pipeline's fragment stage.
        let planes = gpu.caps().clip_distances;
        let stages = [
            Some(((&whole, whole.vertex), (&whole, whole.fragment))),
            Some((
                (
                    &cut,
                    if planes {
                        "vs_extrusion_cut_split"
                    } else {
                        cut.vertex
                    },
                ),
                (&cut, cut.fragment),
            )),
            planes.then_some((
                (&EXTRUSION_PLANES_SHADER, EXTRUSION_PLANES_SHADER.vertex),
                (&whole, whole.fragment),
            )),
        ];
        for (kind, stages) in stages.into_iter().enumerate() {
            if let (None, Some((vertex, fragment))) =
                (self.pipelines[index].1[kind][samples], stages)
            {
                let pipeline = self.pipeline(gpu, vertex, fragment, shape);
                self.pipelines[index].1[kind][samples] = Some(pipeline);
            }
        }
    }

    /// A pipeline of a vertex stage and a fragment stage, each a shader and
    /// its entry point.
    fn pipeline(
        &self,
        gpu: &mut dyn Gpu,
        vertex: (&Shader, &str),
        fragment: (&Shader, &str),
        shape: ElevationShape,
    ) -> RenderPipelineId {
        let [vertex_constants, fragment_constants] = shape.constants();
        let target = TargetFormat {
            color: self.format,
            sample_count: self.samples,
        };
        Draw {
            vertex,
            fragment,
            constants: [&vertex_constants, &fragment_constants],
            depth: DEPTH,
            cull_back: true,
            ..Draw::new(vertex.0, &self.layouts, target, None)
        }
        .make(gpu)
    }

    /// Whether every shape's pipelines made at the current sample count,
    /// and the composites', compiled.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu, composites: &Composites) -> bool {
        let samples = usize::from(self.samples == 4);
        let made: Vec<_> = (self.pipelines.iter())
            .flat_map(|(_, p)| p.map(|p| p[samples]))
            .collect();
        made.iter().any(Option::is_some)
            && made.iter().flatten().all(|&p| gpu.pipeline_ready(p.into()))
            && composites.ready(gpu, Mode::Over)
    }

    pub(crate) fn pipelines(&self, shape: ElevationShape) -> Vec<PipelineId> {
        self.slot(shape)
            .iter()
            .flatten()
            .map(|&p| p.into())
            .collect()
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
        let store = if self.single_layer {
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
                | if count == 1 {
                    TextureUsages::TRANSIENT_ATTACHMENT
                } else {
                    TextureUsages::empty()
                },
            sample_count: self.samples,
            mip_levels: 1,
            view_formats: &[],
        }));
        self.size = size;
        self.single_layer = count == 1;
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
            ("extrusion", &[self.format]),
            (size, self.samples),
            (Mode::Over, None),
            node.opacity,
        );
        if changed {
            node.forget_bundle(gpu);
        }
        changed
    }

    /// An extrusion node's pass into its target over the shared depth,
    /// replaying its draws, recorded once.
    pub(crate) fn prepare(
        &self,
        gpu: &mut dyn Gpu,
        node: &mut NodeExtrusion,
        (scene, group): (BindGroupId, BindGroupId),
        buffers: &SceneBuffers,
        (index, shape): (u32, ElevationShape),
        first: bool,
    ) -> RenderPass {
        let pipelines = self.slot(shape);
        let bundle = *node.bundle.get_or_insert_with(|| {
            let draw = RenderCommand::DrawIndexedIndirect {
                buffer: buffers.args,
                offset: crate::args_offset(index),
            };
            let mut commands = vec![
                RenderCommand::SetBindGroup {
                    index: 0,
                    group: scene,
                },
                RenderCommand::SetBindGroup { index: 1, group },
                RenderCommand::SetIndexBuffer {
                    buffer: buffers.compacted,
                    offset: 0,
                },
            ];
            for &pipeline in pipelines.iter().flatten() {
                commands.extend([RenderCommand::SetPipeline(pipeline), draw.clone()]);
            }
            gpu.create_render_bundle(
                &RenderBundleDesc {
                    label: "3d layer",
                    color_formats: &[self.format],
                    depth_format: Some(TextureFormat::Depth32Float),
                    sample_count: self.samples,
                },
                &commands,
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
        let pipelines: Vec<Pipelines> = self.pipelines.into_iter().map(|(_, p)| p).collect();
        (pipelines, self.depth).release(gpu);
    }
}
