//! Extrusion layers draw into a target of their own, resolve it, then
//! composite it at their layer opacity once. The shared depth attachment
//! keeps occlusion between 3D layers and the terrain.
use crate::TargetFormat;
use crate::offscreen::{Apart, Composites, Mode, Offscreen};
use crate::paint::ExtrusionShape;
use crate::pipeline::{DEPTH, Draw};
use crate::release::Release;
use crate::scene::SceneBuffers;
use orb_hal::*;
use orb_shaders::{EXTRUSION_CUT_SHADER, EXTRUSION_PLANES_SHADER, EXTRUSION_SHADER, Shader};

#[derive(Debug)]
pub(crate) struct Extrusions {
    /// Per shape of colour over the elevation and variant (`full`):
    /// whole buildings; cut ones clipped per fragment, which alone discard;
    /// and where clip distances are, cut ones clipped by planes; at one
    /// sample and at four.
    pipelines: Vec<((ExtrusionShape, bool), Pipelines)>,
    /// The shapes and variants of `pipelines` made ahead of need
    /// ([`Renderer::set_pipelines`](crate::Renderer::set_pipelines)) that no node has
    /// drawn with: what draws waits for none of them.
    warm: Vec<(ExtrusionShape, bool)>,
    /// The pipelines are made in their full variants, not their plane
    /// ones ([`Draw::plane`]): the paint pipelines' plane variants aren't
    /// made. On the moto g, the plane variant's vertex stage took 40% less
    /// of a pitched 3D frame's buildings (#180).
    full: bool,
    /// The variant they draw in: `full`'s, or while the plane pipelines
    /// compile, the full ones made before in their place, as the paint
    /// pipelines' `Variants::drawn` does: turning terrain off doesn't
    /// hide the buildings for the plane ones' compile.
    drawn_full: bool,
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

/// An extrusion node's two draws, each with the node group it binds: the
/// buildings its placements may draw whole, which the whole pipeline
/// draws, and its cut stream ([`Scene::cut_row`](crate::scene::Scene::cut_row)),
/// what they draw cut, which the cut pipelines draw.
#[derive(Clone, Debug)]
pub(crate) struct Streams {
    pub(crate) whole: (BindGroupId, RenderCommand),
    pub(crate) cut: (BindGroupId, RenderCommand),
}

impl Streams {
    /// Records each of `draws` whose pipeline is made, a pipeline over a
    /// stream, binding each stream's group as it changes.
    fn record<'a>(
        &self,
        draws: impl Iterator<Item = (Option<RenderPipelineId>, &'a (BindGroupId, RenderCommand))>,
        commands: &mut Vec<RenderCommand>,
    ) {
        let mut bound = None;
        for (pipeline, (group, draw)) in draws {
            let Some(pipeline) = pipeline else {
                continue;
            };
            if bound != Some(*group) {
                bound = Some(*group);
                commands.push(RenderCommand::SetBindGroup {
                    index: 1,
                    group: *group,
                });
            }
            commands.extend([RenderCommand::SetPipeline(pipeline), draw.clone()]);
        }
    }
}

type Pipelines = [[Option<RenderPipelineId>; 2]; 3];

impl Extrusions {
    pub(crate) fn new(
        scene: BindGroupLayoutId,
        node: BindGroupLayoutId,
        format: TextureFormat,
    ) -> Self {
        Self {
            pipelines: Vec::new(),
            warm: Vec::new(),
            full: true,
            drawn_full: true,
            layouts: [scene, node],
            samples: 1,
            format,
            depth: None,
            size: [0; 2],
            single_layer: false,
        }
    }

    /// Sets whether the pipelines are made in their full variants.
    pub(crate) fn set_full(&mut self, full: bool) {
        self.full = full;
    }

    /// Picks the variant the pipelines draw in (`drawn_full`): the one
    /// they're made in, or the full one while the plane one's pipelines
    /// compile, if every shape's full ones are made. True when that
    /// changed, and bundles recorded with the other draw no more.
    pub(crate) fn settle(&mut self, gpu: &mut dyn Gpu) -> bool {
        let samples = usize::from(self.samples == 4);
        let made = |full: bool| {
            (self.drawing())
                .filter(move |((_, f), p)| *f == full && p.iter().any(|p| p[samples].is_some()))
        };
        let plane_compiled = made(false).all(|(_, p)| {
            (p.iter().filter_map(|p| p[samples])).all(|p| gpu.pipeline_ready(p.into()))
        });
        let full_cover =
            made(false).all(|((shape, _), _)| made(true).any(|((s, _), _)| s == shape));
        let drawn = self.full || (!plane_compiled && full_cover);
        std::mem::replace(&mut self.drawn_full, drawn) != drawn
    }

    /// The pipelines of shapes nodes draw with.
    fn drawing(&self) -> impl Iterator<Item = &((ExtrusionShape, bool), Pipelines)> {
        (self.pipelines.iter()).filter(|(key, _)| !self.warm.contains(key))
    }

    /// The whole, cut and planes pipelines for `shape` at the current
    /// sample count in the variant they draw in.
    fn slot(&self, shape: ExtrusionShape) -> [Option<RenderPipelineId>; 3] {
        self.slot_in(shape, self.drawn_full)
    }

    /// The whole, cut and planes pipelines for `shape` at the current
    /// sample count in the full variant or the plane one.
    fn slot_in(&self, shape: ExtrusionShape, full: bool) -> [Option<RenderPipelineId>; 3] {
        let samples = usize::from(self.samples == 4);
        (self.pipelines.iter())
            .find(|(s, _)| *s == (shape, full))
            .map_or([None; 3], |(_, p)| p.map(|p| p[samples]))
    }

    /// Makes the pipelines for `shape` at the current sample count and
    /// variant, for the first extrusion node of that shape, or (`warm`)
    /// ahead of any.
    pub(crate) fn make(&mut self, gpu: &mut dyn Gpu, shape: ExtrusionShape, warm: bool) {
        let samples = usize::from(self.samples == 4);
        let key = (shape, self.full);
        let index = match self.pipelines.iter().position(|(s, _)| *s == key) {
            Some(i) => i,
            None => {
                self.pipelines.push((key, [[None; 2]; 3]));
                if warm {
                    self.warm.push(key);
                }
                self.pipelines.len() - 1
            }
        };
        if !warm {
            self.warm.retain(|k| *k != key);
        }
        let (whole, cut) = (EXTRUSION_SHADER, EXTRUSION_CUT_SHADER);
        // With clip distances, the cut pipeline keeps the buildings of
        // placements with several clip rectangles, and the planes one draws
        // the rest with the whole pipeline's fragment stage.
        let planes = gpu.caps().clip_distances;
        // Buildings shaded by their colour alone pass nothing else.
        let whole_stages = match shape.plain {
            true => ("vs_extrusion_plain", "fs_extrusion_plain"),
            false => (whole.vertex, whole.fragment),
        };
        let stages = [
            Some(((&whole, whole_stages.0), (&whole, whole_stages.1))),
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
        shape: ExtrusionShape,
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
            plane: !self.full,
            state: !shape.stateless,
            ..Draw::new(vertex.0, &self.layouts, target, None)
        }
        .make(gpu)
    }

    /// Whether every shape's pipelines made at the current sample count
    /// in the variant they draw in, and the composites', compiled.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu, composites: &Composites) -> bool {
        let samples = usize::from(self.samples == 4);
        let made: Vec<_> = (self.drawing())
            .filter(|((_, full), _)| *full == self.drawn_full)
            .flat_map(|(_, p)| p.map(|p| p[samples]))
            .collect();
        made.iter().any(Option::is_some)
            && made.iter().flatten().all(|&p| gpu.pipeline_ready(p.into()))
            && composites.ready(gpu, Mode::Over)
    }

    /// What a frame waits for to draw `shape`: the pipelines in the
    /// variant they're made in.
    pub(crate) fn pipelines(&self, shape: ExtrusionShape) -> Vec<PipelineId> {
        self.slot_in(shape, self.full)
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
        scene: BindGroupId,
        buffers: &SceneBuffers,
        (streams, shape): (&Streams, ExtrusionShape),
        first: bool,
    ) -> RenderPass {
        let [whole, cut, planes] = self.slot(shape);
        let bundle = *node.bundle.get_or_insert_with(|| {
            let mut commands = vec![
                RenderCommand::SetBindGroup {
                    index: 0,
                    group: scene,
                },
                RenderCommand::SetIndexBuffer {
                    buffer: buffers.compacted,
                    offset: 0,
                },
            ];
            // The buildings, each pipeline over its stream.
            let [w, c] = [&streams.whole, &streams.cut];
            let draws = [(whole, w), (cut, c), (planes, c)].into_iter();
            streams.record(draws, &mut commands);
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
            label: "3d layer",
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
