//! Extrusion layers draw into a target of their own, resolve it, then
//! composite it at their layer opacity once. The shared depth attachment
//! keeps occlusion between 3D layers and the terrain.
//!
//! What a layer darkens on the ground beneath its buildings (style-spec
//! #1797's shadows, #1796's ground occlusion) draws after them in the
//! same pass, from the buildings' own vertices (`extrusion_ground.wgsl`):
//! black whose alpha is how dark, the darkest kept where shapes overlap,
//! depth-tested against the buildings, so what they stand over is never
//! shaded (in a city at a pitch, most of it), and the composite darkens
//! the map beneath by what is left.
//!
//! On a plane's terrain they draw instead into the ground mask, flat, one
//! pass before the layer's, and the layer's pass draws the terrain's mesh
//! after the buildings, reading the mask where it lies (`Terrain`'s
//! decal), depth-tested against them in the same way: the darkness
//! falls on the surface drawn, wherever the DEM bends between the
//! buildings' vertices.
use crate::TargetFormat;
use crate::offscreen::{Apart, Composites, Mode, Offscreen, Target};
use crate::paint::ElevationShape;
use crate::pipeline::{DEPTH, Draw};
use crate::release::Release;
use crate::scene::SceneBuffers;
use orb_hal::*;
use orb_shaders::{
    EXTRUSION_CUT_SHADER, EXTRUSION_GROUND_SHADER, EXTRUSION_PLANES_SHADER, EXTRUSION_SHADER,
    Shader,
};

/// What an extrusion node darkens on the ground beneath its buildings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Ground {
    /// style-spec#1797's shadows, cast from the light.
    pub(crate) shadows: bool,
    /// style-spec#1796's ambient occlusion along the walls' feet.
    pub(crate) occlusion: bool,
    /// On a plane's terrain: into the ground mask, which the terrain's
    /// decal carries onto the terrain.
    pub(crate) mask: bool,
}

impl Ground {
    /// Whether it darkens anything.
    pub(crate) fn any(self) -> bool {
        self.shadows || self.occlusion
    }

    /// What of it a node's recorded draw holds: none into the mask, which
    /// draws in a pass of its own.
    pub(crate) fn recorded(self) -> Self {
        if self.mask { Self::default() } else { self }
    }

    /// Each pass's vertex entry point, if it draws.
    fn passes(self) -> [(bool, &'static str); 2] {
        [
            (self.shadows, "vs_extrusion_shadow"),
            (self.occlusion, "vs_extrusion_occlusion"),
        ]
    }
}

#[derive(Debug)]
pub(crate) struct Extrusions {
    /// Per shape of colour over the elevation and variant (`full`):
    /// whole buildings; cut ones clipped per fragment, which alone discard;
    /// and where clip distances are, cut ones clipped by planes; at one
    /// sample and at four.
    pipelines: Vec<((ElevationShape, bool), Pipelines)>,
    /// The shapes and variants of `pipelines` made ahead of need
    /// ([`Renderer::prewarm`](crate::Renderer::prewarm)) that no node has
    /// drawn with: what draws waits for none of them.
    warm: Vec<(ElevationShape, bool)>,
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
    /// The shadow and ground occlusion passes, at one sample and at four,
    /// made for the first node that draws each; and into the ground mask,
    /// likewise.
    ground: [[Option<RenderPipelineId>; 2]; 2],
    mask: [[Option<RenderPipelineId>; 2]; 2],
    /// The ground mask, its size and samples, made for the first node
    /// drawing into it. It draws multisampled as the layer's target does,
    /// resolved into the texture the decal samples: drawn at one sample,
    /// every shadow's edge stood in steps along the terrain.
    mask_target: Option<(Target, [u32; 2], u32)>,
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
/// what they draw cut, which the cut pipelines draw. What the buildings
/// darken on the ground draws over both.
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
            ground: [[None; 2]; 2],
            mask: [[None; 2]; 2],
            mask_target: None,
            layouts: [scene, node],
            samples: 1,
            format,
            depth: None,
            size: [0; 2],
            single_layer: false,
        }
    }

    /// Sets whether the pipelines are made in their full variants; true
    /// when that changed, and the new variant's are to be made.
    pub(crate) fn set_full(&mut self, full: bool) -> bool {
        std::mem::replace(&mut self.full, full) != full
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
    fn drawing(&self) -> impl Iterator<Item = &((ElevationShape, bool), Pipelines)> {
        (self.pipelines.iter()).filter(|(key, _)| !self.warm.contains(key))
    }

    /// The whole, cut and planes pipelines for `shape` at the current
    /// sample count in the variant they draw in.
    fn slot(&self, shape: ElevationShape) -> [Option<RenderPipelineId>; 3] {
        self.slot_in(shape, self.drawn_full)
    }

    /// The whole, cut and planes pipelines for `shape` at the current
    /// sample count in the full variant or the plane one.
    fn slot_in(&self, shape: ElevationShape, full: bool) -> [Option<RenderPipelineId>; 3] {
        let samples = usize::from(self.samples == 4);
        (self.pipelines.iter())
            .find(|(s, _)| *s == (shape, full))
            .map_or([None; 3], |(_, p)| p.map(|p| p[samples]))
    }

    /// Makes the pipelines for `shape` at the current sample count and
    /// variant, for the first extrusion node of that shape, or (`warm`)
    /// ahead of any.
    pub(crate) fn make(&mut self, gpu: &mut dyn Gpu, shape: ElevationShape, warm: bool) {
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

    /// Makes the ground passes `ground` draws at the current sample count,
    /// unless they are made: depth-tested against what stands before them
    /// (the terrain, earlier 3D layers, the node's own buildings) without
    /// writing it, keeping the darkest of what overlaps.
    pub(crate) fn make_ground(&mut self, gpu: &mut dyn Gpu, ground: Ground) {
        if ground.mask {
            self.make_mask(gpu, ground);
            return;
        }
        let samples = usize::from(self.samples == 4);
        let target = TargetFormat {
            color: self.format,
            sample_count: self.samples,
        };
        let shader = &EXTRUSION_GROUND_SHADER;
        for (kind, (on, entry)) in ground.passes().into_iter().enumerate() {
            if on && self.ground[kind][samples].is_none() {
                let draw = Draw {
                    vertex: (shader, entry),
                    depth: Some(DepthState {
                        format: TextureFormat::Depth32Float,
                        write: false,
                    }),
                    ..Draw::new(shader, &self.layouts, target, Some(BlendState::Max))
                };
                self.ground[kind][samples] = Some(draw.make(gpu));
            }
        }
    }

    /// Makes the ground passes `ground` draws into the ground mask at the
    /// current sample count, unless they are made: the darkness alone,
    /// flat on its plane, the darkest kept.
    fn make_mask(&mut self, gpu: &mut dyn Gpu, ground: Ground) {
        let samples = usize::from(self.samples == 4);
        let target = TargetFormat {
            color: TextureFormat::R8Unorm,
            sample_count: self.samples,
        };
        let shader = &EXTRUSION_GROUND_SHADER;
        for (kind, (on, entry)) in ground.passes().into_iter().enumerate() {
            if on && self.mask[kind][samples].is_none() {
                let draw = Draw {
                    label: "extrusion ground mask",
                    vertex: (shader, entry),
                    fragment: (shader, "fs_extrusion_ground_mask"),
                    constants: [&[("ground_mask", 1.0)], &[]],
                    ..Draw::new(shader, &self.layouts, target, Some(BlendState::Max))
                };
                self.mask[kind][samples] = Some(draw.make(gpu));
            }
        }
    }

    /// The ground passes `ground` draws whose pipelines are made: into the
    /// layer's target or into the mask, at the current sample count.
    fn ground_made(&self, ground: Ground) -> [Option<RenderPipelineId>; 2] {
        let samples = usize::from(self.samples == 4);
        let mut made = [None; 2];
        for (kind, (on, _)) in ground.passes().into_iter().enumerate() {
            let pipeline = if ground.mask {
                self.mask[kind][samples]
            } else {
                self.ground[kind][samples]
            };
            made[kind] = pipeline.filter(|_| on);
        }
        made
    }

    /// What of `ground` can draw now: the passes whose pipelines compiled.
    /// A node draws its buildings without the rest until they have.
    pub(crate) fn ground_ready(&self, gpu: &mut dyn Gpu, ground: Ground) -> Ground {
        let [shadows, occlusion] = self
            .ground_made(ground)
            .map(|p| p.is_some_and(|p| gpu.pipeline_ready(p.into())));
        Ground {
            shadows,
            occlusion,
            mask: ground.mask,
        }
    }

    /// The ground passes' pipelines `ground` draws with, made or not.
    pub(crate) fn ground_pipelines(&self, ground: Ground) -> Vec<PipelineId> {
        (self.ground_made(ground).into_iter().flatten())
            .map(PipelineId::from)
            .collect()
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
    pub(crate) fn pipelines(&self, shape: ElevationShape) -> Vec<PipelineId> {
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
    /// replaying its draws, recorded once: its buildings, then what
    /// `ground` darkens on the ground around them. Drawn first, the
    /// ground passes shaded every shadow's whole area, most of it then
    /// covered: three times the layer's GPU time over Midtown at pitch 60.
    /// After, the depth test drops what the buildings cover before it
    /// shades, and the max blend can't change a building's opaque pixel
    /// where a shadow on terrain, pulled nearer, still passes.
    pub(crate) fn prepare(
        &self,
        gpu: &mut dyn Gpu,
        node: &mut NodeExtrusion,
        scene: BindGroupId,
        buffers: &SceneBuffers,
        (streams, shape, ground): (&Streams, ElevationShape, Ground),
        first: bool,
    ) -> RenderPass {
        let ground = self.ground_made(ground.recorded());
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
            // The buildings, each pipeline over its stream, then what
            // they darken on the ground, over both.
            let [w, c] = [&streams.whole, &streams.cut];
            let draws = [(whole, w), (cut, c), (planes, c)].into_iter();
            let ground = [c, w]
                .into_iter()
                .flat_map(|s| ground.iter().map(move |&g| (g, s)));
            streams.record(draws.chain(ground), &mut commands);
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

impl Extrusions {
    /// The ground mask for a 3D target of `size`, made or remade to fit,
    /// to sample: as much larger as its plane shrinks into it
    /// (`GROUND_MASK_SCALE`), at the current sample count.
    pub(crate) fn mask_target(&mut self, gpu: &mut dyn Gpu, size: [u32; 2]) -> TextureId {
        let max = gpu.caps().limits.max_texture_dimension_2d;
        let size = size.map(|s| ((s as f32 / GROUND_MASK_SCALE).ceil() as u32).clamp(1, max));
        let samples = self.samples;
        if let Some((target, made, made_samples)) = &self.mask_target
            && (*made, *made_samples) == (size, samples)
        {
            return target.texture(0);
        }
        self.mask_target.take().map(|(t, ..)| t).release(gpu);
        let format = [TextureFormat::R8Unorm];
        let target = Target::new(gpu, "ground mask", &format, size, samples);
        let texture = target.texture(0);
        self.mask_target = Some((target, size, samples));
        texture
    }

    /// The pass drawing what a node darkens on the ground into the mask,
    /// as `ground` (made, into the mask) says: with the node's `draw`, as
    /// its buildings draw.
    pub(crate) fn mask_pass(
        &self,
        scene: BindGroupId,
        buffers: &SceneBuffers,
        (streams, ground): (&Streams, Ground),
    ) -> Option<RenderPass> {
        let (target, ..) = self.mask_target.as_ref()?;
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
        let [w, c] = [&streams.whole, &streams.cut];
        let ground = self.ground_made(ground);
        let ground = [c, w]
            .into_iter()
            .flat_map(|s| ground.iter().map(move |&g| (g, s)));
        streams.record(ground, &mut commands);
        Some(RenderPass {
            label: "ground mask",
            colors: vec![target.attachment(0, LoadOp::Clear([0.0; 4]))],
            depth: None,
            commands,
        })
    }
}

/// `project.wgsl`'s: the ground mask's clip x and y shrink by it.
const GROUND_MASK_SCALE: f32 = 0.8;

impl crate::release::Release for Extrusions {
    fn release(self, gpu: &mut dyn Gpu) {
        let pipelines: Vec<Pipelines> = self.pipelines.into_iter().map(|(_, p)| p).collect();
        let mask = self.mask_target.map(|(t, ..)| t);
        (pipelines, self.ground, self.mask, self.depth, mask).release(gpu);
    }
}
