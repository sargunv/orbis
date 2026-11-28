//! Extrusion layers draw into a target of their own, resolve it, then
//! composite it at their layer opacity once. The shared depth attachment
//! keeps occlusion between 3D layers and the terrain, whose depth the
//! first draws into it.
use crate::TargetFormat;
use crate::awaits::Lacks;
use crate::offscreen::{Apart, Composites, Mode, Offscreen, Recorded};
use crate::paint::ExtrusionShape;
use crate::pipeline::{Cache, Draw, depth_test};
use crate::release::Release;
use crate::scene::SceneBuffers;
use orb_hal::*;
use orb_shaders::{EXTRUSION_CUT_SHADER, EXTRUSION_PLANES_SHADER, EXTRUSION_SHADER, Shader};

#[derive(Debug)]
pub(crate) struct Extrusions {
    /// The device has clip distances: the whole pipeline draws the cut
    /// stream too, its buildings of a placement with one clip rectangle,
    /// in one compile fewer than a pipeline of their own.
    planes: bool,
    layouts: [BindGroupLayoutId; 2],
    pub(crate) samples: u32,
    /// The targets' format (RGBA8, whatever the map's).
    format: TextureFormat,
    /// The depth's format with a stencil
    /// ([`depth_format`](crate::pipeline::depth_format)), which the map's
    /// passes take while terrain may draw in them, and which the depth the
    /// 3D layers share with them takes then too.
    stencil_format: TextureFormat,
    /// The depth 3D passes share, and its format.
    depth: Option<(TextureId, TextureFormat)>,
    size: [u32; 2],
    single_layer: bool,
}

/// An extrusion layer's node, drawn apart.
pub(crate) type NodeExtrusion = Apart;

/// An extrusion node's two streams, each with the node group it binds and
/// its draws, which bind their blocks' scene groups: the buildings its
/// placements may draw whole, which the whole pipeline draws, and its cut
/// stream ([`Scene::second_row`](crate::scene::Scene::second_row)), what
/// they draw cut, which the cut pipelines draw. Each draws its tiles in
/// their order ([`Scene::segments`](crate::scene::Scene::segments)).
#[derive(Clone, Debug)]
pub(crate) struct Streams {
    pub(crate) whole: (BindGroupId, Vec<RenderCommand>),
    pub(crate) cut: (BindGroupId, Vec<RenderCommand>),
}

impl Streams {
    /// Records each of `draws`, a pipeline over a stream, binding each
    /// stream's group as it changes.
    fn record<'a>(
        draws: impl Iterator<Item = (RenderPipelineId, &'a (BindGroupId, Vec<RenderCommand>))>,
        commands: &mut Vec<RenderCommand>,
    ) {
        let mut bound = None;
        for (pipeline, (group, draws)) in draws {
            if draws.is_empty() {
                continue;
            }
            if bound != Some(*group) {
                bound = Some(*group);
                commands.push(RenderCommand::SetBindGroup {
                    index: 1,
                    group: *group,
                });
            }
            commands.push(RenderCommand::SetPipeline(pipeline));
            commands.extend(draws.iter().cloned());
        }
    }
}

impl Extrusions {
    pub(crate) fn new(
        caps: &Caps,
        (scene, node): (BindGroupLayoutId, BindGroupLayoutId),
        (format, stencil_format): (TextureFormat, TextureFormat),
    ) -> Self {
        Self {
            planes: caps.clip_distances,
            layouts: [scene, node],
            samples: 1,
            format,
            stencil_format,
            depth: None,
            size: [0; 2],
            single_layer: false,
        }
    }

    /// The whole and cut draws of `shape` over a depth with a stencil or
    /// without (`stencil`). With clip distances, the whole pipeline's
    /// vertex stage also clips the cut buildings of placements with one
    /// clip rectangle by its planes, and the cut pipeline keeps those of
    /// placements with several.
    fn draws<'a>(
        &'a self,
        shape: ExtrusionShape,
        stencil: bool,
        constants: &'a [Vec<(&'static str, f64)>; 2],
    ) -> [Draw<'a>; 2] {
        let (whole, cut) = (&EXTRUSION_SHADER, &EXTRUSION_CUT_SHADER);
        let (whole_vertex, cut_vertex) = match self.planes {
            true => (
                (&EXTRUSION_PLANES_SHADER, "vs_extrusion_planes"),
                "vs_extrusion_cut_split",
            ),
            false => ((whole, whole.vertex), cut.vertex),
        };
        // Buildings shaded by their colour alone pass nothing else.
        let whole_stages = match (shape.plain, self.planes) {
            (true, true) => (
                (whole_vertex.0, "vs_extrusion_planes_plain"),
                "fs_extrusion_plain",
            ),
            (true, false) => ((whole, "vs_extrusion_plain"), "fs_extrusion_plain"),
            (false, _) => (whole_vertex, whole.fragment),
        };
        let stages: [((&Shader, _), (&Shader, _)); 2] = [
            (whole_stages.0, (whole, whole_stages.1)),
            ((cut, cut_vertex), (cut, cut.fragment)),
        ];
        stages.map(|(vertex, fragment)| Draw {
            vertex,
            fragment,
            constants: [&constants[0], &constants[1]],
            depth: depth_test(self.depth_format(stencil)),
            cull_back: true,
            state: !shape.stateless,
            patterns: shape.pattern,
            ..Draw::new(vertex.0, &self.layouts, self.format(), None)
        })
    }

    /// Makes the whole and cut pipelines of `shape` over a depth with a
    /// stencil or without, in each variant of `variants` asked for.
    pub(crate) fn make(
        &self,
        (gpu, cache): (&mut dyn Gpu, &mut Cache),
        (shape, stencil): (ExtrusionShape, bool),
        variants: [bool; 2],
    ) {
        let constants = shape.constants();
        for draw in self.draws(shape, stencil, &constants) {
            cache.make_variants(gpu, variants, draw);
        }
    }

    /// What a frame draws `shape` with over a depth with a stencil or
    /// without, whole and cut, `full` as in
    /// [`Variants::drawn`](crate::pipeline::Variants::drawn): nothing
    /// until both compiled, `lacks` gaining what it waits for.
    pub(crate) fn drawn(
        &self,
        (gpu, cache): (&mut dyn Gpu, &Cache),
        (shape, stencil): (ExtrusionShape, bool),
        full: bool,
        lacks: &mut Lacks,
    ) -> Option<[RenderPipelineId; 2]> {
        let constants = shape.constants();
        let [whole, cut] = self
            .draws(shape, stencil, &constants)
            .map(|draw| cache.variants(draw).drawn(gpu, full, lacks));
        Some([whole?, cut?])
    }

    /// What a frame waits for to draw `shape` over a depth with a stencil
    /// or without (`full` as in
    /// [`Variants::wanted`](crate::pipeline::Variants::wanted)).
    pub(crate) fn wanted(
        &self,
        cache: &Cache,
        (shape, stencil): (ExtrusionShape, bool),
        full: bool,
    ) -> impl Iterator<Item = PipelineId> {
        let constants = shape.constants();
        let draws = self.draws(shape, stencil, &constants);
        let wanted = draws.map(|draw| cache.variants(draw).wanted(full));
        wanted.into_iter().flatten().map(PipelineId::from)
    }

    /// Draws at four samples when `antialias`, else at one; whether that
    /// changed, so the pipelines are asked for again and the targets made
    /// anew.
    pub(crate) fn set_antialias(&mut self, antialias: bool) -> bool {
        let samples = if antialias { 4 } else { 1 };
        if samples == self.samples {
            return false;
        }
        self.samples = samples;
        self.size = [0; 2];
        true
    }

    /// What their passes draw into: their targets' format at their sample
    /// count.
    pub(crate) fn format(&self) -> TargetFormat {
        TargetFormat {
            color: self.format,
            sample_count: self.samples,
        }
    }

    /// The depth's format with a stencil, as the map's passes have while
    /// terrain may draw in them, or without: a stencil's memory, and the
    /// 24-bit depth where the device has no `depth32float-stencil8`, only
    /// for maps that need it.
    fn depth_format(&self, stencil: bool) -> TextureFormat {
        match stencil {
            true => self.stencil_format,
            false => TextureFormat::Depth32Float,
        }
    }

    /// The depth 3D passes share, once a resize made it, and how its
    /// passes store it: a single pass's is transient.
    pub(crate) fn depth(&self) -> Option<(TextureId, StoreOp)> {
        let store = if self.single_layer {
            StoreOp::Discard
        } else {
            StoreOp::Store
        };
        self.depth.map(|(d, _)| (d, store))
    }

    /// Sizes the depth `count` passes share, with a stencil or without:
    /// the extrusion layers', and the map's where they split at them;
    /// whether it was made anew.
    pub(crate) fn resize(
        &mut self,
        gpu: &mut dyn Gpu,
        size: [u32; 2],
        (count, stencil): (usize, bool),
    ) -> bool {
        let format = self.depth_format(stencil);
        if count == 0
            || (self.size == size
                && self.single_layer == (count == 1)
                && self.depth.is_some_and(|(_, made)| made == format))
        {
            return false;
        }
        self.depth.take().map(|(depth, _)| depth).release(gpu);
        let depth = gpu.create_texture(&TextureDesc {
            label: "3d depth",
            width: size[0],
            height: size[1],
            array_layers: 1,
            dimension: TextureViewDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT
                | if count == 1 {
                    TextureUsages::TRANSIENT_ATTACHMENT
                } else {
                    TextureUsages::empty()
                },
            sample_count: self.samples,
            mip_levels: 1,
        });
        self.depth = Some((depth, format));
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
        Offscreen::fit(
            &mut node.offscreen,
            gpu,
            composites,
            ("extrusion", self.format),
            (size, self.samples),
            (Mode::Over, None),
            node.opacity,
        )
    }

    /// What a node's bundle drawing with `pipelines` (whole and cut)
    /// records over, `draws` the renderer's draws' and the extrusions'
    /// segments' revisions.
    pub(crate) fn recorded(
        &self,
        [whole, cut]: [RenderPipelineId; 2],
        draws: [u64; 2],
    ) -> Recorded {
        Recorded {
            draws,
            pipelines: [Some(whole), Some(cut)],
            target: (self.depth.map(|(_, format)| format), self.samples),
        }
    }

    /// An extrusion node's pass into its target over the shared depth,
    /// which the `first` clears, replaying its draws after `before` (the
    /// terrain's depth): its buildings (`streams`, which a bundle recorded
    /// over `recorded` needs no more), each stream drawn with its pipeline.
    pub(crate) fn prepare(
        &self,
        gpu: &mut dyn Gpu,
        node: &mut NodeExtrusion,
        buffers: &SceneBuffers,
        (streams, recorded): (Option<Streams>, Recorded),
        (first, mut before): (bool, Vec<RenderCommand>),
    ) -> RenderPass {
        let (depth, format) = self.depth.expect("sized");
        let [whole, cut] = recorded
            .pipelines
            .map(|p| p.expect("an extrusion draws whole and cut"));
        let bundle = node.bundle(gpu, recorded, |gpu| {
            let streams = streams.expect("a bundle to record has its draws");
            let mut commands = vec![RenderCommand::SetIndexBuffer {
                buffer: buffers.compacted,
                offset: 0,
            }];
            // The buildings, each pipeline over its stream; with clip
            // distances the whole pipeline draws the cut stream's
            // buildings it clips by planes too.
            let [w, c] = [&streams.whole, &streams.cut];
            let planes = self.planes.then_some((whole, c));
            let draws = [(whole, w), (cut, c)].into_iter().chain(planes);
            Streams::record(draws, &mut commands);
            gpu.create_render_bundle(
                &RenderBundleDesc {
                    label: "3d layer",
                    color_formats: &[self.format],
                    depth_format: Some(format),
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
                    .attachment(LoadOp::Clear([0.0; 4])),
            ],
            depth: Some(DepthAttachment {
                store: self.depth().expect("sized").1,
                target: depth,
                load: if first {
                    LoadOp::Clear(0.0)
                } else {
                    LoadOp::Load
                },
            }),
            commands: {
                before.push(RenderCommand::ExecuteBundles(vec![bundle]));
                before
            },
        }
    }
}

impl crate::release::Release for Extrusions {
    fn release(self, gpu: &mut dyn Gpu) {
        self.depth.map(|(depth, _)| depth).release(gpu);
    }
}
