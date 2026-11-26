//! Symbol layers' draws. Where the device has compute, a dispatch a
//! frame places every quad of the symbol nodes' visible chunks
//! (`symbol_place_compute.wgsl`, a workgroup for each candidate row of
//! theirs; one for point labels and one for line labels), writing where the quad lands, turned and scaled, and its paint
//! into two texels of the placement target; each node then draws its
//! quads' corners there (`symbol_draw.wgsl`), in one to three passes over
//! its one list (halos drawn apart). Placed in every corner of every pass
//! instead, quads ran their label's whole placement four to twelve times a
//! frame: most of the symbols' time on the moto g (#192).
//!
//! Without compute (WebGL 2), each corner still places its quad itself
//! (`symbol_place_inline.wgsl`). Points drawn into the target's layers as
//! colour targets placed nothing on an Adreno 619 once a frame or two had
//! drawn (its uniforms read as another buffer's), where Chrome on a Mac and
//! ANGLE drew them right.

use orb_hal::*;
#[cfg(cpu_driven)]
use orb_shaders::SYMBOL_INLINE_SHADERS;
use orb_shaders::{SYMBOL_PLACE_SHADERS, SYMBOL_SHADER, Shader};

use crate::awaits::Lacks;
use crate::pipeline::{Attachments, Cache, Draw, Variants};
use crate::release::Release;
use crate::scene::{Backing, PAGE_CHUNKS, PLACED_WIDTH, Tables};

/// What a symbol pass draws of a layer's icons or of its text
/// (`symbol_draw.wgsl`'s `PASS_*`): halo and fill, the halo, the fill, or
/// nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SymbolPass {
    Whole = 0,
    Halo = 1,
    Fill = 2,
    Skip = 3,
}

/// The symbol passes there are, each `[icons, text]`. A symbol layer draws
/// its icons, then its text, each halo and fill together or, where glyphs
/// overlap (`separate_halo`), every halo, then every fill: one or more of
/// these over the one list ([`symbol_passes`]).
const SYMBOL_PASSES: [[SymbolPass; 2]; 6] = {
    use SymbolPass::*;
    [
        [Whole, Whole],
        [Halo, Skip],
        [Fill, Whole],
        [Whole, Halo],
        [Skip, Fill],
        [Fill, Halo],
    ]
};

/// The passes of [`SYMBOL_PASSES`] a symbol layer draws, in order, by
/// whether its icons and its text draw their halos apart.
fn symbol_passes(icons_apart: bool, text_apart: bool) -> &'static [usize] {
    match (icons_apart, text_apart) {
        (false, false) => &[0],
        (true, false) => &[1, 2],
        (false, true) => &[3, 4],
        (true, true) => &[1, 5, 4],
    }
}

/// A symbol quad's two triangles over its corners (tl, tr, bl, br;
/// `symbol_draw.wgsl`'s `vs_symbol`). Drawn indexed, each corner runs the
/// vertex stage once rather than once per triangle.
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 1, 2, 3];

/// The placement target's texels, two a quad (`symbol.wgsl`'s layout).
const PLACED_FORMAT: TextureFormat = TextureFormat::Rgba32Uint;

/// The most workgroups a dispatch has along x (WebGPU's default limit),
/// one for each candidate row of the symbol nodes: far more than a view
/// lays out.
const MAX_WORKGROUPS: u32 = 65535;

/// What the placement dispatch reads besides the scene's tables, the node
/// uniforms and the cull's draw arguments, needs of their buffers' usage
/// on `caps`'s device.
pub(crate) fn read_usage(caps: &Caps) -> BufferUsages {
    if caps.compute {
        BufferUsages::STORAGE
    } else {
        BufferUsages::empty()
    }
}

/// The placement dispatch's kind ([`SYMBOL_PLACE_SHADERS`] by `kind / 2`,
/// with property rows `kind % 2`): point labels' alone, or with line
/// labels' where any symbol layer places along lines; `None` without
/// symbol layers. One dispatch places every layer's quads.
pub(crate) type Placings = Option<usize>;

/// How the symbol draws find their quads placed.
#[derive(Debug)]
enum Placing {
    /// By a dispatch before them.
    Dispatched(Box<Dispatch>),
    /// By each corner itself: the draw's pipelines by the layer's kind
    /// ([`Symbols::make`]), into the map's passes as they are now.
    #[cfg(cpu_driven)]
    Inline([Variants; 4]),
}

/// The placement dispatch, its target and the draws that read it.
#[derive(Debug)]
struct Dispatch {
    /// The draw's pipelines, without and with property rows, into the
    /// map's passes as they are now (the renderer's [`Cache`] keeps those
    /// for the others), each made for the first symbol layer that draws
    /// with it.
    draws: [Variants; 2],
    /// The dispatch's pipelines, by their [`Placings`] kind; its layout,
    /// and by scene block slot its group with the scene's entries it was
    /// made with.
    pipelines: [Variants<ComputePipelineId>; 4],
    layout: BindGroupLayoutId,
    groups: Vec<Option<(BindGroupId, Vec<BindGroupEntry>)>>,
    /// The placement target: two layers of [`PLACED_WIDTH`] × `rows`
    /// texels.
    target: TextureId,
    rows: u32,
}

/// The symbol pipelines, the placements and the draws' passes.
#[derive(Debug)]
pub(crate) struct Symbols {
    placing: Placing,
    /// The draw's group 2: a pass's uniform, the placement target where a
    /// dispatch writes it, and the glyph pages' chunks.
    pass_layout: BindGroupLayoutId,
    pass_buffer: BufferId,
    pass_stride: u64,
    /// One group per [`SYMBOL_PASSES`] entry.
    passes: [BindGroupId; SYMBOL_PASSES.len()],
    /// The glyph pages' chunks the groups bind.
    glyphs: [TextureId; PAGE_CHUNKS],
    /// [`QUAD_INDICES`], the index buffer the draws take.
    quad_indices: BufferId,
    /// The draws' groups' layouts: the scene's, the node's and
    /// [`pass_layout`](Self::pass_layout).
    layouts: [BindGroupLayoutId; 3],
    /// The map's passes.
    target: Attachments,
}

impl Symbols {
    /// The symbols of a renderer whose tables are on `backing`, drawing
    /// glyphs from the chunks of `glyphs` into the map's passes, `target`,
    /// with the scene's and the node's groups of `(scene, node)`.
    pub(crate) fn new(
        gpu: &mut dyn Gpu,
        backing: Backing,
        glyphs: [TextureId; PAGE_CHUNKS],
        (target, (scene, node)): (Attachments, (BindGroupLayoutId, BindGroupLayoutId)),
    ) -> Self {
        use BindGroupLayoutEntry as E;
        let placing = Placing::new(gpu, backing);
        let vf = ShaderStages::VERTEX | ShaderStages::FRAGMENT;
        let d2_array = TextureViewDimension::D2Array;
        let mut entries = vec![E::uniform(0, vf)];
        if placing.target().is_some() {
            entries.push(E::uint_texture(1, ShaderStages::VERTEX, d2_array));
        }
        entries.extend(
            (0..PAGE_CHUNKS as u32)
                .map(|k| E::texture(GLYPH_BINDING + k, ShaderStages::FRAGMENT, true, d2_array)),
        );
        let pass_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "symbol pass",
            entries: &entries,
        });
        let pass_stride = u64::from(gpu.caps().limits.min_uniform_buffer_offset_alignment).max(16);
        let pass_buffer = gpu.create_buffer(&BufferDesc {
            label: "symbol passes",
            size: SYMBOL_PASSES.len() as u64 * pass_stride,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        for (i, pass) in SYMBOL_PASSES.iter().enumerate() {
            let [icons, text] = pass.map(|p| p as u32);
            let at = i as u64 * pass_stride;
            gpu.write_buffer(pass_buffer, at, bytemuck::cast_slice(&[icons, text, 0, 0]));
        }
        let quad_indices = gpu.create_buffer(&BufferDesc {
            label: "quad indices",
            size: std::mem::size_of_val(&QUAD_INDICES) as u64,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
        });
        gpu.write_buffer(quad_indices, 0, bytemuck::cast_slice(&QUAD_INDICES));
        let pass = (pass_layout, pass_buffer, pass_stride);
        let passes = pass_groups(gpu, pass, placing.target(), glyphs);
        Self {
            placing,
            pass_layout,
            pass_buffer,
            pass_stride,
            passes,
            glyphs,
            quad_indices,
            layouts: [scene, node, pass_layout],
            target,
        }
    }

    /// Draws glyphs from the chunks of `glyphs` from now on: whether the
    /// draws' bindings changed, which the bundles record.
    pub(crate) fn set_glyphs(
        &mut self,
        gpu: &mut dyn Gpu,
        glyphs: [TextureId; PAGE_CHUNKS],
    ) -> bool {
        if glyphs == self.glyphs {
            return false;
        }
        self.glyphs = glyphs;
        let pass = (self.pass_layout, self.pass_buffer, self.pass_stride);
        let passes = pass_groups(gpu, pass, self.placing.target(), glyphs);
        std::mem::replace(&mut self.passes, passes).release(gpu);
        true
    }

    /// Gives the placement target `rows` rows (the scene's
    /// `placed_rows`); whether the draws' bindings changed, which the
    /// bundles record.
    /// The placement target's rows, 0 before it has one.
    pub(crate) fn rows(&self) -> u32 {
        self.placing.dispatch().map_or(0, |dispatch| dispatch.rows)
    }

    pub(crate) fn fit(&mut self, gpu: &mut dyn Gpu, rows: u32) -> bool {
        let Some(dispatch) = self.placing.dispatch_mut() else {
            return false;
        };
        if rows == dispatch.rows {
            return false;
        }
        let target = placed_target(gpu, rows);
        let old = std::mem::replace(&mut dispatch.target, target);
        dispatch.rows = rows;
        let groups: Vec<BindGroupId> = (dispatch.groups.drain(..).flatten())
            .map(|(group, _)| group)
            .collect();
        let pass = (self.pass_layout, self.pass_buffer, self.pass_stride);
        let passes = pass_groups(gpu, pass, Some(target), self.glyphs);
        (old, groups, std::mem::replace(&mut self.passes, passes)).release(gpu);
        true
    }

    /// Binds the scene for the placement dispatch, `scene` each block
    /// slot's entries (frame, tables, terrain, node uniforms and draw
    /// arguments, as [`compute_layout`] has them; `None` for a free slot),
    /// where they changed.
    pub(crate) fn bind(&mut self, gpu: &mut dyn Gpu, scene: Vec<Option<Vec<BindGroupEntry>>>) {
        let Some(dispatch) = self.placing.dispatch_mut() else {
            return;
        };
        let mut scene = scene;
        let slots = scene.len().max(dispatch.groups.len());
        scene.resize(slots, None);
        dispatch.groups.resize_with(slots, || None);
        for (slot, entries) in scene.into_iter().enumerate() {
            let bound = &mut dispatch.groups[slot];
            if bound.as_ref().map(|(_, e)| e) == entries.as_ref() {
                continue;
            }
            let made = entries.map(|scene| {
                let mut entries = scene.clone();
                let d2_array = TextureViewDimension::D2Array;
                entries.push(BindGroupEntry::texture(19, dispatch.target, d2_array));
                let made = gpu.create_bind_group(&BindGroupDesc {
                    label: "symbol placements",
                    layout: dispatch.layout,
                    entries: &entries,
                });
                (made, scene)
            });
            std::mem::replace(bound, made)
                .map(|(old, _)| old)
                .release(gpu);
        }
    }

    /// The draws' pipelines, by the slot [`draw`](Self::draw) gives.
    fn draws(&self) -> &[Variants] {
        match &self.placing {
            Placing::Dispatched(dispatch) => &dispatch.draws,
            #[cfg(cpu_driven)]
            Placing::Inline(draws) => draws,
        }
    }

    fn draws_mut(&mut self) -> &mut [Variants] {
        match &mut self.placing {
            Placing::Dispatched(dispatch) => &mut dispatch.draws,
            #[cfg(cpu_driven)]
            Placing::Inline(draws) => draws,
        }
    }

    /// The draw of a symbol layer of `kind` (as in [`make`](Self::make)),
    /// and its slot among the draws' pipelines: one draw for every kind
    /// after a dispatch placed them, with property rows and without, but
    /// one for each where every corner places its quad itself.
    fn draw(&self, kind: usize) -> (Draw<'_>, usize) {
        let (shader, slot): (&Shader, _) = match &self.placing {
            Placing::Dispatched(_) => (&SYMBOL_SHADER, kind % 2),
            #[cfg(cpu_driven)]
            Placing::Inline(_) => (&SYMBOL_INLINE_SHADERS[kind / 2], kind),
        };
        let blend = Some(BlendState::PremultipliedAlpha);
        let draw = Draw {
            rows: kind % 2 == 1,
            ..Draw::new(shader, &self.layouts, self.target, blend)
        };
        (draw, slot)
    }

    /// Draws into the map's passes as `target` from now on, with or
    /// without their depth: with the pipelines made for them before, if
    /// any.
    pub(crate) fn retarget(&mut self, cache: &Cache, target: Attachments) {
        self.target = target;
        // Each slot's own index is a kind that draws with it.
        let found: Vec<Variants> = (0..self.draws().len())
            .map(|slot| Variants::find(cache, self.draw(slot).0))
            .collect();
        self.draws_mut().copy_from_slice(&found);
    }

    /// Makes the pipelines a symbol layer of `kind` draws with (along
    /// lines `kind / 2`, with property rows `kind % 2`) where the symbol
    /// layers place with `placings`, each variant of `variants` asked
    /// for.
    pub(crate) fn make(
        &mut self,
        (gpu, cache): (&mut dyn Gpu, &mut Cache),
        (kind, placings): (usize, Placings),
        variants: [bool; 2],
    ) {
        let (draw, slot) = self.draw(kind);
        let mut made = self.draws()[slot];
        made.make((gpu, cache), variants, draw);
        self.draws_mut()[slot] = made;
        match &mut self.placing {
            Placing::Dispatched(dispatch) => {
                let layouts = [dispatch.layout];
                if let Some(placing) = placings {
                    let shader = &SYMBOL_PLACE_SHADERS[placing / 2];
                    dispatch.pipelines[placing].make_with(variants, |full| {
                        let mut constants = Vec::new();
                        if !full && shader.takes(shader.compute, orb_shaders::PLANE) {
                            constants.push((orb_shaders::PLANE, 1.0));
                        }
                        let rows = orb_shaders::FEATURE_ROWS;
                        if placing % 2 == 0 && shader.takes(shader.compute, rows) {
                            constants.push((rows, 0.0));
                        }
                        crate::pipeline::compute(gpu, shader, &layouts, &constants)
                    });
                }
            }
            #[cfg(cpu_driven)]
            Placing::Inline(_) => {}
        }
    }

    /// The pipelines a frame waits for (`full` as in
    /// [`Variants::wanted`]), `kind` and `placings` as in
    /// [`make`](Self::make).
    pub(crate) fn wanted(
        &self,
        (kind, placings): (usize, Placings),
        full: bool,
    ) -> impl Iterator<Item = PipelineId> {
        let [place, draw] = match &self.placing {
            Placing::Dispatched(dispatch) => [
                placings.and_then(|p| dispatch.pipelines[p].wanted(full).map(Into::into)),
                dispatch.draws[kind % 2].wanted(full).map(Into::into),
            ],
            #[cfg(cpu_driven)]
            Placing::Inline(draws) => [None, draws[kind].wanted(full).map(Into::into)],
        };
        [place, draw].into_iter().flatten()
    }

    /// Node `i`'s draws into `commands`, a symbol layer's icon node, `kind`
    /// and `placings` as in [`make`](Self::make), whose icons and text draw
    /// their halos apart as `apart` says, `quads` its draw as indexed
    /// arguments, and whether it drew: `lacks` gains what it waits for,
    /// the variant a stand-in stands for ([`Variants::drawn`]) too.
    /// Nothing draws while a pipeline compiles (each placement's too, as
    /// [`pass`](Self::pass) dispatches only once all have).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commands(
        &self,
        gpu: &mut dyn Gpu,
        (kind, placings, full): (usize, Placings, bool),
        (scene, group): (BindGroupId, BindGroupId),
        apart: [bool; 2],
        quads: RenderCommand,
        commands: &mut Vec<RenderCommand>,
        lacks: &mut Lacks,
    ) -> bool {
        let draw = match &self.placing {
            Placing::Dispatched(dispatch) => {
                if dispatch.placements(gpu, placings, full, lacks).is_none() {
                    return false;
                }
                dispatch.draws[kind % 2].drawn(gpu, full, lacks)
            }
            #[cfg(cpu_driven)]
            Placing::Inline(draws) => draws[kind].drawn(gpu, full, lacks),
        };
        let Some(draw) = draw else {
            return false;
        };
        commands.extend([
            RenderCommand::SetPipeline(draw),
            RenderCommand::SetBindGroup {
                index: 0,
                group: scene,
            },
            RenderCommand::SetBindGroup { index: 1, group },
            RenderCommand::SetIndexBuffer {
                buffer: self.quad_indices,
                offset: 0,
            },
        ]);
        for &pass in symbol_passes(apart[0], apart[1]) {
            commands.extend([
                RenderCommand::SetBindGroup {
                    index: 2,
                    group: self.passes[pass],
                },
                quads.clone(),
            ]);
        }
        true
    }

    /// The frame's placement dispatches, after the cull, which wrote the
    /// draw arguments, and the terrain's depth, which hides labels; before
    /// any pass that draws them. One for each block, by slot, of the
    /// placement target's quads its symbol nodes take, `blocks`
    /// (`placings` and `full` as in [`commands`](Self::commands)).
    pub(crate) fn pass(
        &self,
        gpu: &mut dyn Gpu,
        (placings, full): (Placings, bool),
        blocks: &[(u32, std::ops::Range<u32>)],
    ) -> Option<Pass> {
        let dispatch = self.placing.dispatch()?;
        let pipeline = dispatch.placements(gpu, placings, full, &mut Lacks::default())?;
        let mut commands = Vec::new();
        for (slot, quads) in blocks {
            let rows =
                ((quads.end - quads.start) / orb_layout::MAX_CHUNK_INSTANCES).min(MAX_WORKGROUPS);
            let Some(Some((group, _))) = dispatch.groups.get(*slot as usize) else {
                continue;
            };
            let Some(pipeline) = pipeline.filter(|_| rows > 0) else {
                continue;
            };
            if commands.is_empty() {
                commands.push(ComputeCommand::SetPipeline(pipeline));
            }
            commands.extend([
                ComputeCommand::SetBindGroup {
                    index: 0,
                    group: *group,
                },
                ComputeCommand::Dispatch {
                    x: rows,
                    y: 1,
                    z: 1,
                },
            ]);
        }
        if commands.is_empty() {
            return None;
        }
        Some(Pass::Compute(ComputePass {
            label: "symbol placements",
            commands,
        }))
    }
}

impl Dispatch {
    /// The pipeline of `placings` (`full` as in [`Variants::drawn`]), if
    /// any, once it has compiled; else `lacks` gains what it waits for.
    /// Dispatched every frame, a stand-in serves until the one it wants
    /// compiled, and records nothing again.
    fn placements(
        &self,
        gpu: &mut dyn Gpu,
        placings: Placings,
        full: bool,
        lacks: &mut Lacks,
    ) -> Option<Option<ComputePipelineId>> {
        let Some(placing) = placings else {
            return Some(None);
        };
        let mut wants = Lacks::default();
        let drawn = self.pipelines[placing].drawn(gpu, full, &mut wants);
        if drawn.is_none() {
            lacks.extend(&wants);
        }
        drawn.map(Some)
    }
}

impl Placing {
    fn new(gpu: &mut dyn Gpu, backing: Backing) -> Self {
        // Only WebGL 2's build (`cpu-driven`) meets a device without compute.
        #[cfg(cpu_driven)]
        if !gpu.caps().compute {
            return Self::Inline(Default::default());
        }
        Self::Dispatched(Box::new(Dispatch {
            draws: Default::default(),
            pipelines: Default::default(),
            layout: compute_layout(gpu, backing),
            groups: Vec::new(),
            target: placed_target(gpu, 1),
            rows: 1,
        }))
    }

    /// The placement dispatch, where there is one.
    fn dispatch(&self) -> Option<&Dispatch> {
        match self {
            Self::Dispatched(dispatch) => Some(dispatch),
            #[cfg(cpu_driven)]
            Self::Inline(_) => None,
        }
    }

    fn dispatch_mut(&mut self) -> Option<&mut Dispatch> {
        match self {
            Self::Dispatched(dispatch) => Some(dispatch),
            #[cfg(cpu_driven)]
            Self::Inline(_) => None,
        }
    }

    /// The placement target, where a dispatch writes one.
    fn target(&self) -> Option<TextureId> {
        self.dispatch().map(|dispatch| dispatch.target)
    }
}

/// A placement target of `rows` rows.
fn placed_target(gpu: &mut dyn Gpu, rows: u32) -> TextureId {
    gpu.create_texture(&TextureDesc {
        label: "symbol placements",
        width: PLACED_WIDTH,
        height: rows,
        array_layers: 2,
        dimension: TextureViewDimension::D2Array,
        format: PLACED_FORMAT,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        sample_count: 1,
        mip_levels: 1,
    })
}

/// The placement dispatch's group 0 (`symbol_place_compute.wgsl`): the
/// frame, the tables (the indices for the chunks' rows they hold), the
/// terrain's DEM and depth, every node's uniform, the cull's draw
/// arguments and the target: eight storage buffers, WebGPU's least.
fn compute_layout(gpu: &mut dyn Gpu, backing: Backing) -> BindGroupLayoutId {
    use BindGroupLayoutEntry as E;
    let cs = ShaderStages::COMPUTE;
    let tables = (Tables::layout_entries(backing).into_iter()).map(|e| BindGroupLayoutEntry {
        visibility: cs,
        ..e
    });
    let mut entries = vec![E::uniform(0, cs)];
    entries.extend(tables);
    entries.extend([
        E::texture(14, cs, false, TextureViewDimension::D2Array),
        E::texture(15, cs, false, TextureViewDimension::D2),
        E::storage(17, cs),
        E::storage(18, cs),
        BindGroupLayoutEntry {
            binding: 19,
            visibility: cs,
            ty: BindingType::StorageTexture {
                format: PLACED_FORMAT,
                dimension: TextureViewDimension::D2Array,
            },
        },
    ]);
    gpu.create_bind_group_layout(&BindGroupLayoutDesc {
        label: "symbol placements",
        entries: &entries,
    })
}

/// The glyph pages' first chunk's binding in the draw's group 2, the
/// others' after it (`symbol_draw.wgsl`).
const GLYPH_BINDING: u32 = 2;

/// The draw's group for each of [`SYMBOL_PASSES`]: its uniform, at
/// `stride` apart in `buffer`, the placement target if there is one, and
/// the glyph pages' chunks.
fn pass_groups(
    gpu: &mut dyn Gpu,
    (layout, buffer, stride): (BindGroupLayoutId, BufferId, u64),
    target: Option<TextureId>,
    glyphs: [TextureId; PAGE_CHUNKS],
) -> [BindGroupId; SYMBOL_PASSES.len()] {
    let d2_array = TextureViewDimension::D2Array;
    std::array::from_fn(|i| {
        let mut entries = vec![BindGroupEntry::slice(0, buffer, i as u64 * stride, 16)];
        entries.extend(target.map(|t| BindGroupEntry::texture(1, t, d2_array)));
        entries.extend(
            (glyphs.iter().zip(GLYPH_BINDING..))
                .map(|(&texture, binding)| BindGroupEntry::texture(binding, texture, d2_array)),
        );
        gpu.create_bind_group(&BindGroupDesc {
            label: "symbol pass",
            layout,
            entries: &entries,
        })
    })
}

impl Release for Symbols {
    fn release(self, gpu: &mut dyn Gpu) {
        match self.placing {
            Placing::Dispatched(dispatch) => {
                let Dispatch {
                    pipelines,
                    layout,
                    groups,
                    target,
                    ..
                } = *dispatch;
                let groups: Vec<BindGroupId> =
                    groups.into_iter().flatten().map(|(g, _)| g).collect();
                (pipelines, (layout, groups, target)).release(gpu);
            }
            #[cfg(cpu_driven)]
            Placing::Inline(_) => {}
        }
        (self.pass_layout, self.pass_buffer).release(gpu);
        (self.passes, self.quad_indices).release(gpu);
    }
}
