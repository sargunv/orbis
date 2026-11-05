//! The Modern GPU-driven path (plan §8.3, §8.5).
//!
//! A frame has a cull and main pass: one cull/compact dispatch, a workgroup per render
//! node that keeps its visible `(chunk, tile placement)` candidates in draw
//! order and writes its indirect arguments, and the main pass, which
//! replays a render bundle of one indirect draw per render node. The bundle
//! is recorded again only when the node list changes or an image page
//! grows; tile arrival changes buffers, not commands. Frames in which DEM
//! tiles arrived first run the prepare pass (see the `image` module). Heatmap
//! nodes prepare density targets after culling and composite in layer order.
//!
//! Fill, line and extrusion nodes draw compacted indices, which the cull
//! dispatch keeps in each candidate's slot of one index buffer, with one
//! `drawIndexedIndirect` per node; point families and image tiles pull
//! their vertices (decision 0002).

#![forbid(unsafe_code)]

mod atmosphere;
#[cfg(feature = "compat")]
mod compat;
mod extrusion;
mod heatmap;
mod image;
mod layer_composite;
mod paint;
mod terrain;
mod tiles;
pub use atmosphere::{AtmospherePaint, AtmosphereView, GlobeSky};
pub use layer_composite::Blend;
pub use terrain::{
    DRAPE_FORMAT, DRAPE_SLOT_PX, DrapeLayout, TERRAIN_EXTENT, TERRAIN_GRID, TerrainDem,
    TerrainGlobe, TerrainPaint, TerrainTile,
};

pub use image::{DemPrepare, IMAGE_EXTENT, IMAGE_ROW_WORDS, ImageFade, PageId, PageKind};
pub use orb_scene::Pixels;
pub use orb_scene::{
    ARGS_WORDS, ClipRect, DemRef, FrameBlock, FrameGlobe, FrameLights, FrameLod, FrameTerrain,
    GLOBE_PATCH_WORDS, ImageAtlas, ImageBindings, ImageKey, ImageMetadata, ImageUpload, MeshData,
    MeshId, MeshSize, NodeCull, NodeDraw, OutOfSpace, PlacementTarget, RowGlobe, Scene,
    SceneCapacity, SceneStats, TileId, TileRow, TileWarp, WARP_PATCH_WORDS, Words,
};
pub use orb_scene::{DESKTOP_BUDGET, MOBILE_BUDGET, PoolMemory, SceneMemory};
pub use orb_scene::{
    GLYPH_PAGE, GLYPH_ROW_WORDS, GlyphAtlas, PLACED_WORDS, QUAD_WORDS, SYMBOL_HEADER_WORDS,
    SYMBOL_STATE_WORDS, SymbolMeshData, pack_image_rect, symbol_header,
};
pub use paint::{
    CIRCLE_ROW_WORDS, CirclePaint, ColorReliefPaint, Curve, EXTRUSION_ROW_WORDS, ExtrusionPaint,
    FILL_ROW_WORDS, FillPaint, HEATMAP_ROW_WORDS, HeatmapPaint, HillshadeMethod, HillshadePaint,
    LINE_ROW_WORDS, LinePaint, MAX_LIGHTS, NodePaint, PatternPaint, RasterPaint, SYMBOL_ROW_WORDS,
    StateLookup, SymbolPaint, SymbolSize, Translate, pack_color,
};
pub use tiles::{GpuTile, ImageBuild, ImageMesh, TileKey, UploadError, dem_prepares};

use orb_hal::*;
use orb_shaders::{
    CIRCLE_SHADER, CULL_SHADER, FILL_OUTLINE_SHADER, FILL_SHADER, LINE_SHADER, SYMBOL_SHADER,
    Shader,
};

/// The layout family a render node draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    FillExtrusion,
    Fill,
    /// A fill's antialiasing outline: 1 px line segments, drawn with fill
    /// paint (outline colour and opacity).
    FillOutline,
    Line,
    Circle,
    Heatmap,
    /// Image tiles of a `raster` source.
    Raster,
    /// Slopes of a `raster-dem` source's tiles.
    Hillshade,
    /// Elevations of a `raster-dem` source's tiles through a colour ramp.
    ColorRelief,
    /// A symbol layer's icons.
    SymbolIcon,
    /// A symbol layer's text.
    SymbolText,
}

impl Family {
    /// Whether the family draws image tiles from a page.
    pub fn is_image(self) -> bool {
        matches!(self, Self::Raster | Self::Hillshade | Self::ColorRelief)
    }

    /// Whether the family draws symbol quads.
    pub fn is_symbol(self) -> bool {
        matches!(self, Self::SymbolIcon | Self::SymbolText)
    }

    /// Whether terrain drapes the family's layers over its surface rather
    /// than drawing them in 3D or on screen (GL JS's render to texture).
    pub fn is_draped(self) -> bool {
        matches!(
            self,
            Self::Fill
                | Self::FillOutline
                | Self::Line
                | Self::Raster
                | Self::Hillshade
                | Self::ColorRelief
        )
    }

    /// How the family's nodes draw (decision 0002): meshes from compacted
    /// indices, instanced quads by vertex pulling.
    fn draw(self) -> NodeDraw {
        match self {
            Self::Fill | Self::FillOutline | Self::Line | Self::FillExtrusion => NodeDraw::Indexed,
            Self::Circle | Self::Heatmap | Self::SymbolIcon | Self::SymbolText => NodeDraw::Points,
            Self::Raster | Self::Hillshade | Self::ColorRelief => NodeDraw::Images,
        }
    }
}

/// Where node `index`'s draw arguments start in the scene's args buffer.
fn args_offset(index: u32) -> u64 {
    u64::from(index) * orb_scene::ARGS_WORDS as u64 * 4
}

/// The attachment a renderer draws into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetFormat {
    pub color: TextureFormat,
    pub sample_count: u32,
}

/// Counters for one frame (plan §16).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    pub dispatches: u32,
    /// Draws issued: one indirect draw per node, and prepare passes'.
    pub draws: u32,
    /// Whether the bundle was re-recorded this frame.
    pub bundle_recorded: bool,
    pub candidates: u32,
    /// The cull's candidates were laid out again: tiles or placements came
    /// or went, or nodes changed how they draw.
    pub candidates_rebuilt: bool,
    /// Objects the CPU wrote since the last frame (plan §16): tile
    /// placement rows and node uniforms. A camera-only frame rewrites only
    /// the placements in view.
    pub placements_written: u32,
    pub paints_written: u32,
    /// DEM tiles the prepare pass processed.
    pub prepared: u32,
    /// Pipelines the device is still compiling (plan §8.4).
    pub pipelines_pending: u32,
    /// Nodes not drawn because a pipeline they need is still compiling.
    pub nodes_waiting: u32,
    /// Something in the frame waited for a pipeline (nodes, terrain, the
    /// DEM prepare, the sky): the frame isn't complete.
    pub waited: bool,
}

/// A render node: its family, and the source an image node draws the
/// tiles of (by style key).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderNode {
    pub family: Family,
    pub source: Option<u32>,
}

#[derive(Debug)]
struct Node {
    family: Family,
    source: Option<u32>,
    group: BindGroupId,
    cull: NodeCull,
    /// The uniform words last written, to skip unchanged writes.
    written: Vec<u32>,
    /// What an image family's group binds.
    image: Option<image::NodeImage>,
    heatmap: Option<heatmap::NodeHeatmap>,
    extrusion: Option<extrusion::NodeExtrusion>,
    /// A symbol node draws its halos in a pass of their own.
    separate_halo: bool,
    /// On terrain, the drape stack a draped node draws into.
    stack: Option<u8>,
    /// An extrusion node's distance fade, as last written.
    fade: [f32; 2],
    /// Its style layer's key, layer opacity and blend mode
    /// (`*-layer-opacity`, `*-layer-blend`): below opacity 1 or blending
    /// other than normally, the layer composites once.
    composite: Option<(u32, f32, Blend)>,
}

#[derive(Debug)]
pub struct Renderer {
    target: TargetFormat,
    atmosphere: atmosphere::Atmosphere,
    scene: Scene,
    draw_group: BindGroupId,
    /// The cull dispatch's pipeline and group; none on Compat, which culls
    /// on the CPU.
    cull: Option<(ComputePipelineId, BindGroupId)>,
    draw_layout: BindGroupLayoutId,
    node_layout: BindGroupLayoutId,
    /// The vector families' pipelines, each made for the family's first
    /// node.
    paint: PaintPipelines,
    /// Symbols, made for the first symbol node.
    symbol_pipeline: Option<RenderPipelineId>,
    /// The symbol shader's pass (group 2): halo and fill together, halo
    /// only, fill only.
    symbol_passes: [BindGroupId; 3],
    symbol_pass_layout: BindGroupLayoutId,
    /// Nodes changed since pipelines were last made for them.
    pipelines_stale: bool,
    /// Pipelines the device had compiled when the bundles were recorded.
    made_seen: u32,
    /// Nodes the main bundle left out, waiting for a pipeline, and draped
    /// nodes the drape bundle left out.
    waiting: u32,
    drape_waiting: u32,
    node_uniforms: BufferId,
    node_stride: u64,
    nodes: Vec<Node>,
    tiles: tiles::Tiles,
    /// The main pass's bundles: one, or one more than the groups that
    /// blend other than normally (empty until recorded).
    bundles: Vec<RenderBundleId>,
    /// Node uniform writes since the last frame.
    paints_written: u32,
    images: image::Images,
    heatmaps: heatmap::Heatmaps,
    extrusions: extrusion::Extrusions,
    /// Layers drawn whole and composited at their layer opacity.
    composites: layer_composite::LayerComposites,
    /// Made for the first terrain.
    terrain: Option<terrain::Terrain>,
    /// The scene group over the drape frame, and the draped nodes' bundle.
    drape_group: Option<BindGroupId>,
    drape_bundle: Option<RenderBundleId>,
    /// Draped families' pipelines for the drape atlas, made for the first
    /// terrain when the target's format isn't the atlas's.
    drape_pipelines: Option<DrapePipelines>,
    /// Bound for elevated content without terrain.
    no_dem: TextureId,
    /// The DEM texture the scene groups bind.
    dem_bound: TextureId,
    /// Bound for symbols and circles without terrain.
    no_occlusion: TextureId,
    /// The terrain depth bound in the scene groups.
    occlusion_bound: TextureId,
    glyphs: GlyphAtlas,
    /// The glyph atlas revision the scene group binds.
    glyphs_bound: u32,
    /// The scene's buffer revision its groups bind.
    buffers_bound: u64,
    cull_layout: BindGroupLayoutId,
    /// What the scene group binds besides the scene's buffers.
    image_bindings: ImageBindings,
    nearest: SamplerId,
}

/// Bytes reserved per node uniform; every node struct fits.
const NODE_UNIFORM_SIZE: u64 = 512;
/// Where an extrusion node's distance fade is (`extrusion.wgsl`), past
/// its paint.
const EXTRUSION_FADE_WORD: u64 = 52;

impl Renderer {
    pub fn new(gpu: &mut dyn Gpu, target: TargetFormat, capacity: SceneCapacity) -> Self {
        let images = ImageAtlas::new(gpu, 512, 4, 4096);
        Self::with_images(gpu, target, capacity, images.bindings())
    }

    /// Uses the Runtime's shared image array and rect table.
    pub fn with_images(
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        capacity: SceneCapacity,
        images: ImageBindings,
    ) -> Self {
        let scene = Scene::new(gpu, capacity);
        let b = *scene.buffers();
        let storage = |binding, visibility, read_only| BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Storage { read_only },
        };
        let uniform = |binding, visibility| BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Uniform,
        };

        let vf = ShaderStages::VERTEX | ShaderStages::FRAGMENT;
        let vs = ShaderStages::VERTEX;
        let vf_storage = |binding| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Storage { read_only: true },
        };
        let draw_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "scene",
            entries: &[
                uniform(0, vf),
                vf_storage(1),
                storage(2, vs, true),
                storage(3, vs, true),
                storage(4, vs, true),
                storage(5, vs, true),
                storage(6, vs, true),
                storage(7, vs, true),
                storage(8, ShaderStages::FRAGMENT, true),
                BindGroupLayoutEntry {
                    binding: 9,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample: TextureSampleType::Float { filterable: true },
                        dimension: TextureViewDimension::D2Array,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 10,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler { filtering: true },
                },
                BindGroupLayoutEntry {
                    binding: 11,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample: TextureSampleType::Float { filterable: true },
                        dimension: TextureViewDimension::D2,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 12,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample: TextureSampleType::Float { filterable: true },
                        dimension: TextureViewDimension::D2Array,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 13,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler { filtering: true },
                },
                BindGroupLayoutEntry {
                    binding: 14,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Texture {
                        sample: TextureSampleType::Float { filterable: false },
                        dimension: TextureViewDimension::D2Array,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 15,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Texture {
                        sample: TextureSampleType::Float { filterable: false },
                        dimension: TextureViewDimension::D2,
                    },
                },
            ],
        });
        let node_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "node",
            entries: &[uniform(0, vf)],
        });
        let cs = ShaderStages::COMPUTE;
        let cull_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "cull",
            entries: &[
                uniform(0, cs),
                storage(1, cs, true),
                storage(2, cs, true),
                storage(3, cs, false),
                storage(4, cs, true),
                storage(5, cs, false),
                storage(6, cs, false),
                storage(7, cs, true),
                storage(8, cs, false),
            ],
        });

        let glyphs = GlyphAtlas::new(gpu, capacity.glyph_rows, b.props, scene.glyph_table());
        let nearest = gpu.create_sampler(&SamplerDesc {
            label: "nearest",
            filter: FilterMode::Nearest,
            address: AddressMode::ClampToEdge,
            mipmap_filter: FilterMode::Nearest,
            anisotropy: 1,
        });
        let no_dem = gpu.create_texture(&TextureDesc {
            label: "no dem",
            width: 1,
            height: 1,
            array_layers: 1,
            dimension: TextureViewDimension::D2Array,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            mip_levels: 1,
        });
        // Nothing hides symbols and circles without terrain: its depth is
        // the far plane, 0 in reversed-Z.
        let no_occlusion = gpu.create_texture(&TextureDesc {
            label: "no terrain depth",
            width: 1,
            height: 1,
            array_layers: 1,
            dimension: TextureViewDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            mip_levels: 1,
        });
        let draw_group = scene_group(
            gpu,
            draw_layout,
            &b,
            b.frame,
            &images,
            (glyphs.texture(), nearest, no_dem, no_occlusion),
        );
        let modern = gpu.caps().tier == Tier::Modern;
        let cull_group = modern.then(|| cull_group(gpu, cull_layout, &b));

        let symbol_pass_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "symbol pass",
            entries: &[uniform(0, ShaderStages::FRAGMENT)],
        });
        let pass_stride = u64::from(gpu.caps().limits.min_uniform_buffer_offset_alignment).max(16);
        let pass_buffer = gpu.create_buffer(&BufferDesc {
            label: "symbol passes",
            size: 3 * pass_stride,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let symbol_passes = [0u32, 1, 2].map(|pass| {
            gpu.write_buffer(
                pass_buffer,
                u64::from(pass) * pass_stride,
                bytemuck::cast_slice(&[pass, 0, 0, 0]),
            );
            gpu.create_bind_group(&BindGroupDesc {
                label: "symbol pass",
                layout: symbol_pass_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: pass_buffer,
                        offset: u64::from(pass) * pass_stride,
                        size: Some(16),
                    },
                }],
            })
        });
        // The cull pipeline first: nothing draws without it (plan §8.4).
        let cull = cull_group.map(|group| {
            let module = gpu.create_shader_module(&ShaderModuleDesc {
                label: CULL_SHADER.label,
                source: ShaderSource::Wgsl(CULL_SHADER.source().into()),
            });
            let pipeline = gpu.create_compute_pipeline(&ComputePipelineDesc {
                label: CULL_SHADER.label,
                bind_group_layouts: &[cull_layout],
                compute: ShaderStage {
                    module,
                    entry_point: CULL_SHADER.compute,
                    constants: &[],
                },
            });
            (pipeline, group)
        });

        let image_bindings = images;
        let images = image::Images::new(gpu, target, draw_layout);
        let heatmaps = heatmap::Heatmaps::new(gpu, target, draw_layout, node_layout);
        let composites = layer_composite::LayerComposites::new(gpu, target);
        let extrusions = extrusion::Extrusions::new(gpu, target, draw_layout, node_layout);
        let node_stride = NODE_UNIFORM_SIZE.max(u64::from(
            gpu.caps().limits.min_uniform_buffer_offset_alignment,
        ));
        let node_uniforms = gpu.create_buffer(&BufferDesc {
            label: "node uniforms",
            size: node_stride * u64::from(capacity.nodes),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let atmosphere = atmosphere::Atmosphere::new(gpu);
        Self {
            target,
            atmosphere,
            scene,
            draw_group,
            cull,
            draw_layout,
            node_layout,
            paint: PaintPipelines::default(),
            symbol_pipeline: None,
            symbol_passes,
            symbol_pass_layout,
            pipelines_stale: false,
            made_seen: 0,
            waiting: 0,
            drape_waiting: 0,
            node_uniforms,
            node_stride,
            nodes: Vec::new(),
            tiles: Default::default(),
            bundles: Vec::new(),
            paints_written: 0,
            images,
            heatmaps,
            composites,
            extrusions,
            terrain: None,
            drape_group: None,
            drape_bundle: None,
            drape_pipelines: None,
            no_dem,
            dem_bound: no_dem,
            no_occlusion,
            occlusion_bound: no_occlusion,
            glyphs_bound: glyphs.revision(),
            buffers_bound: 0,
            cull_layout,
            nearest,
            glyphs,
            image_bindings,
        }
    }

    /// The scene, to read.
    pub fn scene_ref(&self) -> &Scene {
        &self.scene
    }

    pub fn scene(&mut self) -> &mut Scene {
        &mut self.scene
    }

    /// The glyph atlas symbol quads sample.
    pub fn glyphs(&mut self) -> &mut GlyphAtlas {
        &mut self.glyphs
    }

    /// Uses four samples for 3D surfaces by default, or one when the host
    /// explicitly disables antialiasing (e.g. reference image comparisons).
    pub fn set_antialias_3d(&mut self, gpu: &mut dyn Gpu, enabled: bool) {
        if self.extrusions.set_antialias(enabled) {
            self.pipelines_stale = true;
            self.invalidate_bundle(gpu);
        }
    }

    /// The scene's per-node draw arguments and cull counters, for
    /// diagnostics: [`ARGS_WORDS`] words per node, as
    /// `cull.wgsl` writes them, and the number of nodes.
    pub fn draw_args(&self) -> (BufferId, u32) {
        (self.scene.buffers().args, self.nodes.len() as u32)
    }

    /// Replaces the render nodes, in draw order, none of them drawing a
    /// source's image tiles. The scene must hold no meshes.
    pub fn set_nodes(&mut self, gpu: &mut dyn Gpu, families: &[Family]) {
        let nodes: Vec<RenderNode> = families
            .iter()
            .map(|&family| RenderNode {
                family,
                source: None,
            })
            .collect();
        self.remap_nodes(gpu, &nodes, &vec![None; nodes.len()]);
    }

    /// Replaces the render nodes, in draw order, keeping some with their
    /// meshes: new node `i` is old node `retained[i]`, of the same family
    /// and source, or a new node. Old nodes not retained must hold no
    /// vector meshes; their image tiles' quads go with them, and new image
    /// nodes draw the tiles their source has resident. Paint is written
    /// again for every node. Returns whether every resident image tile
    /// found room in the new nodes.
    pub fn remap_nodes(
        &mut self,
        gpu: &mut dyn Gpu,
        nodes: &[RenderNode],
        retained: &[Option<u32>],
    ) -> bool {
        assert_eq!(nodes.len(), retained.len());
        let families: Vec<Family> = nodes.iter().map(|n| n.family).collect();
        let families = &families[..];
        assert!(
            families.len() as u32 <= self.scene.capacity().nodes,
            "{} render nodes exceed the scene's {}",
            families.len(),
            self.scene.capacity().nodes
        );
        let mut old: Vec<Option<Node>> = self.nodes.drain(..).map(Some).collect();
        let mut map = vec![None; old.len()];
        let mut kept: Vec<Option<Node>> = Vec::with_capacity(families.len());
        for (i, (new, &from)) in nodes.iter().zip(retained).enumerate() {
            let node = from.map(|from| {
                let node = old[from as usize]
                    .take()
                    .expect("each old node is retained once");
                assert_eq!(
                    (node.family, node.source),
                    (new.family, new.source),
                    "a retained node keeps its family and source"
                );
                map[from as usize] = Some(i as u32);
                node
            });
            kept.push(node);
        }
        // Image tiles' quads in nodes leaving go first.
        for (index, _) in map.iter().enumerate().filter(|(_, m)| m.is_none()) {
            let meshes: Vec<MeshId> = self.tiles.meshes_of(index as u32).collect();
            for mesh in meshes {
                self.scene.remove_mesh(mesh);
            }
        }
        self.tiles.remap(&map);
        for node in old.into_iter().flatten() {
            gpu.destroy_bind_group(node.group);
            if let Some(extrusion) = node.extrusion {
                extrusion::Extrusions::release(gpu, extrusion);
            }
            if let Some(heatmap) = node.heatmap {
                heatmap::Heatmaps::release(gpu, heatmap);
            }
            if let Some(image) = node.image {
                image::Images::release_node(gpu, image);
            }
        }
        let mut added = Vec::new();
        for (i, ((&family, node), desc)) in families.iter().zip(kept).zip(nodes).enumerate() {
            let uniform = self.node_uniform(i as u32);
            if node.is_none() && family.is_image() {
                added.push(i as u32);
            }
            let (mut image, cull) = match node {
                Some(node) => {
                    // Groups bind the uniform at the node's index.
                    gpu.destroy_bind_group(node.group);
                    if let Some(extrusion) = node.extrusion {
                        extrusion::Extrusions::release(gpu, extrusion);
                    }
                    if let Some(heatmap) = node.heatmap {
                        heatmap::Heatmaps::release(gpu, heatmap);
                    }
                    (node.image, node.cull)
                }
                None => (
                    family.is_image().then(image::NodeImage::default),
                    NodeCull {
                        margin_px: 0.0,
                        elevation: 0.0,
                        draw: family.draw(),
                        visible: true,
                        target: PlacementTarget::Map,
                    },
                ),
            };
            let group = match &mut image {
                Some(image) => self.images.node_group(gpu, family, image, uniform),
                None => gpu.create_bind_group(&BindGroupDesc {
                    label: "node",
                    layout: self.node_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: uniform,
                    }],
                }),
            };
            self.nodes.push(Node {
                family,
                source: desc.source,
                group,
                extrusion: (family == Family::FillExtrusion)
                    .then(extrusion::NodeExtrusion::default),
                heatmap: (family == Family::Heatmap).then(|| heatmap::Heatmaps::node(gpu)),
                cull,
                written: Vec::new(),
                image,
                separate_halo: false,
                stack: None,
                composite: None,
                fade: [0.0; 2],
            });
        }
        self.assign_stacks();
        let cull: Vec<NodeCull> = self.nodes.iter().map(|n| n.cull).collect();
        self.scene.remap_nodes(&map, &cull);
        self.pipelines_stale = true;
        self.invalidate_bundle(gpu);
        let complete = self.add_image_nodes(gpu, &added);
        let sources: Vec<u32> = self.tiles.sources().map(|(source, _)| source).collect();
        for source in sources {
            self.drop_unused_page(source);
        }
        self.destroy_retired_pages(gpu);
        complete
    }

    /// Puts draped nodes in their drape stacks while terrain is on: each
    /// run of them between nodes that aren't draped is a stack, and a layer
    /// that composites once (below opacity 1 or blending other than
    /// normally) has a stack of its own, whose terrain layer composites
    /// as the layer's.
    fn assign_stacks(&mut self) {
        let on = self.terrain.as_ref().is_some_and(|t| t.paint.is_some());
        let mut stacks = 0u8;
        let mut previous: Option<Option<u32>> = None;
        for node in &mut self.nodes {
            let draped = on && node.family.is_draped();
            let group = composite_group(node.composite);
            if draped && previous != Some(group) {
                stacks = stacks.saturating_add(1);
            }
            previous = draped.then_some(group);
            node.stack = draped.then(|| stacks - 1);
            node.cull.target = match node.stack {
                Some(stack) => PlacementTarget::Drape(stack),
                None => PlacementTarget::Map,
            };
        }
    }

    /// Drape stacks of the current nodes while terrain is on.
    pub fn drape_stacks(&self) -> u32 {
        self.nodes
            .iter()
            .filter_map(|n| n.stack)
            .max()
            .map_or(0, |s| u32::from(s) + 1)
    }

    /// Turns terrain on over a DEM page, or off. Draped nodes then draw
    /// only drape placements: their stack's ([`PlacementTarget::Drape`]).
    pub fn set_terrain(&mut self, gpu: &mut dyn Gpu, terrain: Option<(PageId, TerrainPaint)>) {
        // Compat draws no terrain (plan §8.7).
        let terrain = terrain.filter(|_| gpu.caps().tier == Tier::Modern);
        let was = self.terrain.as_ref().is_some_and(|t| t.paint.is_some());
        match terrain {
            Some((page, paint)) => {
                let target = self.target;
                let dem = self.images.dem_texture(page);
                let dim = self.images.page_size(page)[0];
                let atmosphere = self.atmosphere.uniform;
                let t = self
                    .terrain
                    .get_or_insert_with(|| terrain::Terrain::new(gpu, target, atmosphere));
                t.set_paint(gpu, paint, dem, dim);
            }
            None => {
                if let Some(t) = &mut self.terrain {
                    t.disable(gpu);
                }
            }
        }
        if was != self.terrain.as_ref().is_some_and(|t| t.paint.is_some()) {
            self.assign_stacks();
            self.sync_cull();
            self.pipelines_stale = true;
            self.invalidate_bundle(gpu);
        }
    }

    /// Whether terrain is on: draped nodes draw drape placements.
    pub fn terrain_on(&self) -> bool {
        self.terrain.as_ref().is_some_and(|t| t.paint.is_some())
    }

    /// Makes room for `tile_slots` terrain tiles' drape slots, one per
    /// drape stack. A new layout generation means every slot is blank.
    pub fn reserve_drape(
        &mut self,
        gpu: &mut dyn Gpu,
        tile_slots: u32,
    ) -> Result<DrapeLayout, OutOfSpace> {
        let stacks = self.drape_stacks();
        let terrain = self.terrain.as_mut().expect("terrain is on");
        let before = terrain.layout().generation;
        terrain.reserve(gpu, tile_slots, stacks)?;
        let layout = terrain.layout();
        if layout.generation != before {
            // Stack layers and the main bundle's composites follow it.
            self.invalidate_bundle(gpu);
        }
        Ok(layout)
    }

    /// Redraws these terrain tiles' drape slots this frame: their drape
    /// placements are in the scene.
    pub fn redraw_drape(&mut self, tile_slots: &[u32]) {
        if let Some(terrain) = &mut self.terrain {
            terrain.redraw(tile_slots);
        }
    }

    /// This frame's terrain tiles.
    pub fn set_terrain_tiles(&mut self, gpu: &mut dyn Gpu, tiles: &[TerrainTile]) {
        if let Some(terrain) = &mut self.terrain {
            terrain.set_tiles(gpu, tiles);
        }
    }

    fn node_uniform(&self, node: u32) -> BindingResource {
        BindingResource::Buffer {
            buffer: self.node_uniforms,
            offset: u64::from(node) * self.node_stride,
            size: Some(NODE_UNIFORM_SIZE),
        }
    }

    /// Sets a node's layer composite: its style layer `layer` (nodes of
    /// one layer share it) draws whole and composites once at `opacity`
    /// in `blend` when the opacity is below 1 or the blend isn't normal.
    pub fn set_layer_composite(
        &mut self,
        gpu: &mut dyn Gpu,
        node: u32,
        layer: u32,
        opacity: f32,
        blend: Blend,
    ) {
        let n = &mut self.nodes[node as usize];
        let grouped = |c: Option<(u32, f32, Blend)>| {
            c.filter(|&(_, o, b)| o < 1.0 || b != Blend::Normal)
                .map(|(l, _, b)| (l, b))
        };
        let before = grouped(n.composite);
        n.composite = Some((layer, opacity, blend));
        if grouped(n.composite) != before {
            if n.stack.is_some() {
                // The layer gains or loses a drape stack of its own.
                self.assign_stacks();
                self.sync_cull();
            }
            self.invalidate_bundle(gpu);
        } else if before.is_some() {
            self.composites.set_opacity(gpu, node as usize, opacity);
        }
    }

    fn drop_bundles(&mut self, gpu: &mut dyn Gpu) {
        for bundle in self.bundles.drain(..) {
            gpu.destroy_render_bundle(bundle);
        }
    }

    fn invalidate_bundle(&mut self, gpu: &mut dyn Gpu) {
        self.drop_bundles(gpu);
        if let Some(bundle) = self.drape_bundle.take() {
            gpu.destroy_render_bundle(bundle);
        }
    }

    /// A page of tile textures, `size` texels per tile, for image nodes to
    /// bind with [`bind_page`](Self::bind_page).
    pub fn create_page(&mut self, gpu: &mut dyn Gpu, kind: PageKind, size: [u32; 2]) -> PageId {
        self.images.create_page(gpu, kind, size)
    }

    /// Destroys a page. Nodes that bind it must be rebound or replaced.
    pub fn remove_page(&mut self, gpu: &mut dyn Gpu, page: PageId) {
        self.images.remove_page(gpu, page);
    }

    /// The size of a page's tiles in texels.
    pub fn page_size(&self, page: PageId) -> [u32; 2] {
        self.images.page_size(page)
    }

    /// A free layer of `page` for one tile, growing the page if needed.
    /// GPU bytes of the raster and DEM pages and of the glyph pages.
    pub fn texture_bytes(&self) -> (u64, u64) {
        (self.images.bytes(), self.glyphs.bytes())
    }

    /// A layer of `page`, which grows to the device's layer limit.
    pub fn alloc_layer(&mut self, gpu: &mut dyn Gpu, page: PageId) -> Result<u32, OutOfSpace> {
        self.images.alloc_layer(gpu, page)
    }

    /// Tile page layers in use.
    pub fn used_layers(&self) -> u64 {
        self.images.used_layers()
    }

    pub fn free_layer(&mut self, page: PageId, layer: u32) {
        self.images.free_layer(page, layer);
    }

    /// Writes a tile's image into its layer: premultiplied colour for a
    /// raster page, the encoded elevations for a DEM page.
    pub fn write_layer(&mut self, gpu: &mut dyn Gpu, page: PageId, layer: u32, pixels: Pixels<'_>) {
        self.images.write_layer(gpu, page, layer, pixels);
    }

    /// Queues a DEM tile for the next frame's prepare pass; a later job for
    /// the same layer replaces an earlier one.
    pub fn prepare_dem(&mut self, page: PageId, job: DemPrepare) {
        self.images.prepare_dem(page, job);
    }

    /// Sets a color-relief node's ramp: per stop, its premultiplied rgba8
    /// colour and its elevation packed as a DEM texel.
    pub fn set_ramp(&mut self, gpu: &mut dyn Gpu, node: u32, colors: &[u32], elevations: &[u32]) {
        let image = self.nodes[node as usize]
            .image
            .as_mut()
            .expect("ramps belong to color-relief nodes");
        self.images.set_ramp(gpu, image, colors, elevations);
    }

    /// Updates a heatmap's 256-texel premultiplied RGBA ramp.
    pub fn set_heatmap_ramp(&mut self, gpu: &mut dyn Gpu, node: u32, colors: &[u32]) {
        heatmap::Heatmaps::set_ramp(
            gpu,
            self.nodes[node as usize]
                .heatmap
                .as_mut()
                .expect("heatmap node"),
            colors,
        );
    }

    /// Rebuilds image node groups whose pages or ramps changed.
    fn refresh_image_groups(&mut self, gpu: &mut dyn Gpu) {
        let mut rebuilt = false;
        for i in 0..self.nodes.len() {
            let uniform = self.node_uniform(i as u32);
            let node = &mut self.nodes[i];
            let Some(image) = &mut node.image else {
                continue;
            };
            if !self.images.stale(image) {
                continue;
            }
            gpu.destroy_bind_group(node.group);
            node.group = self.images.node_group(gpu, node.family, image, uniform);
            rebuilt = true;
        }
        if rebuilt {
            self.invalidate_bundle(gpu);
        }
    }

    /// How far node `node`'s drawn pixels reach past its chunks' bounds,
    /// from its last paint.
    pub fn cull_margin_px(&self, node: u32) -> f32 {
        self.nodes[node as usize].cull.margin_px
    }

    /// Writes a node's paint values. The paint's family must match the
    /// node's.
    pub fn set_paint(&mut self, gpu: &mut dyn Gpu, node: u32, paint: &NodePaint) {
        let entry = &mut self.nodes[node as usize];
        assert!(
            paint.fits(entry.family),
            "paint family mismatch for node {node}"
        );
        let mut words = paint.uniform_words(node);
        if let NodePaint::Symbol(p) = paint {
            words[14] = self.glyphs.table_base();
            words[15] = (GLYPH_PAGE as f32).to_bits();
            words[18] = (self.image_bindings.size as f32).to_bits();
            if entry.separate_halo != p.separate_halo {
                entry.separate_halo = p.separate_halo;
                for bundle in self.bundles.drain(..) {
                    gpu.destroy_render_bundle(bundle);
                }
            }
        }
        if words != entry.written {
            gpu.write_buffer(
                self.node_uniforms,
                u64::from(node) * self.node_stride,
                bytemuck::cast_slice(&words),
            );
            entry.written = words;
            self.paints_written += 1;
        }
        let margin = paint.cull_margin_px();
        let elevation = match paint {
            NodePaint::Extrusion(p) | NodePaint::ExtrusionPattern(p, _) => {
                p.height.max(p.base).max(0.0)
            }
            _ => 0.0,
        };
        if margin != entry.cull.margin_px || elevation != entry.cull.elevation {
            entry.cull.margin_px = margin;
            entry.cull.elevation = elevation;
            self.scene.set_node(node, entry.cull);
        }
    }

    /// Fades extrusion node `node` out where the cover's zoom falls below
    /// its layer's: `fade[0]` is the layer's minzoom less the source's
    /// centre zoom, `fade[1]` 1 to fade (`FrameLod`, `extrusion.wgsl`).
    pub fn set_extrusion_fade(&mut self, gpu: &mut dyn Gpu, node: u32, fade: [f32; 2]) {
        let entry = &mut self.nodes[node as usize];
        if entry.fade != fade {
            entry.fade = fade;
            gpu.write_buffer(
                self.node_uniforms,
                u64::from(node) * self.node_stride + EXTRUSION_FADE_WORD * 4,
                bytemuck::cast_slice(&fade),
            );
        }
    }

    /// Shows or hides a node without re-recording the bundle or laying the
    /// cull's candidates out again.
    pub fn set_visible(&mut self, node: u32, visible: bool) {
        let entry = &mut self.nodes[node as usize];
        if entry.cull.visible != visible {
            entry.cull.visible = visible;
            self.scene.set_node(node, entry.cull);
        }
    }

    fn sync_cull(&mut self) {
        let cull: Vec<NodeCull> = self.nodes.iter().map(|n| n.cull).collect();
        self.scene.set_nodes(&cull);
    }

    /// Sets the frame's atmosphere: the style's sky, drawn before the map,
    /// and the fog on terrain. Without a sky the host's backing colour
    /// shows, matching GL JS's transparent default, and terrain has no fog.
    pub fn set_atmosphere(
        &mut self,
        gpu: &mut dyn Gpu,
        paint: Option<AtmospherePaint>,
        view: AtmosphereView,
    ) {
        let frame = self.scene.buffers().frame;
        self.atmosphere.set(gpu, self.target, frame, paint, view);
    }

    /// Asks for the pipelines the nodes need and counts those compiling:
    /// a frame is complete only once none are (plan §8.4).
    pub fn pipelines(&mut self, gpu: &mut dyn Gpu) -> PipelineCounts {
        if std::mem::take(&mut self.pipelines_stale) {
            self.make_pipelines(gpu);
        }
        gpu.poll_pipelines()
    }

    /// Makes the pipelines the nodes draw with that aren't made yet, in
    /// draw order: the device compiles them in the order they are asked
    /// for (the cull's was first), and [`prioritize`](Self::prioritize)
    /// moves what the view shows ahead.
    fn make_pipelines(&mut self, gpu: &mut dyn Gpu) {
        let layouts = [self.draw_layout, self.node_layout];
        let drape_target = TargetFormat {
            color: DRAPE_FORMAT,
            sample_count: 1,
        };
        let own_drape = self.target.color != DRAPE_FORMAT;
        for i in 0..self.nodes.len() {
            let node = &self.nodes[i];
            let family = node.family;
            if node.stack.is_some() {
                // Into the drape atlas, with its format's pipelines if
                // the target's differs.
                if !own_drape {
                    self.make_node_pipeline(gpu, family);
                    continue;
                }
                let drape = self
                    .drape_pipelines
                    .get_or_insert_with(DrapePipelines::default);
                if family.is_image() {
                    self.images.make_for(
                        gpu,
                        &mut drape.images,
                        drape_target,
                        self.draw_layout,
                        family,
                    );
                } else {
                    drape.paint.make(gpu, drape_target, &layouts, family);
                }
                continue;
            }
            self.make_node_pipeline(gpu, family);
        }
    }

    fn make_node_pipeline(&mut self, gpu: &mut dyn Gpu, family: Family) {
        let layouts = [self.draw_layout, self.node_layout];
        match family {
            Family::Fill | Family::FillOutline | Family::Line | Family::Circle => {
                self.paint.make(gpu, self.target, &layouts, family);
            }
            Family::SymbolIcon | Family::SymbolText => {
                let layouts = [self.draw_layout, self.node_layout, self.symbol_pass_layout];
                if self.symbol_pipeline.is_none() {
                    self.symbol_pipeline = Some(symbol_pipeline(gpu, self.target, &layouts));
                }
            }
            Family::Raster | Family::Hillshade | Family::ColorRelief => {
                self.images.make(gpu, family);
            }
            Family::Heatmap => self.heatmaps.make(gpu),
            Family::FillExtrusion => self.extrusions.make(gpu),
        }
    }

    /// The pipelines node `i` draws with, made or not.
    fn node_pipelines(&self, i: usize, out: &mut Vec<PipelineId>) {
        let node = &self.nodes[i];
        if node.stack.is_some() {
            if let Some(terrain) = &self.terrain {
                out.extend(terrain.pipelines());
            }
            if let Some(d) = self.drape_pipelines.as_ref() {
                out.extend(
                    d.paint
                        .get(node.family)
                        .or_else(|| {
                            node.family
                                .is_image()
                                .then(|| d.images.get(node.family))
                                .flatten()
                        })
                        .map(PipelineId::from),
                );
            }
        }
        match node.family {
            Family::Heatmap => out.extend(self.heatmaps.pipelines()),
            Family::FillExtrusion => out.extend(self.extrusions.pipelines()),
            Family::SymbolIcon | Family::SymbolText => {
                out.extend(self.symbol_pipeline.map(PipelineId::from));
            }
            family if family.is_image() => {
                out.extend(self.images.pipeline(family).map(PipelineId::from));
                if family == Family::Raster {
                    out.extend(self.images.mips_pipeline());
                } else {
                    out.extend(self.images.prepare_pipelines());
                }
            }
            family => out.extend(self.paint.get(family).map(PipelineId::from)),
        }
    }

    /// Moves the pipelines of nodes with meshes in the scene ahead of the
    /// others still compiling: what the current view needs first. Nodes
    /// earlier in draw order go first (the bottom layers cover most of
    /// the view), the cull before everything.
    fn prioritize(&self, gpu: &mut dyn Gpu) {
        let mut wanted = Vec::new();
        if let Some((cull, _)) = self.cull {
            wanted.push(cull.into());
        }
        for i in 0..self.nodes.len() {
            if self.scene.node_meshes(i as u32).next().is_some() {
                self.node_pipelines(i, &mut wanted);
            }
        }
        if self.atmosphere.enabled {
            wanted.extend(self.atmosphere.pipelines());
        }
        // Each moves to the front: the last asked for compiles first.
        let mut seen = std::collections::HashSet::new();
        wanted.retain(|&id| seen.insert(id) && !gpu.pipeline_ready(id));
        for &id in wanted.iter().rev() {
            gpu.prioritize_pipeline(id);
        }
    }

    /// Draws a frame into `target`, cleared to `clear` (premultiplied RGBA).
    pub fn render(
        &mut self,
        gpu: &mut dyn Gpu,
        frame: &FrameBlock,
        target: TextureId,
        clear: [f64; 4],
    ) -> FrameStats {
        self.destroy_retired_pages(gpu);
        self.bind_scene(gpu);
        let mut stats = self.render_modern(gpu, frame, target, clear);
        stats.draws += u32::from(self.atmosphere.enabled);
        stats.draws -= stats.nodes_waiting;
        stats
    }

    /// Binds what the scene groups bind anew when it changed: the glyph
    /// pages grew into new textures, the terrain changed, or a scene pool
    /// moved into a larger buffer.
    fn bind_scene(&mut self, gpu: &mut dyn Gpu) {
        let dem = self
            .terrain
            .as_ref()
            .and_then(terrain::Terrain::dem_texture)
            .unwrap_or(self.no_dem);
        let occlusion = self
            .terrain
            .as_ref()
            .and_then(terrain::Terrain::occlusion_texture)
            .unwrap_or(self.no_occlusion);
        let buffers = self.scene.buffers_revision();
        if self.glyphs.revision() != self.glyphs_bound
            || dem != self.dem_bound
            || occlusion != self.occlusion_bound
            || buffers != self.buffers_bound
        {
            if buffers != self.buffers_bound {
                self.buffers_bound = buffers;
                if let Some((_, group)) = &mut self.cull {
                    gpu.destroy_bind_group(*group);
                    *group = cull_group(gpu, self.cull_layout, self.scene.buffers());
                }
            }
            gpu.destroy_bind_group(self.draw_group);
            let b = *self.scene.buffers();
            self.draw_group = scene_group(
                gpu,
                self.draw_layout,
                &b,
                b.frame,
                &self.image_bindings,
                (self.glyphs.texture(), self.nearest, dem, occlusion),
            );
            self.glyphs_bound = self.glyphs.revision();
            self.dem_bound = dem;
            self.occlusion_bound = occlusion;
            if let Some(group) = self.drape_group.take() {
                gpu.destroy_bind_group(group);
            }
            self.invalidate_bundle(gpu);
            // Extrusion and heatmap nodes record the scene group too.
            for node in &mut self.nodes {
                if let Some(extrusion) = &mut node.extrusion {
                    extrusion.forget_bundle(gpu);
                }
                if let Some(heatmap) = &mut node.heatmap {
                    heatmap.forget_bundle(gpu);
                }
            }
        }
    }

    fn render_modern(
        &mut self,
        gpu: &mut dyn Gpu,
        frame: &FrameBlock,
        target: TextureId,
        clear: [f64; 4],
    ) -> FrameStats {
        let (candidates, candidates_rebuilt) = self.scene.prepare(gpu, frame);
        // Preparing may have grown the compacted index buffer.
        self.bind_scene(gpu);
        self.refresh_image_groups(gpu);
        let compiling = self.pipelines(gpu);
        let made = compiling.made;
        if compiling.made != self.made_seen {
            // Pipelines arrived: record again what waited for them.
            self.made_seen = compiling.made;
            if self.waiting > 0 || self.drape_waiting > 0 {
                self.invalidate_bundle(gpu);
            }
        }
        if compiling.pending > 0 {
            self.prioritize(gpu);
        }
        let terrain_on = self.terrain.as_ref().is_some_and(terrain::Terrain::active);
        if let Some(terrain) = self.terrain.as_ref().filter(|_| terrain_on) {
            orb_scene::write_frame(gpu, terrain.drape_frame, &terrain.drape_frame(frame));
            if self.drape_group.is_none() {
                let b = *self.scene.buffers();
                self.drape_group = Some(scene_group(
                    gpu,
                    self.draw_layout,
                    &b,
                    terrain.drape_frame,
                    &self.image_bindings,
                    // Draped content isn't hidden by the terrain it's on.
                    (
                        self.glyphs.texture(),
                        self.nearest,
                        self.dem_bound,
                        self.no_occlusion,
                    ),
                ));
                self.invalidate_bundle(gpu);
            }
        }
        let stacks = if terrain_on { self.drape_stacks() } else { 0 };
        let density_size = frame
            .viewport
            .map(|v| (v * frame.pixel_ratio / 4.0).floor().max(1.0) as u32);
        let size = frame
            .viewport
            .map(|v| (v * frame.pixel_ratio).max(1.0) as u32);
        let mut resized = self.extrusions.resize(
            gpu,
            size,
            self.nodes.iter().filter(|n| n.extrusion.is_some()).count() + stacks as usize,
        );
        if let Some(terrain) = self.terrain.as_mut().filter(|_| terrain_on) {
            resized |= terrain.resize(gpu, size, self.extrusions.samples);
        }
        for i in 0..self.nodes.len() {
            let uniform = self.node_uniform(i as u32);
            if let Some(extrusion) = &mut self.nodes[i].extrusion {
                resized |= self.extrusions.resize_node(gpu, extrusion, uniform, size);
            }
            if let Some(heatmap) = &mut self.nodes[i].heatmap {
                resized |= self.heatmaps.resize(gpu, heatmap, uniform, density_size);
            }
        }
        if resized {
            self.invalidate_bundle(gpu);
        }
        let bundle_recorded = self.bundles.is_empty();
        if bundle_recorded {
            let indexed = self.nodes.iter().any(|n| n.cull.draw == NodeDraw::Indexed);
            let index_buffer = RenderCommand::SetIndexBuffer {
                buffer: self.scene.buffers().compacted,
                format: IndexFormat::Uint32,
                offset: 0,
            };
            let mut ready = Ready {
                cull: self
                    .cull
                    .is_none_or(|(cull, _)| gpu.pipeline_ready(cull.into())),
                terrain: self
                    .terrain
                    .as_mut()
                    .filter(|_| terrain_on)
                    .is_some_and(|t| t.ready(gpu)),
                prepare: self.images.prepare_ready(gpu),
                mips: self.images.mips_ready(gpu),
            };
            let mut commands = Vec::with_capacity(self.nodes.len() * 4 + 1);
            if indexed {
                commands.push(index_buffer.clone());
            }
            // Runs of one layer's nodes below layer opacity 1 or blending
            // other than normally draw into a texture of their own; draped
            // nodes take the terrain's way.
            // On terrain, a draped layer's group is its drape stack: the
            // stack's terrain layer is the texture it composites.
            let mut runs: Vec<layer_composite::Run> = Vec::new();
            for (i, node) in self.nodes.iter().enumerate() {
                let Some(layer) = composite_group(node.composite) else {
                    continue;
                };
                let (_, opacity, blend) = node.composite.expect("grouped");
                let stack = node.stack.filter(|_| terrain_on);
                match runs.last_mut() {
                    Some(run)
                        if run.nodes.end == i
                            && composite_group(self.nodes[run.nodes.start].composite)
                                == Some(layer) =>
                    {
                        run.nodes.end = i + 1;
                    }
                    _ => runs.push(layer_composite::Run {
                        nodes: i..i + 1,
                        opacity,
                        blend,
                        source: stack
                            .and_then(|s| self.terrain.as_ref()?.stack_texture(u32::from(s))),
                    }),
                }
            }
            self.composites.regroup(gpu, runs, size);
            // Until the composites compile, nothing draws: a split main
            // pass needs its blend draw.
            if !self.composites.ready(gpu) {
                self.composites.clear(gpu);
                ready.cull = false;
            }
            self.waiting = 0;
            for g in 0..self.composites.groups.len() {
                if self.composites.groups[g].drawn_elsewhere() {
                    continue;
                }
                let mut group = vec![RenderCommand::SetIndexBuffer {
                    buffer: self.scene.buffers().compacted,
                    format: IndexFormat::Uint32,
                    offset: 0,
                }];
                for i in self.composites.groups[g].nodes.clone() {
                    if !self.node_commands(gpu, &ready, i, self.draw_group, &mut group) {
                        self.waiting += 1;
                    }
                }
                self.composites.groups[g].bundle = Some(gpu.create_render_bundle(
                    &RenderBundleDesc {
                        label: "layer composite",
                        color_formats: &[self.target.color],
                        depth_format: None,
                        sample_count: self.target.sample_count,
                    },
                    &group,
                ));
            }
            let layers = RenderBundleDesc {
                label: "layers",
                color_formats: &[self.target.color],
                depth_format: None,
                sample_count: self.target.sample_count,
            };
            for i in 0..self.nodes.len() {
                let (start, member) = self.composites.at(i);
                match start {
                    // The main pass splits here: its blend draw opens the
                    // next pass, before the next bundle.
                    Some(group) if group.blend != Blend::Normal => {
                        let segment = std::mem::take(&mut commands);
                        self.bundles
                            .push(gpu.create_render_bundle(&layers, &segment));
                        if indexed {
                            commands.push(index_buffer.clone());
                        }
                    }
                    Some(group) => {
                        commands.extend(self.composites.composite_commands(group));
                        // The composite changed the pipeline and bindings.
                        if indexed {
                            commands.push(index_buffer.clone());
                        }
                    }
                    None => {}
                }
                if !member && !self.node_commands(gpu, &ready, i, self.draw_group, &mut commands) {
                    self.waiting += 1;
                }
            }
            if let Some(drape_group) = self.drape_group.filter(|_| terrain_on) {
                let mut drape = vec![RenderCommand::SetIndexBuffer {
                    buffer: self.scene.buffers().compacted,
                    format: IndexFormat::Uint32,
                    offset: 0,
                }];
                let before = self.drape_waiting;
                self.drape_waiting = 0;
                for i in 0..self.nodes.len() {
                    if self.nodes[i].stack.is_some()
                        && !self.node_commands(gpu, &ready, i, drape_group, &mut drape)
                    {
                        self.drape_waiting += 1;
                    }
                }
                if self.drape_waiting < before
                    && let Some(terrain) = &mut self.terrain
                {
                    // Slots drawn without them lack those layers.
                    terrain.redraw_all();
                }
                self.drape_bundle = Some(gpu.create_render_bundle(
                    &RenderBundleDesc {
                        label: "drape",
                        color_formats: &[DRAPE_FORMAT],
                        depth_format: None,
                        sample_count: 1,
                    },
                    &drape,
                ));
            }
            self.bundles
                .push(gpu.create_render_bundle(&layers, &commands));
        }

        let mut passes = Vec::with_capacity(3);
        let mut stats = FrameStats {
            draws: self.nodes.len() as u32,
            bundle_recorded,
            candidates,
            candidates_rebuilt,
            placements_written: self.scene.take_placements_written(),
            paints_written: std::mem::take(&mut self.paints_written),
            pipelines_pending: compiling.pending,
            nodes_waiting: self.waiting,
            ..FrameStats::default()
        };
        if let Some((prepare, prepared)) = self.images.prepare_pass(gpu) {
            stats.dispatches += prepared.dispatches;
            stats.draws += prepared.draws;
            stats.prepared = prepared.tiles;
            passes.extend(prepare);
        }
        if let Some((mips, dispatches)) = self.images.mip_pass(gpu) {
            stats.dispatches += dispatches;
            passes.push(mips);
        }
        // The dispatch writes every node's arguments, zeros included, so it
        // runs even when nothing is resident. Indexed nodes' batches add to
        // cleared arguments.
        #[cfg(feature = "compat")]
        if self.cull.is_none() {
            compat::cull(gpu, &mut self.scene, frame);
        }
        if let Some((pipeline, group)) = self
            .cull
            .filter(|&(cull, _)| !self.nodes.is_empty() && gpu.pipeline_ready(cull.into()))
        {
            if self.nodes.iter().any(|n| n.cull.draw == NodeDraw::Indexed) {
                passes.push(Pass::ClearBuffer {
                    buffer: self.scene.buffers().args,
                    offset: 0,
                    size: Some(args_offset(self.nodes.len() as u32)),
                });
            }
            let [x, y] = self.scene.cull_grid();
            stats.dispatches += 1;
            passes.push(Pass::Compute(ComputePass {
                label: "cull".into(),
                timestamps: None,
                commands: vec![
                    ComputeCommand::SetPipeline(pipeline),
                    ComputeCommand::SetBindGroup { index: 0, group },
                    ComputeCommand::Dispatch { x, y, z: 1 },
                ],
            }));
        }
        let cull_ready = self
            .cull
            .is_none_or(|(cull, _)| gpu.pipeline_ready(cull.into()));
        for (index, node) in self.nodes.iter_mut().enumerate() {
            if let Some(heatmap) = &mut node.heatmap
                && cull_ready
                && let Some(pass) = self.heatmaps.prepare(
                    gpu,
                    heatmap,
                    self.draw_group,
                    node.group,
                    self.scene.buffers().args,
                    index as u32,
                )
            {
                stats.draws += 1;
                passes.push(Pass::Render(pass));
            }
        }
        if let Some(terrain) = self.terrain.as_mut().filter(|_| terrain_on) {
            let depth = self.extrusions.depth().expect("sized above");
            let b = self.scene.buffers();
            stats.draws += terrain.passes(
                gpu,
                (b.frame, b.props),
                self.drape_bundle,
                depth,
                &mut passes,
            );
        }
        let mut first = !terrain_on;
        let extrusions_ready = cull_ready && self.extrusions.composite(gpu).is_some();
        for (index, node) in self.nodes.iter_mut().enumerate() {
            if let Some(extrusion) = node.extrusion.as_mut().filter(|_| extrusions_ready) {
                passes.push(Pass::Render(self.extrusions.prepare(
                    gpu,
                    extrusion,
                    self.draw_group,
                    node.group,
                    self.scene.buffers(),
                    index as u32,
                    first,
                )));
                first = false;
                stats.draws += 1;
            }
        }
        passes.extend(self.composites.passes());
        passes.extend(self.composites.main_passes(
            self.atmosphere.commands(gpu),
            &self.bundles,
            self.atmosphere.commands_after(gpu),
            target,
            clear,
        ));
        gpu.submit(&passes);
        // Every check passed only if nothing was compiling or compiled
        // during the frame.
        let end = gpu.poll_pipelines();
        stats.pipelines_pending = end.pending;
        stats.waited =
            compiling.pending > 0 || end.pending > 0 || end.made != made || self.waiting > 0;
        stats
    }

    /// Node `i`'s commands as a bundle records them, over the scene group
    /// `scene`: the map's, or the drape's for draped nodes on terrain. In
    /// the map, a drape stack draws its terrain layer in its first node's
    /// place. `false` when the node waits for a pipeline and records
    /// nothing.
    fn node_commands(
        &self,
        gpu: &mut dyn Gpu,
        ready: &Ready,
        i: usize,
        scene: BindGroupId,
        commands: &mut Vec<RenderCommand>,
    ) -> bool {
        if !ready.cull {
            return false;
        }
        let node = &self.nodes[i];
        let args = args_offset(i as u32);
        if let (Some(stack), true) = (node.stack, scene == self.draw_group) {
            let first = i == 0 || self.nodes[i - 1].stack != Some(stack);
            if !ready.terrain {
                return false;
            }
            if let Some(terrain) = self.terrain.as_ref().filter(|_| first) {
                commands.extend(terrain.composite_commands(u32::from(stack)));
            }
            return true;
        }
        if let Some(extrusion) = &node.extrusion {
            let Some(composite) = self.extrusions.composite(gpu) else {
                return false;
            };
            if let Some(group) = extrusion.group {
                commands.extend([
                    RenderCommand::SetPipeline(composite),
                    RenderCommand::SetBindGroup {
                        index: 0,
                        group: scene,
                    },
                    RenderCommand::SetBindGroup { index: 1, group },
                    RenderCommand::Draw {
                        vertices: 0..3,
                        instances: 0..1,
                    },
                ]);
            }
            return true;
        }
        if node.family == Family::Heatmap {
            let Some(color) = self.heatmaps.color_pipeline(gpu) else {
                return false;
            };
            if let Some(group) = node.heatmap.as_ref().and_then(|h| h.group) {
                commands.extend([
                    RenderCommand::SetPipeline(color),
                    RenderCommand::SetBindGroup {
                        index: 0,
                        group: scene,
                    },
                    RenderCommand::SetBindGroup { index: 1, group },
                    RenderCommand::Draw {
                        vertices: 0..3,
                        instances: 0..1,
                    },
                ]);
            }
            return true;
        }
        if node.family.is_symbol() {
            let Some(pipeline) = self
                .symbol_pipeline
                .filter(|&p| gpu.pipeline_ready(p.into()))
            else {
                return false;
            };
            let passes = if node.separate_halo {
                &self.symbol_passes[1..]
            } else {
                &self.symbol_passes[..1]
            };
            commands.extend([
                RenderCommand::SetPipeline(pipeline),
                RenderCommand::SetBindGroup {
                    index: 0,
                    group: scene,
                },
                RenderCommand::SetBindGroup {
                    index: 1,
                    group: node.group,
                },
            ]);
            for &group in passes {
                commands.extend([
                    RenderCommand::SetBindGroup { index: 2, group },
                    RenderCommand::DrawIndirect {
                        buffer: self.scene.buffers().args,
                        offset: args,
                    },
                ]);
            }
            return true;
        }
        if matches!(node.family, Family::Hillshade | Family::ColorRelief) && !ready.prepare {
            return false;
        }
        if node.family == Family::Raster && !ready.mips {
            return false;
        }
        // Into the drape atlas, with its format's pipelines if the
        // target's differs.
        let drape = self
            .drape_pipelines
            .as_ref()
            .filter(|_| scene != self.draw_group && self.target.color != DRAPE_FORMAT);
        let pipeline = match (drape, node.family.is_image()) {
            (Some(d), true) => d.images.get(node.family),
            (Some(d), false) => d.paint.get(node.family),
            (None, true) => self.images.pipeline(node.family),
            (None, false) => self.paint.get(node.family),
        };
        let Some(pipeline) = pipeline.filter(|&p| gpu.pipeline_ready(p.into())) else {
            return false;
        };
        commands.extend([
            RenderCommand::SetPipeline(pipeline),
            RenderCommand::SetBindGroup {
                index: 0,
                group: scene,
            },
            RenderCommand::SetBindGroup {
                index: 1,
                group: node.group,
            },
            if node.cull.draw == NodeDraw::Indexed {
                RenderCommand::DrawIndexedIndirect {
                    buffer: self.scene.buffers().args,
                    offset: args,
                }
            } else {
                RenderCommand::DrawIndirect {
                    buffer: self.scene.buffers().args,
                    offset: args,
                }
            },
        ]);
        true
    }
}

/// The style layer a node composites once with, when it does: below layer
/// opacity 1 or blending other than normally.
fn composite_group(composite: Option<(u32, f32, Blend)>) -> Option<u32> {
    composite
        .filter(|&(_, o, b)| o < 1.0 || b != Blend::Normal)
        .map(|(l, _, _)| l)
}

/// What one bundle record found compiled besides the nodes' own
/// pipelines.
struct Ready {
    cull: bool,
    terrain: bool,
    /// The DEM prepare's, which hillshade and color relief wait for.
    prepare: bool,
    /// The raster mip levels', which raster nodes wait for.
    mips: bool,
}

/// Draped families' pipelines for the drape atlas's format, each made for
/// its family's first draped node.
#[derive(Debug, Default)]
struct DrapePipelines {
    paint: PaintPipelines,
    images: image::ImagePipelines,
}

/// A module of `shader` for the device's tier: its WGSL on Modern, or on
/// Compat its offline GLSL.
fn shader_module(gpu: &mut dyn Gpu, shader: &Shader) -> ShaderModuleId {
    #[cfg(feature = "compat")]
    if gpu.caps().tier == Tier::Compat {
        return gpu.create_shader_module(&ShaderModuleDesc {
            label: shader.label,
            source: ShaderSource::Glsl(orb_shaders::glsl(shader)),
        });
    }
    gpu.create_shader_module(&ShaderModuleDesc {
        label: shader.label,
        source: ShaderSource::Wgsl(shader.source().into()),
    })
}

/// The vector families' pipelines (fill, fill outline, line, circle), each
/// made for the family's first node.
#[derive(Clone, Copy, Debug, Default)]
struct PaintPipelines {
    pipelines: [Option<RenderPipelineId>; 4],
}

impl PaintPipelines {
    fn index(family: Family) -> Option<usize> {
        Some(match family {
            Family::Fill => 0,
            Family::FillOutline => 1,
            Family::Line => 2,
            Family::Circle => 3,
            _ => return None,
        })
    }

    fn get(&self, family: Family) -> Option<RenderPipelineId> {
        self.pipelines[Self::index(family)?]
    }

    fn make(
        &mut self,
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        layouts: &[BindGroupLayoutId],
        family: Family,
    ) {
        let Some(index) = Self::index(family) else {
            return;
        };
        let (shader, topology) = match family {
            Family::Fill => (&FILL_SHADER, PrimitiveTopology::TriangleList),
            Family::FillOutline => (&FILL_OUTLINE_SHADER, PrimitiveTopology::LineList),
            Family::Line => (&LINE_SHADER, PrimitiveTopology::TriangleList),
            _ => (&CIRCLE_SHADER, PrimitiveTopology::TriangleList),
        };
        self.pipelines[index]
            .get_or_insert_with(|| draw_pipeline(gpu, shader, topology, target, layouts));
    }
}

/// A draw pipeline over the scene's bindings: `layouts` starts with the
/// scene and node groups.
fn draw_pipeline(
    gpu: &mut dyn Gpu,
    shader: &Shader,
    topology: PrimitiveTopology,
    target: TargetFormat,
    layouts: &[BindGroupLayoutId],
) -> RenderPipelineId {
    draw_pipeline_blending(
        gpu,
        shader,
        topology,
        target,
        layouts,
        Some(BlendState::PremultipliedAlpha),
    )
}

/// [`draw_pipeline`] with `blend` in place of premultiplied blending.
fn draw_pipeline_blending(
    gpu: &mut dyn Gpu,
    shader: &Shader,
    topology: PrimitiveTopology,
    target: TargetFormat,
    layouts: &[BindGroupLayoutId],
    blend: Option<BlendState>,
) -> RenderPipelineId {
    let module = shader_module(gpu, shader);
    gpu.create_render_pipeline(&RenderPipelineDesc {
        cull_mode: None,
        label: shader.label,
        bind_group_layouts: layouts,
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
            format: target.color,
            blend,
        }],
        depth: None,
        topology,
        sample_count: target.sample_count,
    })
}

/// The cull pass's group: the scene's tables and index buffers.
fn cull_group(
    gpu: &mut dyn Gpu,
    layout: BindGroupLayoutId,
    b: &orb_scene::SceneBuffers,
) -> BindGroupId {
    let entries: Vec<BindGroupEntry> = [
        b.frame,
        b.tiles,
        b.chunks,
        b.candidates,
        b.node_cull,
        b.visible,
        b.args,
        b.indices,
        b.compacted,
    ]
    .iter()
    .enumerate()
    .map(|(i, &buffer)| BindGroupEntry {
        binding: i as u32,
        resource: BindingResource::buffer(buffer),
    })
    .collect();
    gpu.create_bind_group(&BindGroupDesc {
        label: "cull",
        layout,
        entries: &entries,
    })
}

/// The scene's group: its buffers, the image pages and the glyph pages.
fn scene_group(
    gpu: &mut dyn Gpu,
    layout: BindGroupLayoutId,
    b: &orb_scene::SceneBuffers,
    frame: BufferId,
    images: &ImageBindings,
    (glyphs, nearest, dem, occlusion): (TextureId, SamplerId, TextureId, TextureId),
) -> BindGroupId {
    let whole = BindingResource::buffer;
    let mut entries: Vec<BindGroupEntry> = [
        frame,
        b.tiles,
        b.chunks,
        b.visible,
        b.vertices,
        b.indices,
        b.props,
        b.node_cull,
        images.table,
    ]
    .iter()
    .enumerate()
    .map(|(i, &buffer)| BindGroupEntry {
        binding: i as u32,
        resource: whole(buffer),
    })
    .collect();
    let texture = |binding, texture, dimension| BindGroupEntry {
        binding,
        resource: BindingResource::Texture { texture, dimension },
    };
    entries.extend([
        texture(9, images.texture, TextureViewDimension::D2Array),
        BindGroupEntry {
            binding: 10,
            resource: BindingResource::Sampler(images.sampler),
        },
        texture(11, images.dashes, TextureViewDimension::D2),
        texture(12, glyphs, TextureViewDimension::D2Array),
        BindGroupEntry {
            binding: 13,
            resource: BindingResource::Sampler(nearest),
        },
        texture(14, dem, TextureViewDimension::D2Array),
        texture(15, occlusion, TextureViewDimension::D2),
    ]);
    gpu.create_bind_group(&BindGroupDesc {
        label: "scene",
        layout,
        entries: &entries,
    })
}

/// The symbol pipeline: over the scene, node and pass groups.
fn symbol_pipeline(
    gpu: &mut dyn Gpu,
    target: TargetFormat,
    layouts: &[BindGroupLayoutId],
) -> RenderPipelineId {
    let module = shader_module(gpu, &SYMBOL_SHADER);
    gpu.create_render_pipeline(&RenderPipelineDesc {
        cull_mode: None,
        label: SYMBOL_SHADER.label,
        bind_group_layouts: layouts,
        vertex: ShaderStage {
            module,
            entry_point: SYMBOL_SHADER.vertex,
            constants: &[],
        },
        fragment: ShaderStage {
            module,
            entry_point: SYMBOL_SHADER.fragment,
            constants: &[],
        },
        targets: &[ColorTargetState {
            format: target.color,
            blend: Some(BlendState::PremultipliedAlpha),
        }],
        depth: None,
        topology: PrimitiveTopology::TriangleList,
        sample_count: target.sample_count,
    })
}
