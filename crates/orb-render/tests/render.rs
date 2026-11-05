//! End-to-end frames on the default GPU: layout → scene → cull → bundle.

use orb_hal::{Gpu, TextureDesc, TextureFormat, TextureId, TextureUsages};
use orb_hal_wgpu::WgpuGpu;
use orb_layout::fill;
use orb_render::{
    Blend, ClipRect, Curve, DemRef, Family, FillPaint, FrameBlock, FrameStats, MeshData, NodePaint,
    PlacementTarget, Renderer, SceneCapacity, TargetFormat, TileRow, Words,
};
use orb_tile::{GeometryType, LayerBuilder};

const SIZE: u32 = 64;
const WORLD_PER_PX: f32 = 2.0 / SIZE as f32;
const RED: [u8; 4] = [255, 0, 0, 255];
/// The layer extent the tests' meshes and placements are in.
const EXTENT: u32 = 4096;
const CLEAR: [u8; 4] = [0, 0, 255, 255];

struct Harness {
    gpu: WgpuGpu,
    target: TextureId,
    renderer: Renderer,
    images: orb_render::ImageAtlas,
}

impl Harness {
    fn new() -> Self {
        Self::with_atlas(4, 2, 8)
    }

    fn with_atlas(size: u32, pages: u32, entries: u32) -> Self {
        Self::with(size, pages, entries, Self::SMALL, None)
    }

    const SMALL: SceneCapacity = SceneCapacity {
        placements: 4,
        chunks: 64,
        candidates: 128,
        nodes: 4,
        vertex_words: 4096,
        index_words: 4096,
        compacted_words: 4096,
        prop_words: 64,
        glyph_rows: 16,
        max_vertex_words: 0,
        max_index_words: 0,
        max_compacted_words: 0,
        max_prop_words: 0,
        max_chunks: 0,
        max_candidates: 0,
    };

    /// Pipelines compile on `threads` workers, or (`None`) before their
    /// request returns, so every frame below draws in full.
    fn with(
        size: u32,
        pages: u32,
        entries: u32,
        capacity: SceneCapacity,
        threads: Option<usize>,
    ) -> Self {
        let mut gpu = pollster::block_on(WgpuGpu::offscreen()).expect("a GPU adapter");
        gpu.set_compile_threads(threads);
        let target = gpu.create_texture(&TextureDesc {
            label: "target",
            width: SIZE,
            height: SIZE,
            array_layers: 1,
            dimension: orb_hal::TextureViewDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            sample_count: 1,
            mip_levels: 1,
        });
        let images = orb_render::ImageAtlas::new(&mut gpu, size, pages, entries);
        let renderer = Renderer::with_images(
            &mut gpu,
            TargetFormat {
                color: TextureFormat::Rgba8Unorm,
                sample_count: 1,
            },
            capacity,
            images.bindings(),
        );
        Self {
            gpu,
            target,
            renderer,
            images,
        }
    }

    /// An empty tile to add meshes to and place.
    fn tile(&mut self) -> orb_render::GpuTile {
        let key = orb_render::TileKey {
            source: 0,
            tile: orb_crs::TileIndex::new(0, 0, 0),
        };
        self.renderer.add_tile(&mut self.gpu, key, None).unwrap()
    }

    /// Renders with world space = clip space, so a tile with
    /// [`full_view_tile`] covers the target exactly.
    fn render(&mut self) -> Vec<u8> {
        let stats = self.render_stats();
        assert!(stats.draws > 0);
        pollster::block_on(self.gpu.read_texture(self.target))
    }

    fn render_stats(&mut self) -> FrameStats {
        self.render_stats_at_ratio(1.0)
    }

    fn render_stats_at_ratio(&mut self, pixel_ratio: f32) -> FrameStats {
        let frame = Self::frame(pixel_ratio);
        // The atlas grows pages as images arrive, as a map tells its
        // renderer.
        self.renderer.set_images(self.images.bindings());
        self.renderer
            .render(&mut self.gpu, &frame, self.target, [0.0, 0.0, 1.0, 1.0])
    }

    fn frame(pixel_ratio: f32) -> FrameBlock {
        FrameBlock {
            camera: [0.0; 2],
            world_span: 2.0,
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            viewport: [SIZE as f32; 2],
            world_per_px: WORLD_PER_PX,
            pixel_ratio,
            zoom: 0.0,
            bearing: 0.0,
            align: [0.0; 2],
            extrusion_scale: 1.0,
            ground_horizon: 0.0,
            symbol_fade_change: 1.0,
            time: 0.0,
            pitch: 0.0,
            roll: 0.0,
            camera_to_center: 1.0,
            terrain: None,
            lights: None,
            globe: None,
        }
    }
}

fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    pixels[i..i + 4].try_into().unwrap()
}

const WHOLE_TILE: ClipRect = ClipRect::new([0, 0], [4096, 4096]);

fn full_view_tile(offset_x: f32) -> TileRow {
    TileRow {
        pattern_origin: [0; 2],
        extent: 4096.0,
        wrap: 0,
        origin: [f64::from(-1.0 + offset_x), 1.0],
        scale: 2.0 / 4096.0,
        zoom: 0.0,
        clips: [
            WHOLE_TILE,
            ClipRect::EMPTY,
            ClipRect::EMPTY,
            ClipRect::EMPTY,
        ],
        px_scale: 1.0,
        drape_zoom: 0.0,
        dem: DemRef::NONE,
        terrain_range: [0.0; 2],
        target: PlacementTarget::Map,
        warp: orb_render::TileWarp::NONE,
        globe: None,
    }
}

fn left_half_square() -> orb_layout::Mesh {
    half_square(0)
}

fn half_square(x: i32) -> orb_layout::Mesh {
    let mut b = LayerBuilder::new("water", 4096);
    b.begin_feature(None, GeometryType::Polygon);
    b.begin_part();
    b.begin_ring();
    for v in [[x, 0], [x + 2048, 0], [x + 2048, 4096], [x, 4096]] {
        b.push_vertex(v);
    }
    fill::layout(&b.finish(), &[0])
}

/// A table outside any tile (the feature-state table) grows the property
/// pool to fit, within its limit, rather than failing (#79).
#[test]
fn tables_grow_the_property_pool() {
    let capacity = SceneCapacity {
        max_prop_words: 1 << 12,
        ..Harness::SMALL
    };
    let mut h = Harness::with(4, 2, 8, capacity, None);
    let table = h.renderer.alloc_table(&mut h.gpu, 1024).unwrap();
    assert_eq!(table.len(), 1024);
    assert!(
        h.renderer.alloc_table(&mut h.gpu, 1 << 13).is_err(),
        "past the limit"
    );
}

/// Plan §8: a pool that runs out moves into a buffer twice its size, its
/// meshes copied over on the GPU, and the frame binds the new buffer.
#[test]
fn pools_grow_into_larger_buffers_keeping_their_meshes() {
    let (left, right) = (half_square(0), half_square(2048));
    let capacity = SceneCapacity {
        vertex_words: left.vertices.len() as u32,
        index_words: left.indices.len() as u32,
        max_vertex_words: 1 << 12,
        max_index_words: 1 << 12,
        ..Harness::SMALL
    };
    let mut h = Harness::with(4, 2, 8, capacity, None);
    red_fill(&mut h);
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&left, &[], 0),
            &Default::default(),
        )
        .unwrap();
    let pixels = h.render();
    assert_eq!(pixel(&pixels, 36, 32), CLEAR);
    let mesh = MeshData::new(&right, &[], 0);
    assert!(
        h.renderer
            .add_mesh(&mut h.gpu, tile, (0, EXTENT), &mesh, &Default::default())
            .is_err()
    );
    let size = orb_render::MeshSize {
        vertices: right.vertices.len() as u32,
        indices: right.indices.len() as u32,
        props: 0,
        chunks: right.chunks.len() as u32,
    };
    let revision = h.renderer.pools_revision();
    assert!(h.renderer.grow_for(&mut h.gpu, &[size]));
    assert!(h.renderer.pools_revision() > revision);
    h.renderer
        .add_mesh(&mut h.gpu, tile, (0, EXTENT), &mesh, &Default::default())
        .unwrap();
    let pixels = h.render();
    assert_eq!(pixel(&pixels, 4, 32), RED, "the first mesh copied over");
    assert_eq!(pixel(&pixels, 36, 32), RED, "the second in the grown pool");
    let memory = h.renderer.memory();
    assert_eq!(
        memory.vertices.allocated,
        u64::from(capacity.vertex_words) * 8
    );
    // At its ceiling a pool doesn't grow.
    let huge = orb_render::MeshSize {
        vertices: 1 << 13,
        ..size
    };
    assert!(!h.renderer.grow_for(&mut h.gpu, &[huge]));
}

fn red_fill(h: &mut Harness) {
    h.renderer.set_nodes(&mut h.gpu, &[Family::Fill]);
    h.renderer.set_paint(
        &mut h.gpu,
        0,
        &NodePaint::Fill(FillPaint {
            color: [1.0, 0.0, 0.0, 1.0],
            opacity: 1.0,
            translate: Default::default(),
            table_mask: 0,
            curves: [Curve::Step; 2],
            state: Default::default(),
        }),
    );
}

#[test]
fn nodes_wait_for_their_pipelines_without_blocking_the_frame() {
    let mut h = Harness::with(4, 2, 8, Harness::SMALL, Some(1));
    red_fill(&mut h);
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&left_half_square(), &[], 0),
            &Default::default(),
        )
        .unwrap();
    // Only what the nodes use is made: the cull and the fill.
    let counts = h.renderer.pipelines(&mut h.gpu);
    assert_eq!(counts.made + counts.pending, 2, "{counts:?}");
    // One worker can't have compiled both yet: the fill waits, and the
    // frame draws only the clear colour.
    let first = h.render_stats();
    if first.pipelines_pending > 0 {
        assert_eq!((first.nodes_waiting, first.draws), (1, 0));
        let pixels = pollster::block_on(h.gpu.read_texture(h.target));
        assert_eq!(pixel(&pixels, 4, 32), CLEAR);
    }
    h.gpu.finish_pipelines();
    // The next frame records the fill.
    let next = h.render_stats();
    assert_eq!((next.pipelines_pending, next.nodes_waiting), (0, 0));
    assert!(next.bundle_recorded || first.pipelines_pending == 0);
    let pixels = pollster::block_on(h.gpu.read_texture(h.target));
    assert_eq!(pixel(&pixels, 4, 32), RED);
    assert_eq!(h.gpu.pipelines_created(), 2);
}

#[test]
fn draws_a_fill() {
    let mut h = Harness::new();
    red_fill(&mut h);
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&left_half_square(), &[], 0),
            &Default::default(),
        )
        .unwrap();

    let pixels = h.render();
    assert_eq!(pixel(&pixels, 4, 32), RED);
    assert_eq!(pixel(&pixels, 28, 4), RED);
    assert_eq!(pixel(&pixels, 36, 32), CLEAR);
    assert_eq!(pixel(&pixels, 60, 60), CLEAR);
}

#[test]
fn camera_moves_and_mesh_removal_take_effect() {
    let mut h = Harness::new();
    red_fill(&mut h);
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    let mesh = h
        .renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&left_half_square(), &[], 0),
            &Default::default(),
        )
        .unwrap();
    assert_eq!(pixel(&h.render(), 4, 32), RED);

    // Shift the tile right by half the view: the fill now covers the
    // middle.
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(1.0)])
        .unwrap();
    let pixels = h.render();
    assert_eq!(pixel(&pixels, 4, 32), CLEAR);
    assert_eq!(pixel(&pixels, 40, 32), RED);

    // Entirely off screen: culled.
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(4.0)])
        .unwrap();
    assert!(h.render().as_chunks::<4>().0.iter().all(|p| *p == CLEAR));

    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    h.renderer.remove_mesh(tile, mesh);
    assert!(h.render().as_chunks::<4>().0.iter().all(|p| *p == CLEAR));
}

#[test]
fn adjacent_clip_regions_do_not_double_blend_their_shared_edge() {
    let mut h = Harness::new();
    red_fill(&mut h);
    h.renderer.set_paint(
        &mut h.gpu,
        0,
        &NodePaint::Fill(FillPaint {
            color: [1.0, 0.0, 0.0, 1.0],
            opacity: 0.5,
            translate: Default::default(),
            table_mask: 0,
            curves: [Curve::Step; 2],
            state: Default::default(),
        }),
    );
    let tile = h.tile();
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&left_half_square(), &[], 0),
            &Default::default(),
        )
        .unwrap();
    // Put x=1024 exactly on a pixel centre, so both inclusive clips would
    // shade that column and blend the translucent fill twice.
    let whole = full_view_tile(1.0 / SIZE as f32);
    h.renderer.place(tile, EXTENT, &[whole]).unwrap();
    let reference = h.render();
    let mut left = whole;
    left.clips[0] = ClipRect::new([0, 0], [1024, 4096]);
    let mut right = whole;
    right.clips[0] = ClipRect::new([1024, 0], [4096, 4096]);
    h.renderer.place(tile, EXTENT, &[left, right]).unwrap();
    assert_eq!(h.render(), reference);
}

#[test]
fn placements_repeat_and_clip_a_tile() {
    let mut h = Harness::new();
    red_fill(&mut h);
    let tile = h.tile();
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&left_half_square(), &[], 0),
            &Default::default(),
        )
        .unwrap();

    // Two copies three quarters of a view apart: 0..32 and 48..64 are red.
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0), full_view_tile(1.5)])
        .unwrap();
    let pixels = h.render();
    assert_eq!(pixel(&pixels, 4, 32), RED);
    assert_eq!(pixel(&pixels, 40, 32), CLEAR);
    assert_eq!(pixel(&pixels, 56, 32), RED);

    // One copy clipped to its top-left quadrant.
    let mut row = full_view_tile(0.0);
    row.clips[0] = ClipRect::new([0, 0], [2048, 2048]);
    h.renderer.place(tile, EXTENT, &[row]).unwrap();
    let pixels = h.render();
    assert_eq!(pixel(&pixels, 4, 4), RED);
    assert_eq!(pixel(&pixels, 4, 60), CLEAR);
    assert_eq!(pixel(&pixels, 40, 32), CLEAR);
}

#[test]
fn uploads_streams_from_host_bytes() {
    let mut h = Harness::new();
    red_fill(&mut h);
    let mesh = left_half_square();
    // One host buffer holding both streams, as a job worker transfers them.
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend(mesh.vertices.iter().flat_map(|w| w.to_le_bytes()));
    let indices_at = bytes.len() as u64;
    bytes.extend(mesh.indices.iter().flat_map(|w| w.to_le_bytes()));
    let host = h.gpu.register_bytes(bytes);
    let data = MeshData {
        vertices: Words::Host {
            bytes: host,
            offset: 0,
            len: mesh.vertices.len() as u32,
        },
        indices: Words::Host {
            bytes: host,
            offset: indices_at,
            len: mesh.indices.len() as u32,
        },
        props: Words::Core(&[]),
        row_words: 0,
        chunks: &mesh.chunks,
    };
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    h.renderer
        .add_mesh(&mut h.gpu, tile, (0, EXTENT), &data, &Default::default())
        .unwrap();
    h.gpu.release_host_bytes(host);

    let pixels = h.render();
    assert_eq!(pixel(&pixels, 4, 32), RED);
    assert_eq!(pixel(&pixels, 36, 32), CLEAR);
}

#[test]
fn translates_fills_and_clips_where_they_land() {
    let mut h = Harness::new();
    h.renderer.set_nodes(&mut h.gpu, &[Family::Fill]);
    h.renderer.set_paint(
        &mut h.gpu,
        0,
        &NodePaint::Fill(FillPaint {
            color: [1.0, 0.0, 0.0, 1.0],
            opacity: 1.0,
            translate: orb_render::Translate {
                px: [16.0, 0.0],
                viewport: false,
            },
            table_mask: 0,
            curves: [Curve::Step; 2],
            state: Default::default(),
        }),
    );
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&left_half_square(), &[], 0),
            &Default::default(),
        )
        .unwrap();
    let pixels = h.render();
    // The left half moved 16 px right: 16..48.
    assert_eq!(pixel(&pixels, 8, 32), CLEAR);
    assert_eq!(pixel(&pixels, 20, 32), RED);
    assert_eq!(pixel(&pixels, 44, 32), RED);
    assert_eq!(pixel(&pixels, 52, 32), CLEAR);
}

/// Paint that moves the cull's margin, and hiding and showing a node,
/// change parameters the cull reads: the candidates stay laid out.
#[test]
fn margins_and_visibility_change_without_laying_candidates_out() {
    let mut h = Harness::new();
    red_fill(&mut h);
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&left_half_square(), &[], 0),
            &Default::default(),
        )
        .unwrap();
    assert!(h.render_stats().candidates_rebuilt);
    assert!(!h.render_stats().candidates_rebuilt);
    let translated = |px: f32| {
        NodePaint::Fill(FillPaint {
            color: [1.0, 0.0, 0.0, 1.0],
            opacity: 1.0,
            translate: orb_render::Translate {
                px: [px, 0.0],
                viewport: false,
            },
            table_mask: 0,
            curves: [Curve::Step; 2],
            state: Default::default(),
        })
    };
    h.renderer.set_paint(&mut h.gpu, 0, &translated(16.0));
    assert!(!h.render_stats().candidates_rebuilt);
    let pixels = pollster::block_on(h.gpu.read_texture(h.target));
    assert_eq!(pixel(&pixels, 8, 32), CLEAR);
    assert_eq!(pixel(&pixels, 44, 32), RED);
    h.renderer.set_visible(0, false);
    assert!(!h.render_stats().candidates_rebuilt);
    let pixels = pollster::block_on(h.gpu.read_texture(h.target));
    assert_eq!(pixel(&pixels, 44, 32), CLEAR);
    h.renderer.set_visible(0, true);
    assert!(!h.render_stats().candidates_rebuilt);
    let pixels = pollster::block_on(h.gpu.read_texture(h.target));
    assert_eq!(pixel(&pixels, 44, 32), RED);
}

#[test]
fn compacted_slots_follow_the_view_without_recording() {
    let mut h = Harness::new();
    red_fill(&mut h);
    let tile = h.tile();
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&left_half_square(), &[], 0),
            &Default::default(),
        )
        .unwrap();
    // Two copies, one clipped, and one off screen.
    let mut clipped = full_view_tile(1.5);
    clipped.clips[0] = ClipRect::new([0, 0], [2048, 2048]);
    h.renderer
        .place(
            tile,
            EXTENT,
            &[full_view_tile(0.0), clipped, full_view_tile(4.0)],
        )
        .unwrap();

    let first = h.render_stats();
    assert_eq!((first.draws, first.dispatches), (1, 1));
    assert!(first.bundle_recorded);
    let pixels = pollster::block_on(h.gpu.read_texture(h.target));
    // The whole copy's left half, and the clipped copy's corner.
    assert_eq!(pixel(&pixels, 8, 32), RED);
    assert_eq!(pixel(&pixels, 40, 32), CLEAR);
    assert_eq!(pixel(&pixels, 56, 16), RED);
    assert_eq!(pixel(&pixels, 56, 48), CLEAR);
    // Every slot is written on the first cull: the two visible copies'
    // indices, and hidden ones for the copy off screen.
    let mesh_indices = left_half_square().indices.len() as u32;
    assert_eq!(
        cull_counters(&mut h),
        [2, 3 * mesh_indices, 2 * mesh_indices]
    );
    // The same view replays and rewrites nothing.
    let second = h.render_stats();
    assert_eq!((second.dispatches, second.bundle_recorded), (1, false));
    assert_eq!(pollster::block_on(h.gpu.read_texture(h.target)), pixels);
    assert_eq!(cull_counters(&mut h), [2, 0, 2 * mesh_indices]);
    // The first copy leaves the view and the third enters it: their slots
    // are rewritten, without recording commands.
    h.renderer
        .place(
            tile,
            EXTENT,
            &[full_view_tile(4.0), clipped, full_view_tile(0.0)],
        )
        .unwrap();
    let swapped = h.render_stats();
    assert!(!swapped.bundle_recorded);
    assert_eq!(pollster::block_on(h.gpu.read_texture(h.target)), pixels);
    assert_eq!(
        cull_counters(&mut h),
        [2, 2 * mesh_indices, 2 * mesh_indices]
    );
}

/// The compacted node's chunks kept, indices written and indices drawn by
/// the last cull (`cull.wgsl`'s counters).
fn cull_counters(h: &mut Harness) -> [u32; 3] {
    let (args, _) = h.renderer.draw_args();
    let bytes = pollster::block_on(h.gpu.read_buffer(args));
    let words: &[u32] = bytemuck::cast_slice(&bytes[..orb_render::ARGS_WORDS * 4]);
    [words[5], words[6], words[7]]
}

fn circle_paint(color: [f32; 4]) -> NodePaint {
    NodePaint::Circle(orb_render::CirclePaint {
        color,
        stroke_color: [0.0; 4],
        radius: 8.0,
        blur: 0.0,
        opacity: 1.0,
        stroke_width: 0.0,
        stroke_opacity: 1.0,
        translate: Default::default(),
        pitch_scale_map: false,
        pitch_alignment_map: false,
        table_mask: 0,
        curves: [Curve::Step; 7],
        state: Default::default(),
    })
}

fn point_at(p: [i32; 2]) -> orb_tile::Layer {
    let mut b = LayerBuilder::new("points", 4096);
    b.begin_feature(None, GeometryType::Point);
    b.begin_part();
    b.begin_ring();
    b.push_vertex(p);
    b.finish()
}

#[test]
fn circles_clip_anchors_once_but_keep_their_whole_footprint() {
    let mut h = Harness::new();
    h.renderer.set_nodes(&mut h.gpu, &[Family::Circle]);
    h.renderer
        .set_paint(&mut h.gpu, 0, &circle_paint([0.5, 0.0, 0.0, 0.5]));
    let tile = h.tile();
    let mut left = full_view_tile(0.0);
    left.clips[0] = ClipRect::new([0, 0], [2048, 4096]);
    let mut right = full_view_tile(0.0);
    right.clips[0] = ClipRect::new([2048, 0], [4096, 4096]);
    h.renderer.place(tile, EXTENT, &[left, right]).unwrap();
    let mesh = orb_layout::point::layout(&point_at([2048, 2048]), &[0]);
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&mesh, &[], 0),
            &Default::default(),
        )
        .unwrap();
    let pixels = h.render();
    for x in [28, 32, 36] {
        let p = pixel(&pixels, x, 32);
        // UNORM blend rounding differs by at most one byte across GPUs.
        assert!(
            p.into_iter()
                .zip([128, 0, 128, 255])
                .all(|(a, b)| a.abs_diff(b) <= 1)
        );
    }
    let stable = h.render_stats();
    assert!(!stable.bundle_recorded);
    assert_eq!(stable.paints_written, 0);
}

#[test]
fn circle_sort_keys_merge_across_tiles() {
    let mut h = Harness::new();
    h.renderer.set_nodes(&mut h.gpu, &[Family::Circle]);
    let NodePaint::Circle(mut paint) = circle_paint([0.0; 4]) else {
        unreachable!()
    };
    paint.table_mask = 1;
    h.renderer
        .set_paint(&mut h.gpu, 0, &NodePaint::Circle(paint));
    // Insert the high-key green run first. It must draw above the later
    // low-key red run, even though they arrive as different scene tiles.
    for (key, color) in [(2.0, [0.0, 1.0, 0.0, 1.0]), (1.0, [1.0, 0.0, 0.0, 1.0])] {
        let tile = h.tile();
        h.renderer
            .place(tile, EXTENT, &[full_view_tile(0.0)])
            .unwrap();
        let mesh = orb_layout::point::layout_sorted(&point_at([2048, 2048]), &[0], |_| Some(key));
        let mut props = [0; orb_render::CIRCLE_ROW_WORDS];
        props[0] = orb_render::pack_color(color);
        props[1] = props[0];
        h.renderer
            .add_mesh(
                &mut h.gpu,
                tile,
                (0, EXTENT),
                &MeshData::new(&mesh, &props, orb_render::CIRCLE_ROW_WORDS as u32),
                &Default::default(),
            )
            .unwrap();
    }
    assert_eq!(pixel(&h.render(), 32, 32), [0, 255, 0, 255]);
}

#[test]
fn heatmap_composites_in_layer_order_and_hides_without_recording() {
    let mut h = Harness::new();
    h.renderer
        .set_nodes(&mut h.gpu, &[Family::Heatmap, Family::Fill]);
    h.renderer.set_paint(
        &mut h.gpu,
        0,
        &NodePaint::Heatmap(orb_render::HeatmapPaint {
            radius: 12.0,
            weight: 1.0,
            intensity: 1.0,
            opacity: 1.0,
            table_mask: 0,
            curves: [Curve::Step; 2],
        }),
    );
    let green = orb_render::pack_color([0.0, 1.0, 0.0, 1.0]);
    h.renderer.set_heatmap_ramp(&mut h.gpu, 0, &[green; 256]);
    h.renderer.set_paint(
        &mut h.gpu,
        1,
        &NodePaint::Fill(FillPaint {
            color: [1.0, 0.0, 0.0, 1.0],
            opacity: 1.0,
            translate: Default::default(),
            table_mask: 0,
            curves: [Curve::Step; 2],
            state: Default::default(),
        }),
    );
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    let mesh = orb_layout::point::layout(&point_at([2048, 2048]), &[0]);
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&mesh, &[], 0),
            &Default::default(),
        )
        .unwrap();
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (1, EXTENT),
            &MeshData::new(&left_half_square(), &[], 0),
            &Default::default(),
        )
        .unwrap();
    let pixels = h.render();
    assert_eq!(pixel(&pixels, 16, 32), RED);
    assert_eq!(pixel(&pixels, 48, 32), [0, 255, 0, 255]);
    assert!(h.render_stats_at_ratio(2.0).bundle_recorded);
    assert!(!h.render_stats_at_ratio(2.0).bundle_recorded);
    h.renderer.set_visible(0, false);
    let stats = h.render_stats_at_ratio(2.0);
    assert!(!stats.bundle_recorded);
    assert_eq!(stats.paints_written, 0);
    let pixels = pollster::block_on(h.gpu.read_texture(h.target));
    assert_eq!(pixel(&pixels, 48, 32), CLEAR);
    // Replacing nodes after offscreen work releases the density bindings;
    // the next frame can reuse the scene safely.
    h.renderer.release_tile(tile);
    h.renderer.set_nodes(&mut h.gpu, &[]);
    h.render_stats();
}

/// Growing the glyph pages replaces the texture the scene group binds;
/// the heatmap's own recorded draw binds that group too, so it must be
/// recorded again rather than replayed over the destroyed texture.
#[test]
fn glyph_atlas_growth_rerecords_node_bundles() {
    let mut h = Harness::new();
    h.renderer.set_nodes(&mut h.gpu, &[Family::Heatmap]);
    h.renderer.set_paint(
        &mut h.gpu,
        0,
        &NodePaint::Heatmap(orb_render::HeatmapPaint {
            radius: 12.0,
            weight: 1.0,
            intensity: 1.0,
            opacity: 1.0,
            table_mask: 0,
            curves: [Curve::Step; 2],
        }),
    );
    let green = orb_render::pack_color([0.0, 1.0, 0.0, 1.0]);
    h.renderer.set_heatmap_ramp(&mut h.gpu, 0, &[green; 256]);
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    let mesh = orb_layout::point::layout(&point_at([2048, 2048]), &[0]);
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&mesh, &[], 0),
            &Default::default(),
        )
        .unwrap();
    assert_eq!(pixel(&h.render(), 48, 32), [0, 255, 0, 255]);
    // A page-sized bitmap per row: the ninth outgrows the first 8 layers.
    let revision = h.renderer.glyphs().revision();
    let size = [500, 500];
    let bitmap = vec![255u8; 500 * 500];
    for _ in 0..9 {
        let row = h.renderer.glyphs().reserve().unwrap();
        h.renderer
            .glyphs()
            .upload(&mut h.gpu, row, size, &bitmap)
            .unwrap();
    }
    assert_ne!(h.renderer.glyphs().revision(), revision, "the pages grew");
    assert_eq!(pixel(&h.render(), 48, 32), [0, 255, 0, 255]);
}

#[test]
fn point_instances_address_points_after_the_first_chunk() {
    let mut h = Harness::new();
    h.renderer.set_nodes(&mut h.gpu, &[Family::Circle]);
    h.renderer
        .set_paint(&mut h.gpu, 0, &circle_paint([1.0, 0.0, 0.0, 1.0]));
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    let mut b = LayerBuilder::new("points", 4096);
    b.begin_feature(None, GeometryType::Point);
    b.begin_part();
    b.begin_ring();
    for _ in 0..256 {
        b.push_vertex([128, 128]);
    }
    b.push_vertex([2048, 2048]);
    let mesh = orb_layout::point::layout(&b.finish(), &[0]);
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&mesh, &[], 0),
            &Default::default(),
        )
        .unwrap();
    assert_eq!(pixel(&h.render(), 32, 32), RED);
}

#[test]
fn image_rows_fill_late_span_pages_and_recycle_without_recording_commands() {
    use orb_render::{ImageMetadata, ImageUpload, PatternPaint};
    let mut h = Harness::new();
    let a = h.images.namespace();
    let b = h.images.namespace();
    let first = h.images.resolve(a, "p").unwrap();
    let late = h.images.resolve(b, "p").unwrap();
    assert_ne!(first, late);
    fn upload(pixels: &[u8]) -> ImageUpload<'_> {
        ImageUpload {
            size: [2, 2],
            pixels,
            metadata: ImageMetadata::default(),
            pattern: true,
        }
    }
    h.images
        .upload(&mut h.gpu, first, upload(&[0, 255, 0, 255].repeat(4)))
        .unwrap();
    red_fill(&mut h);
    h.renderer.set_paint(
        &mut h.gpu,
        0,
        &NodePaint::FillPattern(
            FillPaint {
                color: [1.0; 4],
                opacity: 1.0,
                translate: Default::default(),
                table_mask: 0,
                curves: [Curve::Step; 2],
                state: Default::default(),
            },
            PatternPaint {
                from: late,
                to: late,
                from_scale: 1.0,
                to_scale: 1.0,
                mix: 1.0,
                floor_width: 0.0,
            },
        ),
    );
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&left_half_square(), &[], 0),
            &Default::default(),
        )
        .unwrap();
    assert_eq!(pixel(&h.render(), 4, 32), CLEAR);
    // The first 4x4 padded image fills page zero; this upload adds page
    // one, which the next frame binds.
    h.images
        .upload(&mut h.gpu, late, upload(&RED.repeat(4)))
        .unwrap();
    assert_eq!(h.images.size(late), Some([2, 2]));
    assert_eq!(pixel(&h.render(), 4, 32), RED);
    let stats = h.render_stats();
    assert!(!stats.bundle_recorded);
    assert_eq!(stats.paints_written, 0);
    assert_eq!(stats.placements_written, 0);
    h.images.release(&mut h.gpu, b);
    assert_eq!(pixel(&h.render(), 4, 32), CLEAR);
    let next = h.images.namespace();
    assert_eq!(h.images.resolve(next, "new").unwrap(), late);
    h.images
        .upload(&mut h.gpu, late, upload(&[0, 255, 0, 255].repeat(4)))
        .unwrap();
    assert_eq!(pixel(&h.render(), 4, 32), [0, 255, 0, 255]);
    assert!(!h.render_stats().bundle_recorded);
}

#[test]
fn line_texture_rows_are_stable_and_retirement_recycles_sdf_storage() {
    use orb_render::{ImageKey, LinePaint, PatternPaint};
    let mut h = Harness::with_atlas(512, 1, 128);
    h.renderer.set_nodes(&mut h.gpu, &[Family::Line]);
    let namespace = h.images.namespace();
    let key = ImageKey::Dash {
        array: vec![1.0f32.to_bits(); 2],
        round: false,
    };
    let dash = h.images.resolve_key(&mut h.gpu, namespace, &key).unwrap();
    assert_eq!(
        h.images.resolve_key(&mut h.gpu, namespace, &key).unwrap(),
        dash
    );
    let colors = (0..256)
        .map(|i| orb_render::pack_color([1.0 - i as f32 / 255.0, i as f32 / 255.0, 0.0, 1.0]))
        .collect();
    let ramp = h
        .images
        .resolve_key(
            &mut h.gpu,
            namespace,
            &ImageKey::Ramp {
                colors,
                nearest: false,
            },
        )
        .unwrap();
    h.renderer.set_paint(
        &mut h.gpu,
        0,
        &NodePaint::Line(LinePaint {
            color: [1.0, 0.0, 0.0, 1.0],
            opacity: 1.0,
            translate: Default::default(),
            width: 8.0,
            gap_width: 0.0,
            offset: 0.0,
            blur: 0.0,
            table_mask: 0,
            curves: [Curve::Step; 6],
            gradient: true,
            dash: Some(PatternPaint {
                from: dash,
                to: dash,
                from_scale: 1.0,
                to_scale: 1.0,
                mix: 1.0,
                floor_width: 8.0,
            }),
            state: Default::default(),
        }),
    );
    let mut b = LayerBuilder::new("line", 4096);
    b.begin_feature(None, GeometryType::LineString);
    b.begin_part();
    b.begin_ring();
    b.push_vertex([0, 2048]);
    b.push_vertex([4096, 2048]);
    let mesh = orb_layout::line::layout(&b.finish(), &[0], |_| orb_layout::line::LineLayout {
        join: orb_layout::line::LineJoin::Miter,
        cap: orb_layout::line::LineCap::Butt,
        miter_limit: 2.0,
        round_limit: 1.05,
    });
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    let mut props = [0; orb_render::LINE_ROW_WORDS];
    props[19] = ramp;
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&mesh, &props, orb_render::LINE_ROW_WORDS as u32),
            &Default::default(),
        )
        .unwrap();
    let pixels = h.render();
    let c = pixel(&pixels, 4, 32);
    assert!(c[0] > 220 && c[1] > 10 && c[1] < 30 && c[2] == 0, "{c:?}");
    assert_eq!(pixel(&pixels, 12, 32), CLEAR);
    // A ramp shared by two resident tiles survives the first eviction.
    h.images.retain_ramp(ramp);
    h.images.retain_ramp(ramp);
    h.images.release_ramps(&mut h.gpu, [ramp]);
    assert_eq!(pixel(&h.render(), 4, 32), c);
    h.images.release_ramps(&mut h.gpu, [ramp]);
    assert_eq!(pixel(&h.render(), 4, 32), CLEAR);
    // Keep another image live so this exercises rectangle reuse, not a page reset.
    for _ in 0..10 {
        let key = ImageKey::Ramp {
            colors: vec![0xff00ff00; 16384],
            nearest: true,
        };
        let id = h.images.resolve_key(&mut h.gpu, namespace, &key).unwrap();
        h.images.retain_ramp(id);
        h.images.release_ramps(&mut h.gpu, [id]);
    }
    h.images.release(&mut h.gpu, namespace);
    for _ in 0..10 {
        let namespace = h.images.namespace();
        for i in 1..=30 {
            h.images
                .resolve_key(
                    &mut h.gpu,
                    namespace,
                    &ImageKey::Dash {
                        array: vec![(i as f32).to_bits(), 1.0f32.to_bits()],
                        round: true,
                    },
                )
                .unwrap();
        }
        h.images.release(&mut h.gpu, namespace);
    }
    let stats = h.render_stats();
    assert!(!stats.bundle_recorded);
    assert_eq!(stats.paints_written, 0);
    assert_eq!(stats.placements_written, 0);
}

/// A raster node drawing one image tile over the whole view: red, the
/// north-west child of a blue tile a zoom up that is resident with
/// `ancestor`, fading in over 1000 ms from time 0.
fn fading_raster(ancestor: bool) -> Harness {
    use orb_render::{ImageBuild, PageKind, Pixels, RasterPaint, RenderNode, TileKey};
    let mut h = Harness::new();
    let source = 7;
    let node = RenderNode {
        family: Family::Raster,
        source: Some(source),
    };
    assert!(h.renderer.remap_nodes(&mut h.gpu, &[node], &[None]));
    let mut upload = |tile: orb_crs::TileIndex, rgba: [u8; 4], fade_from: Option<f32>| {
        let pixels = rgba.repeat(4);
        let build = ImageBuild {
            key: TileKey { source, tile },
            kind: PageKind::Raster,
            size: [2, 2],
            pixels: Pixels::Core(&pixels),
            mesh: None,
            patches: None,
            fade_from,
            unpack: None,
        };
        h.renderer.upload_image(&mut h.gpu, &build, true).unwrap()
    };
    if ancestor {
        upload(orb_crs::TileIndex::new(0, 0, 0), [0, 0, 255, 255], None);
    }
    let child = upload(
        orb_crs::TileIndex::new(1, 0, 0),
        [255, 0, 0, 255],
        Some(0.0),
    );
    h.renderer.set_paint(
        &mut h.gpu,
        0,
        &NodePaint::Raster(RasterPaint {
            opacity: 1.0,
            hue_rotate: 0.0,
            brightness_min: 0.0,
            brightness_max: 1.0,
            saturation: 0.0,
            contrast: 0.0,
            nearest: true,
            mipmapped: false,
            fade_duration: 1000.0,
        }),
    );
    let row = TileRow {
        extent: 8192.0,
        scale: 2.0 / 8192.0,
        clips: [
            ClipRect::new([0, 0], [8192, 8192]),
            ClipRect::EMPTY,
            ClipRect::EMPTY,
            ClipRect::EMPTY,
        ],
        ..full_view_tile(0.0)
    };
    h.renderer.place(child, 8192, &[row]).unwrap();
    h
}

fn render_at(h: &mut Harness, time: f32) -> [u8; 4] {
    let frame = FrameBlock {
        time,
        ..Harness::frame(1.0)
    };
    h.renderer.render(&mut h.gpu, &frame, h.target, [0.0; 4]);
    pixel(&pollster::block_on(h.gpu.read_texture(h.target)), 16, 16)
}

#[test]
fn raster_tiles_cross_fade_from_their_ancestor_or_fade_in() {
    let close = |a: [u8; 4], b: [u8; 4]| a.iter().zip(b).all(|(&x, y)| x.abs_diff(y) <= 2);
    // Halfway, half the ancestor's blue; then the tile alone.
    let mut h = fading_raster(true);
    let half = render_at(&mut h, 500.0);
    assert!(close(half, [128, 0, 127, 255]), "{half:?}");
    assert_eq!(render_at(&mut h, 1500.0), RED);
    // Without an ancestor it fades in over what's beneath.
    let mut h = fading_raster(false);
    let half = render_at(&mut h, 250.0);
    assert!(close(half, [64, 0, 0, 64]), "{half:?}");
    assert_eq!(render_at(&mut h, 1000.0), RED);
}

/// W3C Compositing 1: `source` (premultiplied) over `backdrop` in `mode`.
fn w3c_composite(mode: Blend, s: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    if mode == Blend::PlusLighter {
        return std::array::from_fn(|i| (s[i] + b[i]).min(1.0));
    }
    let unpremultiply = |c: [f32; 4]| -> [f32; 3] {
        std::array::from_fn(|i| if c[3] > 0.0 { c[i] / c[3] } else { 0.0 })
    };
    let (cs, cb) = (unpremultiply(s), unpremultiply(b));
    let lum = |c: [f32; 3]| 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
    let set_lum = |c: [f32; 3], l: f32| {
        let d = l - lum(c);
        let c = c.map(|v| v + d);
        let (l, n, x) = (lum(c), c[0].min(c[1]).min(c[2]), c[0].max(c[1]).max(c[2]));
        let c = if n < 0.0 {
            c.map(|v| l + (v - l) * l / (l - n))
        } else {
            c
        };
        if x > 1.0 {
            c.map(|v| l + (v - l) * (1.0 - l) / (x - l))
        } else {
            c
        }
    };
    let mixed: [f32; 3] = match mode {
        Blend::Normal => cs,
        Blend::Multiply => std::array::from_fn(|i| cb[i] * cs[i]),
        Blend::Screen => std::array::from_fn(|i| cb[i] + cs[i] - cb[i] * cs[i]),
        Blend::Overlay => std::array::from_fn(|i| {
            if cb[i] <= 0.5 {
                cs[i] * 2.0 * cb[i]
            } else {
                let d = 2.0 * cb[i] - 1.0;
                cs[i] + d - cs[i] * d
            }
        }),
        Blend::Color => set_lum(cs, lum(cb)),
        Blend::Luminosity => set_lum(cb, lum(cs)),
        Blend::PlusLighter => unreachable!(),
    };
    let (a_s, a_b) = (s[3], b[3]);
    let mut out: [f32; 4] = std::array::from_fn(|i| {
        if i < 3 {
            s[i] * (1.0 - a_b) + b[i] * (1.0 - a_s) + a_s * a_b * mixed[i]
        } else {
            0.0
        }
    });
    out[3] = a_s + a_b * (1.0 - a_s);
    out
}

/// Three fills of the left half: an opaque base, then two layers that
/// composite once in `modes`, the second over the first's result.
fn blended_fills(modes: [Blend; 2]) -> Vec<u8> {
    let colors = [
        [0.2, 0.6, 0.4, 1.0],
        [0.9, 0.3, 0.5, 1.0],
        [0.1, 0.4, 0.8, 1.0],
    ];
    let mut h = Harness::new();
    h.renderer.set_nodes(&mut h.gpu, &[Family::Fill; 3]);
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    for (node, color) in colors.into_iter().enumerate() {
        h.renderer.set_paint(
            &mut h.gpu,
            node as u32,
            &NodePaint::Fill(FillPaint {
                color,
                opacity: 1.0,
                translate: Default::default(),
                table_mask: 0,
                curves: [Curve::Step; 2],
                state: Default::default(),
            }),
        );
        let (opacity, blend) = match node {
            0 => (1.0, Blend::Normal),
            _ => (0.75, modes[node - 1]),
        };
        h.renderer
            .set_layer_composite(&mut h.gpu, node as u32, node as u32, opacity, blend);
        h.renderer
            .add_mesh(
                &mut h.gpu,
                tile,
                (node as u32, EXTENT),
                &MeshData::new(&left_half_square(), &[], 0),
                &Default::default(),
            )
            .unwrap();
    }
    h.render()
}

#[test]
fn layer_blends_follow_w3c_compositing() {
    let modes = [
        Blend::Normal,
        Blend::Multiply,
        Blend::Screen,
        Blend::Overlay,
        Blend::Color,
        Blend::Luminosity,
        Blend::PlusLighter,
    ];
    let base = [0.2, 0.6, 0.4, 1.0];
    let layer = |c: [f32; 4]| c.map(|v| v * 0.75);
    for (first, second) in modes.into_iter().zip(modes.into_iter().rev()) {
        let pixels = blended_fills([first, second]);
        let once = w3c_composite(first, layer([0.9, 0.3, 0.5, 1.0]), base);
        let expected = w3c_composite(second, layer([0.1, 0.4, 0.8, 1.0]), once);
        let actual = pixel(&pixels, 8, 32);
        for (a, e) in actual.into_iter().zip(expected) {
            assert!(
                (f32::from(a) - e * 255.0).abs() <= 2.0,
                "{first:?} then {second:?}: {actual:?}, expected {:?}",
                expected.map(|v| v * 255.0)
            );
        }
        // Where the layers draw nothing, the map beneath shows unchanged.
        assert_eq!(pixel(&pixels, 48, 32), CLEAR, "{first:?} then {second:?}");
    }
}

#[test]
fn sky_gradient_composites_over_the_host_backing() {
    let mut h = Harness::new();
    let sky = orb_render::AtmospherePaint {
        sky_color: [0.5, 0.0, 0.0, 0.5],
        horizon_color: [0.5; 4],
        fog_color: [1.0; 4],
        sky_horizon_blend: 1.0,
        horizon_fog_blend: 0.5,
        fog_ground_blend: 0.5,
        atmosphere_blend: 0.8,
        backdrop: [0.0; 4],
        stars: 0.0,
    };
    let view = sky_view();
    h.renderer.set_atmosphere(&mut h.gpu, Some(sky), view);
    let frame = Harness::frame(1.0);
    let clear = [0.0, 0.0, 1.0, 1.0];
    let report = h.renderer.render(&mut h.gpu, &frame, h.target, clear);
    assert_eq!(report.draws, 1);
    let pixels = pollster::block_on(h.gpu.read_texture(h.target));
    // At row 15, GL JS's squared horizon weight is (1 - 16.5/32)^2.
    let horizon_weight = (1.0f32 - 16.5 / 32.0).powi(2);
    let actual = pixel(&pixels, 32, 15);
    let expected = [
        128.0,
        127.5 * horizon_weight,
        127.5 * (1.0 + horizon_weight),
        255.0,
    ];
    for (a, b) in actual.into_iter().zip(expected) {
        assert!((f32::from(a) - b).abs() <= 1.0, "{actual:?}");
    }
    assert_eq!(pixel(&pixels, 32, 40), CLEAR);
    // Removing the style's sky restores transparent/background output.
    h.renderer.set_atmosphere(&mut h.gpu, None, view);
    h.renderer.render(&mut h.gpu, &frame, h.target, clear);
    let pixels = pollster::block_on(h.gpu.read_texture(h.target));
    assert_eq!(pixel(&pixels, 32, 15), CLEAR);
}

/// A flat view whose horizon is row 32, looking along +z with a 60° field
/// of view: its rays leave the target toward the north pole.
fn sky_view() -> orb_render::AtmosphereView {
    let k = (30f32).to_radians().tan();
    orb_render::AtmosphereView {
        horizon: [0.0, 32.0],
        horizon_normal: [0.0, -1.0],
        height: SIZE as f32,
        fog_range: [1.0, 2.0],
        fog_opacity: 0.0,
        // Clip (x, y, z) → (xk, yk, 1) / (1 − z/2): the ray (xk, yk, 1).
        ray_inverse: [
            [k, 0.0, 0.0, 0.0],
            [0.0, k, 0.0, 0.0],
            [0.0, 0.0, 0.0, -0.5],
            [0.0, 0.0, 1.0, 1.0],
        ],
        star_rotation: 0.0,
        globe: None,
    }
}

#[test]
fn stars_and_backdrop_show_behind_a_clear_sky() {
    let mut h = Harness::new();
    let night = orb_render::AtmospherePaint {
        sky_color: [0.0; 4],
        horizon_color: [0.0; 4],
        fog_color: [0.0; 4],
        sky_horizon_blend: 0.5,
        horizon_fog_blend: 0.5,
        fog_ground_blend: 0.5,
        atmosphere_blend: 0.0,
        backdrop: [0.0, 0.0, 0.0, 0.0],
        stars: 1.0,
    };
    let frame = Harness::frame(1.0);
    let clear = [0.0, 0.0, 1.0, 1.0];
    h.renderer
        .set_atmosphere(&mut h.gpu, Some(night), sky_view());
    h.renderer.render(&mut h.gpu, &frame, h.target, clear);
    let pixels = pollster::block_on(h.gpu.read_texture(h.target));
    let lit = |rows: std::ops::Range<u32>| {
        rows.flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .filter(|&(x, y)| pixel(&pixels, x, y)[0] > 16)
            .count()
    };
    // Stars over the host's backing above the horizon, none below it.
    assert!(lit(0..31) > 20, "{} stars", lit(0..31));
    assert_eq!(lit(33..SIZE), 0);
    // An opaque backdrop replaces the backing wherever nothing draws.
    let dusk = orb_render::AtmospherePaint {
        backdrop: [0.0, 0.5, 0.0, 1.0],
        stars: 0.0,
        ..night
    };
    h.renderer
        .set_atmosphere(&mut h.gpu, Some(dusk), sky_view());
    h.renderer.render(&mut h.gpu, &frame, h.target, clear);
    let pixels = pollster::block_on(h.gpu.read_texture(h.target));
    for (x, y) in [(32, 10), (32, 50)] {
        assert_eq!(pixel(&pixels, x, y), [0, 128, 0, 255]);
    }
}

/// Real basemaps have hundreds of render nodes (Americana about 400).
#[test]
fn default_capacity_holds_a_large_basemaps_nodes() {
    let mut gpu = pollster::block_on(WgpuGpu::offscreen()).expect("a GPU adapter");
    let mut renderer = Renderer::new(
        &mut gpu,
        TargetFormat {
            color: TextureFormat::Rgba8Unorm,
            sample_count: 1,
        },
        SceneCapacity::default(),
    );
    renderer.set_nodes(&mut gpu, &[Family::Line; 400]);
}
