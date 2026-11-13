//! End-to-end frames on the default GPU: layout → scene → cull → bundle.

use orb_hal::{Gpu, TextureDesc, TextureFormat, TextureId, TextureUsages};
use orb_hal_wgpu::WgpuGpu;
use orb_layout::fill;
use orb_render::{
    Blend, ClipRect, Curve, DemRef, Family, FillPaint, FrameBlock, FrameLight, FrameStats,
    MeshData, NodePaint, PlacementTarget, Renderer, SceneCapacity, TargetFormat, TileRow, Words,
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

    /// Pipelines compile on a pool of `threads` workers, or (`None`) before their
    /// request returns, so every frame below draws in full.
    fn with(
        size: u32,
        pages: u32,
        entries: u32,
        capacity: SceneCapacity,
        threads: Option<usize>,
    ) -> Self {
        let mut gpu = pollster::block_on(WgpuGpu::offscreen()).expect("a GPU adapter");
        gpu.set_compile_executor(
            threads
                .map(|n| std::sync::Arc::new(orb_host::jobs::ThreadPool::new("pipelines", n)) as _),
        );
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
        let renderer = Renderer::new(
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

    /// Whether the renderer takes the GPU driver: `ORB_DRIVER=cpu` runs
    /// these tests on the CPU one, whose draws record their counts, so
    /// its bundles record again when those change and leave out what
    /// draws nothing.
    fn gpu_driven(&self) -> bool {
        self.gpu.caps().gpu_driven()
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
        assert!(stats.draws > 0 || !self.gpu_driven());
        pollster::block_on(self.gpu.read_texture(self.target))
    }

    fn render_stats(&mut self) -> FrameStats {
        self.render_stats_at_ratio(1.0)
    }

    fn render_stats_at_ratio(&mut self, pixel_ratio: f32) -> FrameStats {
        let frame = Self::frame(pixel_ratio);
        // The atlas uploads what arrived and grows pages as images do, as a
        // map tells its renderer, and the nodes shown ask for their
        // pipelines: these tests give no pipeline set.
        self.images.flush(&mut self.gpu);
        self.renderer.set_images(self.images.bindings());
        self.renderer.request_pipelines(&mut self.gpu);
        self.renderer.render(
            &mut self.gpu,
            &frame,
            self.target,
            [0.0, 0.0, 1.0, 1.0],
            &|| 0.0,
        )
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
            symbol_fade_change: 1.0,
            time: 0.0,
            state_time: 0.0,
            pitch: 0.0,
            roll: 0.0,
            camera_to_center: 1.0,
            terrain: None,
            light: FrameLight {
                color: [1.0; 3],
                intensity: 0.5,
                position: [0.0, 0.0, 1.0],
            },
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
        bound: ClipRect::EVERYWHERE,
        clips: [
            WHOLE_TILE,
            ClipRect::EMPTY,
            ClipRect::EMPTY,
            ClipRect::EMPTY,
        ],
        px_scale: 1.0,
        sphere: None,
        dem: DemRef::NONE,
        terrain_range: [0.0; 2],
        target: PlacementTarget::Map,
        warp: orb_render::TileWarp::NONE,
        globe: None,
        neighbours: 0,
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
/// pool to fit, within its limit, rather than failing.
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

/// A full block of the vertex and index pools adds another rather than
/// move: the meshes in both draw, nothing is copied, and the block a
/// view emptied goes.
#[test]
fn a_full_block_adds_another_copying_nothing() {
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
    let add = |h: &mut Harness, mesh: &orb_layout::Mesh| {
        let data = MeshData::new(mesh, &[], 0);
        (h.renderer)
            .add_mesh(&mut h.gpu, tile, (0, EXTENT), &data, &Default::default())
            .unwrap()
    };
    add(&mut h, &left);
    let pixels = h.render();
    assert_eq!(pixel(&pixels, 36, 32), CLEAR);
    let first = h.renderer.memory().vertices.allocated;
    h.renderer.take_resized();
    let second = add(&mut h, &right);
    let pixels = h.render();
    assert_eq!(pixel(&pixels, 4, 32), RED, "the first block's mesh");
    assert_eq!(pixel(&pixels, 36, 32), RED, "the second block's mesh");
    let copied = h.renderer.take_resized().bytes_copied;
    assert_copied_no_pool(&h, copied);
    assert!(h.renderer.memory().vertices.allocated > first);
    // The second block empties and goes.
    h.renderer.remove_mesh(tile, second);
    let pixels = h.render();
    assert_eq!(pixel(&pixels, 36, 32), CLEAR);
    assert_eq!(h.renderer.memory().vertices.allocated, first);
}

/// The candidate lists grow as a layout needs them, whatever else
/// happens: written whole, nothing copied, every candidate drawn.
#[test]
fn candidate_lists_grow_as_layouts_need() {
    let capacity = SceneCapacity {
        candidates: 1,
        max_candidates: 1 << 10,
        max_compacted_words: 1 << 14,
        ..Harness::SMALL
    };
    let mut h = Harness::with(4, 2, 8, capacity, None);
    red_fill(&mut h);
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    for x in [0, 2048] {
        let mesh = half_square(x);
        let data = MeshData::new(&mesh, &[], 0);
        (h.renderer)
            .add_mesh(&mut h.gpu, tile, (0, EXTENT), &data, &Default::default())
            .unwrap();
    }
    h.renderer.take_resized();
    let pixels = h.render();
    assert_eq!(pixel(&pixels, 4, 32), RED);
    assert_eq!(pixel(&pixels, 36, 32), RED);
    assert_eq!(h.renderer.stats().candidates_dropped, 0);
    let grew = h.renderer.take_resized();
    assert!(grew.moves > 0, "the lists grew");
    assert_copied_no_pool(&h, grew.bytes_copied);
}

/// Asserts that growth copied nothing. On the CPU driver's data textures
/// four small tables share one texture (WebGL 2 has too few texture units
/// to give each its own), and when one grows the others' rows move with
/// it: a few rows of texels, whatever the pools hold.
fn assert_copied_no_pool(h: &Harness, copied: u64) {
    // Rows of at most 4096 words.
    let (rows, row) = (8, 4096 * 4);
    match h.gpu_driven() {
        true => assert_eq!(copied, 0),
        false => assert!(copied <= rows * row, "{copied} bytes copied"),
    }
}

/// A full tile page grows for a tile past its last layer by a chunk of
/// its own, copying none of the layers it holds, and draws from both; a
/// new tile takes a free layer of the first chunk before the second's,
/// and once the second's last tile goes a settled page lets it go, no
/// layer moving. Raster and DEM pages alike.
#[test]
fn full_tile_pages_grow_and_shrink_by_chunks_keeping_their_layers() {
    use orb_render::{ImageBuild, PageKind, Pixels, RasterPaint, RenderNode, TileKey};
    let mut h = Harness::new();
    let source = 5;
    let node = RenderNode {
        family: Family::Raster,
        source: Some(source),
    };
    assert!(h.renderer.remap_nodes(&mut h.gpu, &[node], &[None]));
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
            fade_duration: 0.0,
        }),
    );
    for (kind, source) in [(PageKind::Raster, source), (PageKind::Dem, source + 1)] {
        let upload = |h: &mut Harness, col: u32, rgba: [u8; 4]| {
            let pixels = rgba.repeat(4);
            let build = ImageBuild {
                key: TileKey {
                    source,
                    tile: orb_crs::TileIndex::new(4, col, 0),
                },
                kind,
                size: [2, 2],
                pixels: Pixels::Core(&pixels),
                mesh: None,
                patches: None,
                fade_from: None,
                unpack: (kind == PageKind::Dem).then_some([0.0; 4]),
                area: None,
            };
            h.renderer.upload_image(&mut h.gpu, &build, true)
        };
        let green = [0, 255, 0, 255];
        let blue = [0, 0, 255, 255];
        // Pages start at eight layers before a map sizes them.
        let first = upload(&mut h, 0, RED).unwrap();
        let mut held = Vec::new();
        for col in 1..8 {
            held.push(upload(&mut h, col, green).unwrap());
        }
        h.renderer.take_resized();
        let (one_chunk, _) = h.renderer.texture_bytes();
        let last = upload(&mut h, 8, blue).unwrap();
        let grew = h.renderer.take_resized();
        assert_eq!(grew.moves, 1, "{kind:?}: a chunk made");
        assert_eq!(grew.bytes_copied, 0, "{kind:?}: nothing copied");
        let draws = |h: &mut Harness, first, last| {
            if kind != PageKind::Raster {
                return;
            }
            // The first tile on the left, the last on the right: each
            // keeps its own texels, whichever chunk holds it (the Adreno
            // 619's WebGL 2 copied layer 0 into every layer).
            let row = |from: i16, to: i16| TileRow {
                extent: 8192.0,
                scale: 2.0 / 8192.0,
                clips: [
                    ClipRect::new([from, 0], [to, 8192]),
                    ClipRect::EMPTY,
                    ClipRect::EMPTY,
                    ClipRect::EMPTY,
                ],
                ..full_view_tile(0.0)
            };
            h.renderer.place(first, 8192, &[row(0, 4096)]).unwrap();
            h.renderer.place(last, 8192, &[row(4096, 8192)]).unwrap();
            let pixels = h.render();
            assert_eq!(pixel(&pixels, 16, 32), RED, "the first tile's texels");
            assert_eq!(pixel(&pixels, 48, 32), blue, "the last tile's texels");
        };
        draws(&mut h, first, last);
        // A tile of the first chunk goes: the settling frame keeps the
        // second, which holds the last, and moves nothing.
        h.renderer.release_tile(held.pop().unwrap());
        let two_chunks = h.renderer.texture_bytes().0;
        h.render_stats();
        assert_eq!(h.renderer.take_resized().bytes_copied, 0, "{kind:?}");
        assert_eq!(h.renderer.texture_bytes().0, two_chunks, "{kind:?}");
        // A new tile takes the layer it left, not one of the second chunk's.
        held.push(upload(&mut h, 9, green).unwrap());
        assert_eq!(h.renderer.texture_bytes().0, two_chunks, "{kind:?}");
        draws(&mut h, first, last);
        // The second chunk's last tile goes, and the chunk with it.
        h.renderer.release_tile(last);
        h.render_stats();
        assert_eq!(h.renderer.take_resized().bytes_copied, 0, "{kind:?}");
        assert_eq!(
            h.renderer.texture_bytes().0,
            one_chunk,
            "{kind:?}: one chunk left"
        );
        // The first chunk is full again: the next tile takes a chunk.
        let last = upload(&mut h, 10, blue).unwrap();
        assert_eq!(h.renderer.texture_bytes().0, two_chunks, "{kind:?}");
        assert_eq!(h.renderer.take_resized().bytes_copied, 0, "{kind:?}");
        draws(&mut h, first, last);
        for tile in [first, last].into_iter().chain(held) {
            h.renderer.release_tile(tile);
        }
        h.render_stats();
        if kind != PageKind::Raster {
            continue;
        }
        // Its last tile gone, a page the map still draws from holds no
        // chunk, and the next tiles make its first again and draw.
        assert_eq!(h.renderer.texture_bytes().0, 0, "no chunk held");
        let (first, last) = (upload(&mut h, 11, RED), upload(&mut h, 12, blue));
        let (first, last) = (first.unwrap(), last.unwrap());
        assert_eq!(h.renderer.texture_bytes().0, one_chunk);
        draws(&mut h, first, last);
        h.renderer.release_tile(first);
        h.renderer.release_tile(last);
        h.render_stats();
    }
}

/// A tile page holds eight times its first chunk's layers before it grows
/// by a copy, settling frames on the way moving none.
#[test]
fn tile_pages_grow_eightfold_without_copying_across_settles() {
    use orb_render::{ImageBuild, PageKind, Pixels, TileKey};
    let mut h = Harness::new();
    let upload = |h: &mut Harness, n: u32| {
        let pixels = RED.repeat(4);
        let build = ImageBuild {
            key: TileKey {
                source: 5,
                tile: orb_crs::TileIndex::new(6, n % 64, n / 64),
            },
            kind: PageKind::Raster,
            size: [2, 2],
            pixels: Pixels::Core(&pixels),
            mesh: None,
            patches: None,
            fade_from: None,
            unpack: None,
            area: None,
        };
        h.renderer.upload_image(&mut h.gpu, &build, true).unwrap()
    };
    // Pages start at eight layers before a map sizes them: a ninth tile
    // takes a second chunk.
    for n in 0..9 {
        upload(&mut h, n);
    }
    h.render_stats();
    for n in 9..64 {
        upload(&mut h, n);
        if n % 16 == 0 {
            h.render_stats();
        }
    }
    assert_eq!(h.renderer.take_resized().bytes_copied, 0, "nothing copied");
    upload(&mut h, 64);
    assert!(h.renderer.take_resized().bytes_copied > 0, "past the four");
}

/// A mesh drawing another's vertices and indices takes no pool words of
/// its own, and keeps drawing them when that mesh goes; they free with
/// the last. A mesh with no words of its own (a point node's indices, a
/// symbol's) holds no share of them, though its empty ranges start where
/// the first allocation does.
#[test]
fn shared_geometry_lives_until_its_last_mesh() {
    let mut h = Harness::new();
    red_fill(&mut h);
    let square = left_half_square();
    // The tile's words: the square's vertices, then its indices.
    let words = [&square.vertices[..], &square.indices].concat();
    let (v, i) = (square.vertices.len() as u32, square.indices.len() as u32);
    let build = |spans, chunks, shares| orb_render::MeshBuild {
        node: 0,
        extent: EXTENT,
        spans,
        row_words: 0,
        chunks,
        rows: Default::default(),
        shares,
    };
    let meshes = [
        build([0..v, v..v + i, 0..0], &square.chunks[..], [None; 2]),
        build([0..0, 0..0, 0..0], &square.chunks[..], [Some(0); 2]),
        build([0..0, 0..0, 0..0], &[], [None; 2]),
    ];
    let key = orb_render::TileKey {
        source: 0,
        tile: orb_crs::TileIndex::new(0, 0, 0),
    };
    let tile_build = orb_render::TileBuild {
        key,
        words: Words::Core(&words),
        meshes: &meshes,
        patches: None,
    };
    let (tile, ids) = h.renderer.upload_tile(&mut h.gpu, &tile_build).unwrap();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    let used = |h: &Harness| {
        let m = h.renderer.memory();
        (m.vertices.used, m.indices.used)
    };
    let words = (
        square.vertices.len() as u64 * 4,
        square.indices.len() as u64 * 4,
    );
    assert_eq!(used(&h), words);
    h.renderer.remove_mesh(tile, ids[2]);
    assert_eq!(used(&h), words);
    h.renderer.remove_mesh(tile, ids[0]);
    assert_eq!(pixel(&h.render(), 4, 32), RED);
    assert_eq!(used(&h), words);
    h.renderer.remove_mesh(tile, ids[1]);
    assert_eq!(pixel(&h.render(), 4, 32), CLEAR);
    assert_eq!(used(&h), (0, 0));
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
            curves: [Curve::STEP; 2],
            state: Default::default(),
        }),
    );
}

/// A style with more render nodes than the scene was made for grows its
/// node tables rather than failing, and the last node draws.
#[test]
fn node_tables_grow_for_more_nodes() {
    let mut h = Harness::with(4, 2, 8, Harness::SMALL, None);
    assert_eq!(Harness::SMALL.nodes, 4);
    h.renderer.set_nodes(&mut h.gpu, &[Family::Fill; 6]);
    let last = 5;
    h.renderer.set_paint(
        &mut h.gpu,
        last,
        &NodePaint::Fill(FillPaint {
            color: [1.0, 0.0, 0.0, 1.0],
            opacity: 1.0,
            translate: Default::default(),
            table_mask: 0,
            curves: [Curve::STEP; 2],
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
            (last, EXTENT),
            &MeshData::new(&left_half_square(), &[], 0),
            &Default::default(),
        )
        .unwrap();
    let pixels = h.render();
    assert_eq!(pixel(&pixels, 4, 32), RED);
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
    // Only what the nodes use is made: the cull (on the GPU) and the
    // fill, with its variant for placements whose fragments it needn't
    // test.
    let pipelines = 2 + u32::from(h.gpu_driven());
    // A map without a pipeline set asks for them as nodes show.
    let asked = h.gpu.counters().pipelines_requested;
    h.renderer.request_pipelines(&mut h.gpu);
    assert_eq!(h.gpu.counters().pipelines_requested - asked, 2);
    let counts = h.gpu.poll_pipelines();
    assert_eq!(counts.made + counts.pending, pipelines, "{counts:?}");
    // One worker can't have compiled them all yet: the fill waits, and
    // the frame draws only the clear colour, or it draws in the pipeline
    // that draws every placement while its other variant compiles.
    let first = h.render_stats();
    if first.pipelines_pending > 0 {
        let pixels = pollster::block_on(h.gpu.read_texture(h.target));
        match first.nodes_waiting {
            0 => assert_eq!((first.draws, pixel(&pixels, 4, 32)), (1, RED)),
            _ => assert_eq!((first.draws, pixel(&pixels, 4, 32)), (0, CLEAR)),
        }
    }
    h.gpu.finish_pipelines();
    // The next frame records the fill.
    let next = h.render_stats();
    assert_eq!((next.pipelines_pending, next.nodes_waiting), (0, 0));
    assert!(next.bundle_recorded || first.pipelines_pending == 0);
    let pixels = pollster::block_on(h.gpu.read_texture(h.target));
    assert_eq!(pixel(&pixels, 4, 32), RED);
    assert_eq!(h.gpu.counters().pipelines_compiled, pipelines);
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

/// A chunk whose vertices lie more than `u16::MAX` apart, as earcut makes
/// of a huge ring, draws copies of them within its u16 indices' reach.
#[test]
fn draws_a_fill_whose_chunk_spans_more_vertices_than_u16_indices_reach() {
    let capacity = SceneCapacity {
        vertex_words: 1 << 18,
        ..Harness::SMALL
    };
    let mut h = Harness::with(4, 2, 8, capacity, None);
    red_fill(&mut h);
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    let mut b = orb_layout::MeshBuilder::new(orb_layout::fill::STRIDE);
    let mut vertex = |[x, y]: [i16; 2]| {
        let position = u32::from(x as u16) | u32::from(y as u16) << 16;
        b.push_vertex([x, y], &[position, 0])
    };
    let first = vertex([0, 0]);
    for _ in 0..70_000 {
        vertex([0, 0]);
    }
    let [right_top, right_bottom, left_bottom] = [[2048, 0], [2048, 4096], [0, 4096]].map(vertex);
    b.push_triangle(first, right_top, right_bottom);
    b.push_triangle(first, right_bottom, left_bottom);
    let mesh = b.finish();
    assert!(mesh.chunks[0].first_vertex > first);
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
    assert_eq!(pixel(&pixels, 4, 32), RED);
    assert_eq!(pixel(&pixels, 28, 4), RED);
    assert_eq!(pixel(&pixels, 36, 32), CLEAR);
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
            curves: [Curve::STEP; 2],
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
                rows: None,
            },
            table_mask: 0,
            curves: [Curve::STEP; 2],
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
    let translated = |px: f32| {
        NodePaint::Fill(FillPaint {
            color: [1.0, 0.0, 0.0, 1.0],
            opacity: 1.0,
            translate: orb_render::Translate {
                px: [px, 0.0],
                viewport: false,
                rows: None,
            },
            table_mask: 0,
            curves: [Curve::STEP; 2],
            state: Default::default(),
        })
    };
    // A translated fill clips where it lands, which only its fragments
    // know: from then on its translation changes as its margin does.
    h.renderer.set_paint(&mut h.gpu, 0, &translated(8.0));
    assert!(h.render_stats().candidates_rebuilt);
    assert!(!h.render_stats().candidates_rebuilt);
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
    let dispatches = u32::from(h.gpu_driven());
    assert_eq!((first.draws, first.dispatches), (1, dispatches));
    assert!(first.bundle_recorded);
    let pixels = pollster::block_on(h.gpu.read_texture(h.target));
    // The whole copy's left half, and the clipped copy's corner.
    assert_eq!(pixel(&pixels, 8, 32), RED);
    assert_eq!(pixel(&pixels, 40, 32), CLEAR);
    assert_eq!(pixel(&pixels, 56, 16), RED);
    assert_eq!(pixel(&pixels, 56, 48), CLEAR);
    // Every slot is written on the first cull: the two visible copies'
    // indices, and hidden ones for the copy off screen.
    let mesh_indices = left_half_square().vertex_indices().count() as u32;
    assert_eq!(
        cull_counters(&mut h),
        [2, 3 * mesh_indices, 2 * mesh_indices]
    );
    // The same view replays and rewrites nothing.
    let second = h.render_stats();
    assert_eq!(
        (second.dispatches, second.bundle_recorded),
        (dispatches, false)
    );
    assert_eq!(pollster::block_on(h.gpu.read_texture(h.target)), pixels);
    assert_eq!(cull_counters(&mut h), [2, 0, 2 * mesh_indices]);
    // The first copy leaves the view and the third enters it: their slots
    // are rewritten, without recording commands on the GPU (the CPU's
    // draw reaches the third slot now).
    h.renderer
        .place(
            tile,
            EXTENT,
            &[full_view_tile(4.0), clipped, full_view_tile(0.0)],
        )
        .unwrap();
    let swapped = h.render_stats();
    assert_eq!(swapped.bundle_recorded, !h.gpu_driven());
    assert_eq!(pollster::block_on(h.gpu.read_texture(h.target)), pixels);
    assert_eq!(
        cull_counters(&mut h),
        [2, 2 * mesh_indices, 2 * mesh_indices]
    );
}

/// With `keep_all` the cull keeps the copy off screen too, and draws the
/// same pixels: it drops only what draws nothing. Back on, it drops it
/// again.
#[test]
fn keeping_every_candidate_draws_the_same_pixels() {
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
    let mut clipped = full_view_tile(1.5);
    clipped.clips[0] = ClipRect::new([0, 0], [2048, 2048]);
    h.renderer
        .place(
            tile,
            EXTENT,
            &[full_view_tile(0.0), clipped, full_view_tile(4.0)],
        )
        .unwrap();
    let culled = h.render();
    let mesh_indices = left_half_square().vertex_indices().count() as u32;
    assert_eq!(cull_counters(&mut h)[0], 2);
    h.renderer.set_keep_all(&mut h.gpu, true);
    assert_eq!(h.render(), culled);
    assert_eq!(cull_counters(&mut h), [3, mesh_indices, 3 * mesh_indices]);
    h.renderer.set_keep_all(&mut h.gpu, false);
    assert_eq!(h.render(), culled);
    assert_eq!(cull_counters(&mut h), [2, mesh_indices, 2 * mesh_indices]);
}

/// The chunks kept, indices written and indices drawn by the last cull
/// (`cull.wgsl`'s counters), summed over every row: each node's, and its
/// second stream's.
fn cull_counters(h: &mut Harness) -> [u32; 3] {
    let (args, _) = h.renderer.draw_args();
    let bytes = pollster::block_on(h.gpu.read_buffer(args));
    let words: &[u32] = bytemuck::cast_slice(&bytes);
    let (rows, _) = words.as_chunks::<{ orb_render::ARGS_WORDS }>();
    rows.iter()
        .fold([0; 3], |sum, row| [0, 1, 2].map(|i| sum[i] + row[5 + i]))
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
        pitch_rows: false,
        table_mask: 0,
        curves: [Curve::STEP; 7],
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
        let mut props = [0; orb_shaders::layout::CIRCLE_ROW_WORDS];
        props[0] = orb_render::pack_color(color);
        props[1] = props[0];
        h.renderer
            .add_mesh(
                &mut h.gpu,
                tile,
                (0, EXTENT),
                &MeshData::new(&mesh, &props, orb_shaders::layout::CIRCLE_ROW_WORDS as u32),
                &Default::default(),
            )
            .unwrap();
    }
    assert_eq!(pixel(&h.render(), 32, 32), [0, 255, 0, 255]);
}

/// Where one layer's circles from two tiles overlap, the tile order says
/// which is on top, whichever arrived first and so whatever block of the
/// pools each landed in: a frame never shows the order tiles came in.
#[test]
fn overlapping_tiles_draw_in_tile_order_whatever_block_they_landed_in() {
    let mesh = orb_layout::point::layout(&point_at([2048, 2048]), &[0]);
    let colors = [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]];
    let draw = |arrivals: [usize; 2]| {
        // A block holds one tile's circle: the second to arrive takes
        // another.
        let capacity = SceneCapacity {
            vertex_words: mesh.vertices.len() as u32,
            index_words: (mesh.indices.len() as u32).max(1),
            max_vertex_words: 1 << 12,
            max_index_words: 1 << 12,
            ..Harness::SMALL
        };
        let mut h = Harness::with(4, 2, 8, capacity, None);
        h.renderer.set_nodes(&mut h.gpu, &[Family::Circle]);
        let NodePaint::Circle(mut paint) = circle_paint([0.0; 4]) else {
            unreachable!()
        };
        paint.table_mask = 1;
        h.renderer
            .set_paint(&mut h.gpu, 0, &NodePaint::Circle(paint));
        let mut tiles = [None; 2];
        for i in arrivals {
            let key = orb_render::TileKey {
                source: 0,
                tile: orb_crs::TileIndex::new(1, i as u32, 0),
            };
            let tile = h.renderer.add_tile(&mut h.gpu, key, None).unwrap();
            h.renderer
                .place(tile, EXTENT, &[full_view_tile(0.0)])
                .unwrap();
            let mut props = [0; orb_shaders::layout::CIRCLE_ROW_WORDS];
            props[0] = orb_render::pack_color(colors[i]);
            props[1] = props[0];
            let data = MeshData::new(&mesh, &props, orb_shaders::layout::CIRCLE_ROW_WORDS as u32);
            h.renderer
                .add_mesh(&mut h.gpu, tile, (0, EXTENT), &data, &Default::default())
                .unwrap();
            tiles[i] = Some(tile);
        }
        // Tile 0, then tile 1: the red circle on top.
        h.renderer.set_tile_order(0, &tiles.map(Option::unwrap));
        pixel(&h.render(), 32, 32)
    };
    assert_eq!(draw([0, 1]), [255, 0, 0, 255], "arrived in tile order");
    assert_eq!(draw([1, 0]), [255, 0, 0, 255], "arrived in reverse");
}

/// Where one layer's fills from three tiles overlap, as two tiles'
/// antialiased edges do on a seam's pixels, the tile order says which is
/// on top, whatever block of the pools each landed in: a node with tiles
/// in several blocks draws them in the tiles' order, not the blocks'.
#[test]
fn overlapping_fills_draw_in_tile_order_whatever_block_they_landed_in() {
    let mut b = LayerBuilder::new("water", 4096);
    b.begin_feature(None, GeometryType::Polygon);
    b.begin_part();
    b.begin_ring();
    for v in [[0, 0], [4096, 0], [4096, 4096], [0, 4096]] {
        b.push_vertex(v);
    }
    let mesh = fill::layout(&b.finish(), &[0]);
    let colors = [
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
    ];
    let draw = |arrivals: [usize; 3]| {
        // A block holds one tile's fill: each tile takes a block of its own.
        let capacity = SceneCapacity {
            vertex_words: mesh.vertices.len() as u32,
            index_words: mesh.indices.len() as u32,
            max_vertex_words: 1 << 12,
            max_index_words: 1 << 12,
            ..Harness::SMALL
        };
        let mut h = Harness::with(4, 2, 8, capacity, None);
        h.renderer.set_nodes(&mut h.gpu, &[Family::Fill]);
        let paint = FillPaint {
            color: [0.0; 4],
            opacity: 1.0,
            translate: Default::default(),
            table_mask: 1,
            curves: [Curve::STEP; 2],
            state: Default::default(),
        };
        h.renderer.set_paint(&mut h.gpu, 0, &NodePaint::Fill(paint));
        let mut tiles = [None; 3];
        for i in arrivals {
            let key = orb_render::TileKey {
                source: 0,
                tile: orb_crs::TileIndex::new(2, i as u32, 0),
            };
            let tile = h.renderer.add_tile(&mut h.gpu, key, None).unwrap();
            h.renderer
                .place(tile, EXTENT, &[full_view_tile(0.0)])
                .unwrap();
            let mut props = [0; orb_shaders::layout::FILL_ROW_WORDS];
            props[0] = orb_render::pack_color(colors[i]);
            props[1] = props[0];
            let data = MeshData::new(&mesh, &props, orb_shaders::layout::FILL_ROW_WORDS as u32);
            h.renderer
                .add_mesh(&mut h.gpu, tile, (0, EXTENT), &data, &Default::default())
                .unwrap();
            tiles[i] = Some(tile);
        }
        // Tiles 0, 1, 2: the red fill on top.
        h.renderer.set_tile_order(0, &tiles.map(Option::unwrap));
        pixel(&h.render(), 32, 32)
    };
    for arrivals in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [2, 0, 1]] {
        assert_eq!(draw(arrivals), [255, 0, 0, 255], "arrived {arrivals:?}");
    }
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
            curves: [Curve::STEP; 2],
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
            curves: [Curve::STEP; 2],
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
    assert_eq!(stats.bundle_recorded, !h.gpu_driven());
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
            curves: [Curve::STEP; 2],
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
                curves: [Curve::STEP; 2],
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
            curves: [Curve::STEP; 6],
            gradient: true,
            metrics: true,
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
    // With `lineMetrics`, vertices carry the progress the ramp reads.
    let butt = |_| orb_layout::line::LineLayout {
        join: orb_layout::line::LineJoin::Miter,
        cap: orb_layout::line::LineCap::Butt,
        miter_limit: 2.0,
        round_limit: 1.05,
    };
    let mesh = orb_layout::line::layout_in_extent(&b.finish(), &[0], butt, 4096, true);
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    let mut props = [0; orb_shaders::layout::LINE_ROW_WORDS];
    props[19] = ramp;
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&mesh, &props, orb_shaders::layout::LINE_ROW_WORDS as u32),
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

/// A raster tile reaching past its set's extent, drawn whole and minified:
/// its texels past its area, black fill, take the nearest inside before
/// its mip levels are made, so no fill reaches into its edge at any level,
/// even where the edge isn't on a level's texel grid.
#[test]
fn raster_texels_past_a_tiles_area_take_its_edge() {
    use orb_render::{ImageBuild, PageKind, Pixels, RasterPaint, RenderNode, TileKey};
    let mut h = Harness::new();
    let source = 3;
    let node = RenderNode {
        family: Family::Raster,
        source: Some(source),
    };
    assert!(h.renderer.remap_nodes(&mut h.gpu, &[node], &[None]));
    let (side, inside) = (256u32, 130u32);
    let pixels: Vec<u8> = (0..side * side)
        .flat_map(|i| {
            if i % side < inside {
                RED
            } else {
                [0, 0, 0, 255]
            }
        })
        .collect();
    let build = ImageBuild {
        key: TileKey {
            source,
            tile: orb_crs::TileIndex::new(0, 0, 0),
        },
        kind: PageKind::Raster,
        size: [side, side],
        pixels: Pixels::Core(&pixels),
        mesh: None,
        patches: None,
        fade_from: None,
        unpack: None,
        area: Some([0, 0, inside, side]),
    };
    let tile = h.renderer.upload_image(&mut h.gpu, &build, true).unwrap();
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
            nearest: false,
            mipmapped: true,
            fade_duration: 0.0,
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
    h.renderer.place(tile, 8192, &[row]).unwrap();
    let pixels = h.render();
    for x in 0..SIZE {
        let p = pixel(&pixels, x, SIZE / 2);
        assert!(p[0] >= 253 && p[1] <= 2 && p[2] <= 2, "x {x}: {p:?}");
    }
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
            area: None,
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
        bound: ClipRect::EVERYWHERE,
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
    h.renderer.request_pipelines(&mut h.gpu);
    h.renderer
        .render(&mut h.gpu, &frame, h.target, [0.0; 4], &|| 0.0);
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
                curves: [Curve::STEP; 2],
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

/// An image layer composites once like a fill: a raster over the whole
/// view, in `mode`, over an opaque fill of the left half and the clear
/// colour beyond it (`raster-layer-blend`, and the hillshade's and colour
/// relief's, which draw as image nodes too).
fn blended_raster(mode: Blend) -> Vec<u8> {
    use orb_render::{ImageBuild, PageKind, Pixels, RasterPaint, RenderNode, TileKey};
    let mut h = Harness::new();
    let source = 7;
    let nodes = [
        RenderNode {
            family: Family::Fill,
            source: None,
        },
        RenderNode {
            family: Family::Raster,
            source: Some(source),
        },
    ];
    assert!(h.renderer.remap_nodes(&mut h.gpu, &nodes, &[None, None]));
    let tile = h.tile();
    h.renderer
        .place(tile, EXTENT, &[full_view_tile(0.0)])
        .unwrap();
    h.renderer.set_paint(
        &mut h.gpu,
        0,
        &NodePaint::Fill(FillPaint {
            color: [0.2, 0.6, 0.4, 1.0],
            opacity: 1.0,
            translate: Default::default(),
            table_mask: 0,
            curves: [Curve::STEP; 2],
            state: Default::default(),
        }),
    );
    h.renderer
        .add_mesh(
            &mut h.gpu,
            tile,
            (0, EXTENT),
            &MeshData::new(&left_half_square(), &[], 0),
            &Default::default(),
        )
        .unwrap();
    let pixels = [204u8, 102, 51, 255].repeat(4);
    let build = ImageBuild {
        key: TileKey {
            source,
            tile: orb_crs::TileIndex::new(0, 0, 0),
        },
        kind: PageKind::Raster,
        size: [2, 2],
        pixels: Pixels::Core(&pixels),
        mesh: None,
        patches: None,
        fade_from: None,
        unpack: None,
        area: None,
    };
    let image = h.renderer.upload_image(&mut h.gpu, &build, true).unwrap();
    h.renderer.set_paint(
        &mut h.gpu,
        1,
        &NodePaint::Raster(RasterPaint {
            opacity: 0.75,
            hue_rotate: 0.0,
            brightness_min: 0.0,
            brightness_max: 1.0,
            saturation: 0.0,
            contrast: 0.0,
            nearest: true,
            mipmapped: false,
            fade_duration: 0.0,
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
    h.renderer.place(image, 8192, &[row]).unwrap();
    h.renderer
        .set_layer_composite(&mut h.gpu, 0, 0, 1.0, Blend::Normal);
    h.renderer.set_layer_composite(&mut h.gpu, 1, 1, 1.0, mode);
    h.render()
}

#[test]
fn image_layer_blends_follow_w3c_compositing() {
    let raster = [0.8, 0.4, 0.2, 1.0].map(|v| v * 0.75);
    for mode in [Blend::Normal, Blend::Multiply, Blend::Screen] {
        let pixels = blended_raster(mode);
        for (x, backdrop) in [(8, [0.2, 0.6, 0.4, 1.0]), (48, [0.0, 0.0, 1.0, 1.0])] {
            let expected = w3c_composite(mode, raster, backdrop);
            let actual = pixel(&pixels, x, 32);
            for (a, e) in actual.into_iter().zip(expected) {
                assert!(
                    (f32::from(a) - e * 255.0).abs() <= 2.0,
                    "{mode:?} at x {x}: {actual:?}, expected {:?}",
                    expected.map(|v| v * 255.0)
                );
            }
        }
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
    };
    let view = sky_view();
    h.renderer.set_atmosphere(&mut h.gpu, Some(sky), view);
    let frame = Harness::frame(1.0);
    let clear = [0.0, 0.0, 1.0, 1.0];
    let report = h
        .renderer
        .render(&mut h.gpu, &frame, h.target, clear, &|| 0.0);
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
    h.renderer
        .render(&mut h.gpu, &frame, h.target, clear, &|| 0.0);
    let pixels = pollster::block_on(h.gpu.read_texture(h.target));
    assert_eq!(pixel(&pixels, 32, 15), CLEAR);
}

/// A flat view whose horizon is row 32, looking along +z with a 60° field
/// of view.
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
        globe: None,
    }
}
