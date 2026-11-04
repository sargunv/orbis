//! End-to-end frames on the default GPU: layout → scene → cull → bundle.

use orb_hal::{Gpu, TextureDesc, TextureFormat, TextureId, TextureUsages};
use orb_hal_wgpu::WgpuGpu;
use orb_layout::fill;
use orb_render::{
    ClipRect, Curve, DemRef, Family, FillPaint, FrameBlock, FrameStats, MeshData, NodePaint,
    PlacementTarget, Renderer, SceneCapacity, TargetFormat, TileRow, Words,
};
use orb_tile::{GeometryType, LayerBuilder};

const SIZE: u32 = 64;
const WORLD_PER_PX: f32 = 2.0 / SIZE as f32;
const RED: [u8; 4] = [255, 0, 0, 255];
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
        let mut gpu = pollster::block_on(WgpuGpu::offscreen()).expect("a GPU adapter");
        let target = gpu.create_texture(&TextureDesc {
            label: "target",
            width: SIZE,
            height: SIZE,
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            sample_count: 1,
            mip_levels: 1,
        });
        let capacity = SceneCapacity {
            placements: 4,
            chunks: 64,
            candidates: 128,
            nodes: 4,
            vertex_words: 4096,
            index_words: 4096,
            compacted_words: 4096,
            prop_words: 64,
            glyph_rows: 16,
        };
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
        self.renderer
            .render(&mut self.gpu, &frame, self.target, [0.0, 0.0, 1.0, 1.0])
    }

    fn frame(pixel_ratio: f32) -> FrameBlock {
        FrameBlock {
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
            sky_horizon: 0.0,
            symbol_fade_change: 1.0,
            pitch: 0.0,
            terrain: None,
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
        origin: [-1.0 + offset_x, 1.0],
        scale: 2.0 / 4096.0,
        zoom: 0.0,
        clips: [
            WHOLE_TILE,
            ClipRect::EMPTY,
            ClipRect::EMPTY,
            ClipRect::EMPTY,
        ],
        world_per_px: WORLD_PER_PX,
        dem: DemRef::NONE,
        terrain_range: [0.0; 2],
        target: PlacementTarget::Map,
    }
}

fn left_half_square() -> orb_layout::Mesh {
    let mut b = LayerBuilder::new("water", 4096);
    b.begin_feature(None, GeometryType::Polygon);
    b.begin_part();
    b.begin_ring();
    for v in [[0, 0], [2048, 0], [2048, 4096], [0, 4096]] {
        b.push_vertex(v);
    }
    fill::layout(&b.finish(), &[0])
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
            gpu: Default::default(),
        }),
    );
}

#[test]
fn draws_a_fill() {
    let mut h = Harness::new();
    red_fill(&mut h);
    let tile = h.renderer.scene().insert_tile();
    h.renderer
        .scene()
        .place_tile(&mut h.gpu, tile, &[full_view_tile(0.0)])
        .unwrap();
    h.renderer
        .scene()
        .insert_mesh(
            &mut h.gpu,
            tile,
            0,
            &MeshData::new(&left_half_square(), &[], 0),
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
    let tile = h.renderer.scene().insert_tile();
    h.renderer
        .scene()
        .place_tile(&mut h.gpu, tile, &[full_view_tile(0.0)])
        .unwrap();
    let mesh = h
        .renderer
        .scene()
        .insert_mesh(
            &mut h.gpu,
            tile,
            0,
            &MeshData::new(&left_half_square(), &[], 0),
        )
        .unwrap();
    assert_eq!(pixel(&h.render(), 4, 32), RED);

    // Shift the tile right by half the view: the fill now covers the
    // middle.
    h.renderer
        .scene()
        .place_tile(&mut h.gpu, tile, &[full_view_tile(1.0)])
        .unwrap();
    let pixels = h.render();
    assert_eq!(pixel(&pixels, 4, 32), CLEAR);
    assert_eq!(pixel(&pixels, 40, 32), RED);

    // Entirely off screen: culled.
    h.renderer
        .scene()
        .place_tile(&mut h.gpu, tile, &[full_view_tile(4.0)])
        .unwrap();
    assert!(h.render().as_chunks::<4>().0.iter().all(|p| *p == CLEAR));

    h.renderer
        .scene()
        .place_tile(&mut h.gpu, tile, &[full_view_tile(0.0)])
        .unwrap();
    h.renderer.scene().remove_mesh(mesh);
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
            gpu: Default::default(),
        }),
    );
    let tile = h.renderer.scene().insert_tile();
    h.renderer
        .scene()
        .insert_mesh(
            &mut h.gpu,
            tile,
            0,
            &MeshData::new(&left_half_square(), &[], 0),
        )
        .unwrap();
    // Put x=1024 exactly on a pixel centre, so both inclusive clips would
    // shade that column and blend the translucent fill twice.
    let whole = full_view_tile(1.0 / SIZE as f32);
    h.renderer
        .scene()
        .place_tile(&mut h.gpu, tile, &[whole])
        .unwrap();
    let reference = h.render();
    let mut left = whole;
    left.clips[0] = ClipRect::new([0, 0], [1024, 4096]);
    let mut right = whole;
    right.clips[0] = ClipRect::new([1024, 0], [4096, 4096]);
    h.renderer
        .scene()
        .place_tile(&mut h.gpu, tile, &[left, right])
        .unwrap();
    assert_eq!(h.render(), reference);
}

#[test]
fn placements_repeat_and_clip_a_tile() {
    let mut h = Harness::new();
    red_fill(&mut h);
    let tile = h.renderer.scene().insert_tile();
    h.renderer
        .scene()
        .insert_mesh(
            &mut h.gpu,
            tile,
            0,
            &MeshData::new(&left_half_square(), &[], 0),
        )
        .unwrap();

    // Two copies three quarters of a view apart: 0..32 and 48..64 are red.
    h.renderer
        .scene()
        .place_tile(
            &mut h.gpu,
            tile,
            &[full_view_tile(0.0), full_view_tile(1.5)],
        )
        .unwrap();
    let pixels = h.render();
    assert_eq!(pixel(&pixels, 4, 32), RED);
    assert_eq!(pixel(&pixels, 40, 32), CLEAR);
    assert_eq!(pixel(&pixels, 56, 32), RED);

    // One copy clipped to its top-left quadrant.
    let mut row = full_view_tile(0.0);
    row.clips[0] = ClipRect::new([0, 0], [2048, 2048]);
    h.renderer
        .scene()
        .place_tile(&mut h.gpu, tile, &[row])
        .unwrap();
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
    let tile = h.renderer.scene().insert_tile();
    h.renderer
        .scene()
        .place_tile(&mut h.gpu, tile, &[full_view_tile(0.0)])
        .unwrap();
    h.renderer
        .scene()
        .insert_mesh(&mut h.gpu, tile, 0, &data)
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
            gpu: Default::default(),
        }),
    );
    let tile = h.renderer.scene().insert_tile();
    h.renderer
        .scene()
        .place_tile(&mut h.gpu, tile, &[full_view_tile(0.0)])
        .unwrap();
    h.renderer
        .scene()
        .insert_mesh(
            &mut h.gpu,
            tile,
            0,
            &MeshData::new(&left_half_square(), &[], 0),
        )
        .unwrap();
    let pixels = h.render();
    // The left half moved 16 px right: 16..48.
    assert_eq!(pixel(&pixels, 8, 32), CLEAR);
    assert_eq!(pixel(&pixels, 20, 32), RED);
    assert_eq!(pixel(&pixels, 44, 32), RED);
    assert_eq!(pixel(&pixels, 52, 32), CLEAR);
}

#[test]
fn compacted_slots_follow_the_view_without_recording() {
    let mut h = Harness::new();
    red_fill(&mut h);
    let tile = h.renderer.scene().insert_tile();
    h.renderer
        .scene()
        .insert_mesh(
            &mut h.gpu,
            tile,
            0,
            &MeshData::new(&left_half_square(), &[], 0),
        )
        .unwrap();
    // Two copies, one clipped, and one off screen.
    let mut clipped = full_view_tile(1.5);
    clipped.clips[0] = ClipRect::new([0, 0], [2048, 2048]);
    h.renderer
        .scene()
        .place_tile(
            &mut h.gpu,
            tile,
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
        .scene()
        .place_tile(
            &mut h.gpu,
            tile,
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
        gpu: Default::default(),
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
    let tile = h.renderer.scene().insert_tile();
    let mut left = full_view_tile(0.0);
    left.clips[0] = ClipRect::new([0, 0], [2048, 4096]);
    let mut right = full_view_tile(0.0);
    right.clips[0] = ClipRect::new([2048, 0], [4096, 4096]);
    h.renderer
        .scene()
        .place_tile(&mut h.gpu, tile, &[left, right])
        .unwrap();
    let mesh = orb_layout::point::layout(&point_at([2048, 2048]), &[0]);
    h.renderer
        .scene()
        .insert_mesh(&mut h.gpu, tile, 0, &MeshData::new(&mesh, &[], 0))
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
        let tile = h.renderer.scene().insert_tile();
        h.renderer
            .scene()
            .place_tile(&mut h.gpu, tile, &[full_view_tile(0.0)])
            .unwrap();
        let mesh = orb_layout::point::layout_sorted(&point_at([2048, 2048]), &[0], |_| Some(key));
        let mut props = [0; orb_render::CIRCLE_ROW_WORDS];
        props[0] = orb_render::pack_color(color);
        props[1] = props[0];
        h.renderer
            .scene()
            .insert_mesh(
                &mut h.gpu,
                tile,
                0,
                &MeshData::new(&mesh, &props, orb_render::CIRCLE_ROW_WORDS as u32),
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
            gpu: Default::default(),
        }),
    );
    let tile = h.renderer.scene().insert_tile();
    h.renderer
        .scene()
        .place_tile(&mut h.gpu, tile, &[full_view_tile(0.0)])
        .unwrap();
    let mesh = orb_layout::point::layout(&point_at([2048, 2048]), &[0]);
    h.renderer
        .scene()
        .insert_mesh(&mut h.gpu, tile, 0, &MeshData::new(&mesh, &[], 0))
        .unwrap();
    h.renderer
        .scene()
        .insert_mesh(
            &mut h.gpu,
            tile,
            1,
            &MeshData::new(&left_half_square(), &[], 0),
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
    h.renderer.scene().remove_tile(tile);
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
    let tile = h.renderer.scene().insert_tile();
    h.renderer
        .scene()
        .place_tile(&mut h.gpu, tile, &[full_view_tile(0.0)])
        .unwrap();
    let mesh = orb_layout::point::layout(&point_at([2048, 2048]), &[0]);
    h.renderer
        .scene()
        .insert_mesh(&mut h.gpu, tile, 0, &MeshData::new(&mesh, &[], 0))
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
    let tile = h.renderer.scene().insert_tile();
    h.renderer
        .scene()
        .place_tile(&mut h.gpu, tile, &[full_view_tile(0.0)])
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
        .scene()
        .insert_mesh(&mut h.gpu, tile, 0, &MeshData::new(&mesh, &[], 0))
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
                gpu: Default::default(),
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
    let tile = h.renderer.scene().insert_tile();
    h.renderer
        .scene()
        .place_tile(&mut h.gpu, tile, &[full_view_tile(0.0)])
        .unwrap();
    h.renderer
        .scene()
        .insert_mesh(
            &mut h.gpu,
            tile,
            0,
            &MeshData::new(&left_half_square(), &[], 0),
        )
        .unwrap();
    assert_eq!(pixel(&h.render(), 4, 32), CLEAR);
    // The first 4x4 padded image fills page zero; this upload uses page one.
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
            gpu: Default::default(),
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
    let tile = h.renderer.scene().insert_tile();
    h.renderer
        .scene()
        .place_tile(&mut h.gpu, tile, &[full_view_tile(0.0)])
        .unwrap();
    let mut props = [0; orb_render::LINE_ROW_WORDS];
    props[19] = ramp;
    h.renderer
        .scene()
        .insert_mesh(
            &mut h.gpu,
            tile,
            0,
            &MeshData::new(&mesh, &props, orb_render::LINE_ROW_WORDS as u32),
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

#[test]
fn sky_gradient_composites_over_the_host_backing() {
    let mut h = Harness::new();
    let sky = orb_render::SkyPaint {
        color: [0.5, 0.0, 0.0, 0.5],
        horizon_color: [0.5; 4],
        blend: 1.0,
    };
    h.renderer.set_sky(&mut h.gpu, Some(sky));
    let mut frame = Harness::frame(1.0);
    frame.sky_horizon = 32.0;
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
    h.renderer.set_sky(&mut h.gpu, None);
    h.renderer.render(&mut h.gpu, &frame, h.target, clear);
    let pixels = pollster::block_on(h.gpu.read_texture(h.target));
    assert_eq!(pixel(&pixels, 32, 15), CLEAR);
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
