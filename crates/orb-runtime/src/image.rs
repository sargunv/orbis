//! Image tiles on the scene side: raster and DEM tiles into their source's
//! texture page as one quad per render node, DEM neighbours for the
//! prepare pass, and the paint of raster, hillshade and color-relief nodes.

use std::sync::Arc;

use orb_crs::TileIndex;
use orb_hal::Gpu;
use orb_layout::{Chunk, TileBox};
use orb_render::{
    ColorReliefPaint, DemPrepare, HillshadeMethod, HillshadePaint, IMAGE_EXTENT, IMAGE_ROW_WORDS,
    ImageFade, MAX_LIGHTS, MeshData, MeshSize, NodePaint, PageId, PageKind, Pixels, RasterPaint,
    Renderer, Words,
};
use orb_source::TilePyramid;
use orb_v8::{Color, IlluminationAnchor, LayerKind, Property, PropertyValue, Resampling};

use crate::jobs::{ImagePixels, TileImage};
use crate::map::{Resident, ResidentTile};
use crate::style::{ImageKind, NodePlan, StylePlan};

/// An image tile's layer in its source's page, and the mesh that warps it
/// onto another plane.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResidentImage {
    pub(crate) page: PageId,
    pub(crate) layer: u32,
    pub(crate) warped: Option<Arc<Warped>>,
    /// How it fades in, decided when it arrived.
    pub(crate) fade: ImageFade,
}

/// A tile's warp mesh as its nodes draw it (plan §9.2, `RasterWarp`).
#[derive(Debug, PartialEq)]
pub(crate) struct Warped {
    /// The square of the drawing plane its positions span `IMAGE_EXTENT`
    /// of.
    pub(crate) frame: orb_crs::WorldRect,
    vertices: Vec<u32>,
    indices: Vec<u32>,
    chunks: Vec<Chunk>,
}

impl Warped {
    /// The tile's mesh from `warp`, its linear interpolation within a
    /// quarter of a pixel at 512 pixels per tile side.
    pub(crate) fn new(warp: &orb_crs::Warp, tile: TileIndex) -> Option<Self> {
        Some(Self::of(warp.mesh(tile, 1.0 / 2048.0, 1.0)?, None))
    }

    /// A tile's own square (`frame`) as a grid of `cells` per side, for a
    /// globe to bend, with its top or bottom edge fanned out to the North
    /// or South Pole (caps) as `poles` ask: each cap triangle stretches its
    /// edge's texels to the pole, as GL JS's do.
    pub(crate) fn grid(frame: orb_crs::WorldRect, cells: u32, poles: [bool; 2]) -> Self {
        let n = cells + 1;
        let mut positions: Vec<glam::DVec2> = (0..n * n)
            .map(|k| glam::DVec2::new(f64::from(k % n), f64::from(k / n)) / f64::from(cells))
            .collect();
        let mut sources = positions.clone();
        let mut triangles: Vec<[u32; 3]> = (0..cells * cells)
            .flat_map(|c| {
                let k = c / cells * n + c % cells;
                [[k, k + 1, k + n + 1], [k, k + n + 1, k + n]]
            })
            .collect();
        let pole_y = [
            orb_layout::fill::NORTH_POLE_Y,
            orb_layout::fill::SOUTH_POLE_Y,
        ];
        for ((cap, row), pole) in poles.into_iter().zip([0, cells]).zip(pole_y) {
            if !cap {
                continue;
            }
            // Positions in image units, where the shader reads the pole.
            let y = f64::from(pole) / f64::from(IMAGE_EXTENT);
            for i in 0..cells {
                let tip = positions.len() as u32;
                let middle = (f64::from(i) + 0.5) / f64::from(cells);
                positions.push(glam::DVec2::new(middle, y));
                sources.push(glam::DVec2::new(middle, f64::from(row) / f64::from(cells)));
                triangles.push([row * n + i, row * n + i + 1, tip]);
            }
        }
        Self::of(
            orb_crs::WarpMesh {
                frame,
                sources,
                positions,
                triangles,
            },
            None,
        )
    }

    /// An image source's mesh: its quad's warp at the vertices of a grid
    /// of `cells` per side over the image, in the tile `frame`, with the
    /// image's own positions as sources (GL JS warps per vertex too).
    pub(crate) fn quad(frame: orb_crs::WorldRect, quad: &orb_crs::ImageQuad, cells: u32) -> Self {
        let n = cells + 1;
        let sources: Vec<glam::DVec2> = (0..n * n)
            .map(|k| glam::DVec2::new(f64::from(k % n), f64::from(k / n)) / f64::from(cells))
            .collect();
        let size = frame.size();
        let positions = sources
            .iter()
            .map(|&uv| {
                let p = quad.at(uv);
                glam::DVec2::new((p.x - frame.min.x) / size.x, (frame.max.y - p.y) / size.y)
            })
            .collect();
        let triangles = (0..cells * cells)
            .flat_map(|c| {
                let k = c / cells * n + c % cells;
                [[k, k + 1, k + n + 1], [k, k + n + 1, k + n]]
            })
            .collect();
        let weights: Vec<f32> = sources.iter().map(|&uv| quad.weight(uv) as f32).collect();
        Self::of(
            orb_crs::WarpMesh {
                frame,
                positions,
                sources,
                triangles,
            },
            Some(&weights),
        )
    }

    fn of(mesh: orb_crs::WarpMesh, weights: Option<&[f32]>) -> Self {
        let quad = weights.is_some();
        let units = |v: f64| (v * f64::from(IMAGE_EXTENT)).round() as i16 as u16 as u32;
        let position = |v: f64| ((v * f64::from(IMAGE_EXTENT)) as f32).to_bits();
        let vertices = mesh
            .positions
            .iter()
            .zip(&mesh.sources)
            .enumerate()
            .flat_map(|(k, (p, s))| {
                let weight = weights.map_or(1.0, |w| w[k]);
                let image = units(s.x) | units(s.y) << 16;
                // Clipping is in the image's tile units, but an image
                // source's grid sits in its tile's own units.
                let clip = if quad {
                    units(p.x) | units(p.y) << 16
                } else {
                    image
                };
                [position(p.x), position(p.y), clip, weight.to_bits(), image]
            })
            .collect();
        let indices: Vec<u32> = mesh.triangles.iter().flatten().copied().collect();
        let point = |k: u32| mesh.positions[k as usize] * f64::from(IMAGE_EXTENT);
        let chunks = mesh
            .triangles
            .chunks(256)
            .enumerate()
            .map(|(c, triangles)| {
                let mut min = glam::DVec2::splat(f64::INFINITY);
                let mut max = glam::DVec2::splat(f64::NEG_INFINITY);
                for p in triangles.iter().flatten().map(|&k| point(k)) {
                    min = min.min(p);
                    max = max.max(p);
                }
                // Bounds round outward, so the cull keeps every pixel.
                let min = [min.x.floor() as i16, min.y.floor() as i16];
                let max = [max.x.ceil() as i16, max.y.ceil() as i16];
                Chunk {
                    sort_key: None,
                    first_index: (c * 256 * 3) as u32,
                    index_count: (triangles.len() * 3) as u32,
                    bounds: TileBox { min, max },
                }
            })
            .collect();
        Self {
            frame: mesh.frame,
            vertices,
            indices,
            chunks,
        }
    }
}

/// An image tile to upload.
pub(crate) struct Upload<'a> {
    pub(crate) plan: &'a StylePlan,
    pub(crate) source: usize,
    pub(crate) kind: ImageKind,
    pub(crate) tile: TileIndex,
    pub(crate) image: &'a TileImage,
    /// A DEM source's unpack vector.
    pub(crate) unpack: Option<[f64; 4]>,
    /// The tile's mesh on the surface, from a `RasterWarp` plan.
    pub(crate) warped: Option<Arc<Warped>>,
    /// The style zoom it draws at (decision 0022).
    pub(crate) zoom: f64,
    /// Now, in the frame's milliseconds, and whether the tile fades in
    /// (a raster source with a fade duration).
    pub(crate) time: f32,
    pub(crate) fades: bool,
}

/// How many zooms up a fading tile looks for the ancestor it cross-fades
/// from: the row packs the tile's offsets in it in 12 bits.
const FADE_DEPTH: u8 = 12;

/// How a tile arriving at `time` fades in, by what of its source was
/// resident (GL JS's `getFadeValues`, its tile retention reduced to what
/// the pyramid holds): where descendants covered it (zooming out) it shows
/// at once; otherwise it cross-fades from the nearest resident ancestor, or
/// without one fades in.
fn fade_in(pyramid: &TilePyramid<ResidentTile>, tile: TileIndex, time: f32) -> ImageFade {
    let image = |t: TileIndex| pyramid.get(t).and_then(|r| r.image.as_ref());
    let children = tile.children();
    if children
        .iter()
        .any(|&c| image(c).is_some() || c.children().iter().any(|&g| image(g).is_some()))
    {
        return ImageFade::NONE;
    }
    let mut ancestor = tile.parent();
    let mut depth = 1;
    while let Some(at) = ancestor.filter(|_| depth <= FADE_DEPTH) {
        if let Some(image) = image(at) {
            let offset = |v: u32| (v - (v >> depth << depth)) as u16;
            return ImageFade {
                arrived: time,
                ancestor: Some((image.layer, depth, [offset(tile.col), offset(tile.row)])),
            };
        }
        ancestor = at.parent();
        depth += 1;
    }
    ImageFade {
        arrived: time,
        ancestor: None,
    }
}

/// The quad an image tile on its own plane draws: its corners in tile
/// units, as f32 positions and as image units (the same for a quad).
const QUAD: [u32; 20] = {
    let e = IMAGE_EXTENT;
    let (zero, full, one) = (0f32.to_bits(), (e as f32).to_bits(), 1f32.to_bits());
    [
        zero,
        zero,
        0,
        one,
        0, //
        full,
        zero,
        e,
        one,
        e, //
        zero,
        full,
        e << 16,
        one,
        e << 16, //
        full,
        full,
        e | e << 16,
        one,
        e | e << 16,
    ]
};
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 2, 1, 3];
const QUAD_CHUNK: Chunk = Chunk {
    sort_key: None,
    first_index: 0,
    index_count: 6,
    bounds: TileBox {
        min: [0, 0],
        max: [IMAGE_EXTENT as i16, IMAGE_EXTENT as i16],
    },
};

/// Why an image tile didn't upload.
#[derive(Debug)]
pub(crate) enum UploadError {
    /// Out of layers in the page: evicting tiles, or growing the page, may
    /// make room.
    NoLayer,
    /// Out of scene memory: evicting tiles may make room.
    NoRoom,
    /// The tile itself can't draw.
    Invalid(String),
}

/// What an image tile's quads (or warped meshes) take in the scene: one
/// per node of its source.
pub(crate) fn mesh_sizes(
    plan: &StylePlan,
    source: usize,
    warped: Option<&Warped>,
) -> Vec<MeshSize> {
    let size = match warped {
        Some(warped) => MeshSize {
            vertices: warped.vertices.len() as u32,
            indices: warped.indices.len() as u32,
            props: IMAGE_ROW_WORDS as u32,
            chunks: warped.chunks.len() as u32,
        },
        None => MeshSize {
            vertices: QUAD.len() as u32,
            indices: QUAD_INDICES.len() as u32,
            props: IMAGE_ROW_WORDS as u32,
            chunks: 1,
        },
    };
    let nodes = plan.nodes.iter().filter(|n| n.source == source).count();
    vec![size; nodes]
}

/// Writes an image tile into its source's page, which the first tile
/// creates at its size, and adds its quad to every node of the source. A
/// DEM tile is queued for the prepare pass, and so are its resident
/// neighbours, whose borders it fills.
pub(crate) fn upload(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    page: &mut Option<PageId>,
    pyramid: &TilePyramid<ResidentTile>,
    upload: &Upload<'_>,
    grow_page: bool,
) -> Result<ResidentTile, UploadError> {
    let Upload {
        plan,
        source,
        kind,
        tile,
        image,
        ..
    } = *upload;
    let size = [image.width, image.height];
    if kind == ImageKind::Dem && image.width != image.height {
        return Err(UploadError::Invalid("DEM tiles must be square".into()));
    }
    let page = *page.get_or_insert_with(|| {
        let kind = match kind {
            ImageKind::Raster => PageKind::Raster,
            ImageKind::Dem => PageKind::Dem,
        };
        let page = renderer.create_page(gpu, kind, size);
        for (index, node) in plan.nodes.iter().enumerate() {
            if node.source == source {
                renderer.bind_page(index as u32, page);
            }
        }
        page
    });
    let expected = renderer.page_size(page);
    if expected != size {
        return Err(UploadError::Invalid(format!(
            "a {}×{} image in a source of {}×{} tiles",
            size[0], size[1], expected[0], expected[1]
        )));
    }
    let layer = if grow_page {
        renderer.alloc_layer(gpu, page).ok()
    } else {
        renderer.alloc_existing_layer(page)
    }
    .ok_or(UploadError::NoLayer)?;
    let pixels = match &image.pixels {
        ImagePixels::Core(bytes) => Pixels::Core(bytes),
        &ImagePixels::Host(image) => Pixels::Host {
            image,
            premultiply: kind == ImageKind::Raster,
        },
    };
    renderer.write_layer(gpu, page, layer, pixels);

    let scene_tile = renderer.scene().insert_tile();
    let mut resident = ResidentTile {
        parts: vec![(IMAGE_EXTENT, scene_tile)],
        meshes: Vec::new(),
        series: None,
        warp: orb_render::TileWarp::NONE,
        zoom: upload.zoom,
        image: Some(ResidentImage {
            page,
            layer,
            warped: upload.warped.clone(),
            fade: if upload.fades {
                fade_in(pyramid, tile, upload.time)
            } else {
                ImageFade::NONE
            },
        }),
        symbols: Vec::new(),
        cost: 0,
    };
    for (index, node) in plan.nodes.iter().enumerate() {
        if node.source == source
            && add_node(gpu, renderer, &mut resident, tile, index as u32, node.key).is_err()
        {
            renderer.scene().remove_tile(scene_tile);
            renderer.free_layer(page, layer);
            return Err(UploadError::NoRoom);
        }
    }

    if let Some(unpack) = upload.unpack {
        let layer_of = |t: TileIndex| {
            pyramid
                .get(t)
                .and_then(|r| r.image.as_ref())
                .map(|i| i.layer)
        };
        for job in dem_prepares(tile, layer, unpack.map(|v| v as f32), &layer_of) {
            renderer.prepare_dem(page, job);
        }
    }
    Ok(resident)
}

/// The prepare jobs a DEM tile's arrival at `layer` starts: its own, and
/// its resident neighbours', whose borders it fills. `layer_of` finds a
/// resident tile's layer.
pub(crate) fn dem_prepares(
    tile: TileIndex,
    layer: u32,
    unpack: [f32; 4],
    layer_of: &dyn Fn(TileIndex) -> Option<u32>,
) -> Vec<DemPrepare> {
    let arrived = (tile, layer);
    let mut jobs = vec![DemPrepare {
        layer,
        neighbours: neighbours(layer_of, tile, arrived),
        zoom: f32::from(tile.z),
        unpack,
    }];
    for offset in OFFSETS {
        let Some(neighbour) = offset_tile(tile, offset) else {
            continue;
        };
        if let Some(layer) = layer_of(neighbour) {
            jobs.push(DemPrepare {
                layer,
                neighbours: neighbours(layer_of, neighbour, arrived),
                zoom: f32::from(neighbour.z),
                unpack,
            });
        }
    }
    jobs
}

/// Adds the quad of render node `index` (key `key`) to an image tile.
pub(crate) fn add_node(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    resident: &mut ResidentTile,
    tile: TileIndex,
    index: u32,
    key: u32,
) -> Result<(), orb_render::OutOfSpace> {
    let image = resident.image.as_ref().expect("image tiles hold an image");
    let part = resident.parts[0].1;
    let [top, bottom] = [tile.row, tile.row + 1].map(|row| latitude(tile.z, row));
    let mut row = [0; IMAGE_ROW_WORDS];
    row[..3].copy_from_slice(&[image.layer, top.to_bits(), bottom.to_bits()]);
    row[3..].copy_from_slice(&image.fade.words());
    let mesh = match &image.warped {
        Some(warped) => MeshData {
            vertices: Words::Core(&warped.vertices),
            indices: Words::Core(&warped.indices),
            props: Words::Core(&row),
            row_words: IMAGE_ROW_WORDS as u32,
            chunks: &warped.chunks,
        },
        None => MeshData {
            vertices: Words::Core(&QUAD),
            indices: Words::Core(&QUAD_INDICES),
            props: Words::Core(&row),
            row_words: IMAGE_ROW_WORDS as u32,
            chunks: &[QUAD_CHUNK],
        },
    };
    let mesh = renderer.scene().insert_mesh(gpu, part, index, &mesh)?;
    resident.meshes.push(Resident {
        hits: Default::default(),
        node: key,
        mesh,
        layout_rev: 0,
        rows_rev: 0,
        ramps: Vec::new(),
        states: Vec::new(),
    });
    Ok(())
}

/// Neighbour offsets in [`DemPrepare::neighbours`] order.
const OFFSETS: [(i32, i32); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// The tile `offset` away at the same zoom, wrapping east and west as GL
/// JS's `_getNeighboringTiles` does; `None` past the poles.
fn offset_tile(tile: TileIndex, (dx, dy): (i32, i32)) -> Option<TileIndex> {
    let n = 1i64 << tile.z;
    let row = i64::from(tile.row) + i64::from(dy);
    if !(0..n).contains(&row) {
        return None;
    }
    let col = (i64::from(tile.col) + i64::from(dx)).rem_euclid(n);
    Some(TileIndex::new(tile.z, col as u32, row as u32))
}

/// The layers of `tile`'s resident neighbours, counting `arrived`, a tile
/// not yet in the pyramid.
fn neighbours(
    layer_of: &dyn Fn(TileIndex) -> Option<u32>,
    tile: TileIndex,
    arrived: (TileIndex, u32),
) -> [Option<u32>; 8] {
    OFFSETS.map(|offset| {
        let neighbour = offset_tile(tile, offset)?;
        if neighbour == arrived.0 {
            return Some(arrived.1);
        }
        layer_of(neighbour)
    })
}

/// The latitude of a row edge of the Web Mercator tile grid, in degrees.
fn latitude(z: u8, row: u32) -> f32 {
    let n = (1u64 << z) as f64;
    ((1.0 - 2.0 * f64::from(row) / n) * std::f64::consts::PI)
        .sinh()
        .atan()
        .to_degrees() as f32
}

/// An image node's paint at `zoom` with the map rotated by `bearing`
/// degrees. A color-relief node's ramp is written first, once its source's
/// `unpack` vector is known; until then it has no paint.
pub(crate) fn paint(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    (plan, node, index): (&StylePlan, &NodePlan, u32),
    zoom: f64,
    bearing: f64,
    unpack: Option<[f64; 4]>,
    ramp: &mut Option<u32>,
) -> Option<NodePaint> {
    let nearest = |r: &Property<Resampling>| r.evaluate(zoom) == Resampling::Nearest;
    Some(match &plan.style.layers[node.layer].kind {
        LayerKind::Raster(raster) => {
            let p = &raster.paint;
            let image_source = matches!(
                plan.source_def(node.source).kind,
                orb_v8::SourceKind::Image(_)
            );
            NodePaint::Raster(RasterPaint {
                opacity: p.opacity.evaluate(zoom),
                hue_rotate: p.hue_rotate.evaluate(zoom),
                brightness_min: p.brightness_min.evaluate(zoom),
                brightness_max: p.brightness_max.evaluate(zoom),
                saturation: p.saturation.evaluate(zoom),
                contrast: p.contrast.evaluate(zoom),
                nearest: nearest(&p.resampling) || nearest(&p.raster_resampling),
                // Image sources are one texture without mip levels in GL JS,
                // and don't fade.
                mipmapped: !image_source,
                fade_duration: if image_source {
                    0.0
                } else {
                    p.fade_duration.evaluate(zoom).max(0.0)
                },
            })
        }
        LayerKind::Hillshade(hillshade) => {
            let p = &hillshade.paint;
            let lights = [
                p.illumination_direction.len(),
                p.illumination_altitude.len(),
                p.shadow_color.len(),
                p.highlight_color.len(),
            ]
            .into_iter()
            .max()
            .unwrap_or(1)
            .clamp(1, MAX_LIGHTS);
            // GL JS repeats each array's last value up to the longest.
            fn per_light<T: PropertyValue, U>(
                values: &[Property<T>],
                zoom: f64,
                f: impl Fn(T) -> U,
            ) -> [U; MAX_LIGHTS] {
                std::array::from_fn(|i| f(values[i.min(values.len() - 1)].evaluate(zoom)))
            }
            let turn = if p.illumination_anchor.evaluate(zoom) == IlluminationAnchor::Viewport {
                bearing.to_radians() as f32
            } else {
                0.0
            };
            let color = |c: Color| c.premultiplied();
            NodePaint::Hillshade(HillshadePaint {
                accent: p.accent_color.evaluate(zoom).premultiplied(),
                shadows: per_light(&p.shadow_color, zoom, color),
                highlights: per_light(&p.highlight_color, zoom, color),
                altitudes: per_light(&p.illumination_altitude, zoom, f32::to_radians),
                azimuths: per_light(&p.illumination_direction, zoom, |d| d.to_radians() + turn),
                exaggeration: p.exaggeration.evaluate(zoom),
                method: match p.method.evaluate(zoom) {
                    orb_v8::HillshadeMethod::Standard => HillshadeMethod::Standard,
                    orb_v8::HillshadeMethod::Basic => HillshadeMethod::Basic,
                    orb_v8::HillshadeMethod::Combined => HillshadeMethod::Combined,
                    orb_v8::HillshadeMethod::Igor => HillshadeMethod::Igor,
                    orb_v8::HillshadeMethod::Multidirectional => HillshadeMethod::Multidirectional,
                },
                lights: lights as u32,
                nearest: nearest(&p.resampling),
            })
        }
        LayerKind::ColorRelief(relief) => {
            let unpack = unpack?;
            let size = match *ramp {
                Some(size) => size,
                None => {
                    let max = gpu.caps().limits.max_texture_dimension_2d as usize;
                    let (colors, elevations) = color_ramp(relief.paint.color.as_ref(), unpack, max);
                    renderer.set_ramp(gpu, index, &colors, &elevations);
                    *ramp.insert(colors.len() as u32)
                }
            };
            NodePaint::ColorRelief(ColorReliefPaint {
                opacity: relief.paint.opacity.evaluate(zoom),
                unpack: unpack.map(|v| v as f32),
                ramp_size: size,
                nearest: nearest(&relief.paint.resampling),
            })
        }
        _ => unreachable!("image nodes are raster, hillshade or color-relief layers"),
    })
}

/// GL JS's `_createColorRamp` and `getColorRampTextures`: per stop, its
/// colour as the texel GL JS uploads (unpremultiplied to bytes, then
/// premultiplied again) and its elevation packed with `unpack`. A ramp
/// that isn't a top-level `interpolate` is transparent; ramps longer than
/// `max` keep `max` evenly spaced stops.
fn color_ramp(
    ramp: Option<&orb_v8::PropertyExpression>,
    unpack: [f64; 4],
    max: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut stops: Vec<(f64, [f32; 4])> = Vec::new();
    if let Some(expression) = ramp.map(|r| r.expression())
        && let Some(labels) = expression.interpolate_labels()
    {
        for &label in labels {
            let context = orb_expr::EvaluationContext {
                elevation: label,
                ..orb_expr::EvaluationContext::default()
            };
            let color = expression
                .evaluate_with(&context)
                .ok()
                .and_then(|v| v.as_color())
                .map_or([0.0; 4], |c| c.premultiplied());
            stops.push((label, color));
        }
    }
    if stops.is_empty() {
        stops.push((0.0, [0.0; 4]));
    }
    if stops.len() < 2 {
        stops.push((stops[0].0 + 1.0, stops[0].1));
    }
    if stops.len() > max {
        let step = (stops.len() - 1) as f64 / (max - 1) as f64;
        let mut remapped = Vec::with_capacity(max);
        let mut i = 0.0;
        while i < stops.len() as f64 - 0.5 {
            remapped.push(stops[i.round() as usize]);
            i += step;
        }
        stops = remapped;
    }
    let texel = |[r, g, b, a]: [f32; 4]| {
        let alpha = (a * 255.0).round();
        let channel = |c: f32| {
            let straight = if a > 0.0 {
                (c * 255.0 / a).round()
            } else {
                0.0
            };
            (straight * alpha / 255.0).round() as u32
        };
        channel(r) | channel(g) << 8 | channel(b) << 16 | (alpha as u32) << 24
    };
    stops
        .iter()
        .map(|&(elevation, color)| {
            let [r, g, b] = pack_dem(elevation, unpack);
            (texel(color), r | g << 8 | b << 16 | 0xff << 24)
        })
        .unzip()
}

/// GL JS's `packDEMData`: an elevation as the RGB bytes `unpack` decodes.
/// Channels whose factor is 0 pack to 0.
fn pack_dem(elevation: f64, [red, green, blue, base_shift]: [f64; 4]) -> [u32; 3] {
    let factors = [red, green, blue];
    let Some(min_scale) = factors
        .into_iter()
        .filter(|&f| f != 0.0)
        .min_by(f64::total_cmp)
    else {
        return [0; 3];
    };
    let scaled = ((elevation + base_shift) / min_scale).round();
    factors.map(|factor| {
        if factor == 0.0 {
            0
        } else {
            ((scaled * min_scale / factor).floor() as i64).rem_euclid(256) as u32
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_elevations_like_gl_js() {
        let mapbox = orb_v8::DemEncoding::Mapbox.unpack([0.0; 4]);
        // (v + 10000) / 0.1 = 110000 = 1·65536 + 173·256 + 176.
        assert_eq!(pack_dem(1000.0, mapbox), [1, 173, 176]);
        assert_eq!(pack_dem(5.0, [0.0, 0.0, 0.0, 1.0]), [0, 0, 0]);
        // A zero factor packs to 0 and leaves the others' scale alone.
        assert_eq!(pack_dem(1000.0, [0.0, 25.6, 0.1, 3446.4]), [0, 173, 176]);
    }

    #[test]
    fn ramps_pad_and_quantize_like_gl_js() {
        let (colors, elevations) = color_ramp(None, [6553.6, 25.6, 0.1, 10000.0], 8192);
        assert_eq!(colors, [0, 0]);
        assert_eq!(elevations.len(), 2);
        let json = orb_expr::json::parse(
            r##"["interpolate", ["linear"], ["elevation"], 0, "rgba(255, 0, 0, 0.5)", 100, "#00f"]"##,
        )
        .unwrap();
        let spec = orb_expr::PropertySpec {
            ty: orb_expr::PropertyType::Color,
            default: None,
            zoom: false,
            feature: false,
            interpolated: true,
            tokens: false,
        };
        let ramp = orb_v8::PropertyExpression::parse(&json, &spec).unwrap();
        let (colors, _) = color_ramp(Some(&ramp), [6553.6, 25.6, 0.1, 10000.0], 8192);
        // Half-transparent red: 255 straight, alpha 128, premultiplied 128.
        assert_eq!(colors, [128 | 128 << 24, 0xffff_0000]);
    }

    #[test]
    fn neighbours_wrap_east_and_west() {
        let tile = TileIndex::new(2, 0, 0);
        assert_eq!(offset_tile(tile, (-1, 0)), Some(TileIndex::new(2, 3, 0)));
        assert_eq!(offset_tile(tile, (0, -1)), None);
        assert_eq!(offset_tile(tile, (1, 1)), Some(TileIndex::new(2, 1, 1)));
    }
}
