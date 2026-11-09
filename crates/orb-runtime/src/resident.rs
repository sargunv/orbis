//! Tiles in the scene: what the map keeps of each beside the renderer's
//! handle, their upload with their rows and labels, and what they cost.

use std::collections::HashMap;
use std::sync::Arc;

use orb_crs::TileIndex;
use orb_hal::{Gpu, HostBytesId};
use orb_render::{
    GpuTile, ImageAtlas, MeshData, MeshId, MeshSize, OutOfSpace, Renderer, TileKey, Words,
};
use orb_source::TilePyramid;
use orb_source::endpoint::TileEndpoint;

use crate::MapEvent;
use crate::image::ResidentImage;
use crate::jobs::{ResultMesh, Stream, TileResult};
use crate::map::NodeState;
use crate::state::FeatureStates;
use crate::style::StylePlan;

/// A tile in the scene: one scene tile per layer extent holding a mesh per
/// render node with features, and an image tile's texture layer.
#[derive(Debug)]
pub(crate) struct ResidentTile {
    /// What the renderer holds of it: its meshes in scene tiles, its
    /// patches, an image tile's layer.
    pub(crate) gpu: GpuTile,
    /// What the map keeps of each mesh, symbol meshes included.
    pub(crate) meshes: Vec<Resident>,
    pub(crate) image: Option<ResidentImage>,
    /// A tile drawn through a warp's patches: its series.
    pub(crate) series: Option<Arc<orb_crs::Series>>,
    /// The style zoom it laid out at.
    pub(crate) zoom: f64,
    /// Its symbol layers' buckets; their meshes are among `meshes`.
    pub(crate) symbols: Vec<crate::labels::ResidentSymbols>,
    /// What it costs to keep: GPU bytes of its meshes and texture layer,
    /// and its picking data (see [`tile_cost`]).
    pub(crate) cost: u64,
}

impl ResidentTile {
    pub(crate) fn new(gpu: GpuTile, zoom: f64) -> Self {
        Self {
            gpu,
            meshes: Vec::new(),
            image: None,
            series: None,
            zoom,
            symbols: Vec::new(),
            cost: 0,
        }
    }
}

/// One node's mesh of a resident tile.
#[derive(Debug)]
pub(crate) struct Resident {
    /// The node's key.
    pub(crate) node: u32,
    pub(crate) mesh: MeshId,
    pub(crate) hits: Arc<crate::query::MeshHits>,
    /// The style revisions its geometry and its rows were laid out with.
    pub(crate) layout_rev: u64,
    pub(crate) rows_rev: u64,
    /// Gradient ramps its rows retain.
    pub(crate) ramps: Vec<u32>,
    /// The feature-state indices its features refer to.
    pub(crate) states: Vec<u32>,
}

#[derive(Debug)]
pub(crate) enum SourceState {
    Resolving,
    Ready {
        endpoint: Arc<TileEndpoint>,
        pyramid: TilePyramid<ResidentTile>,
        loading: u32,
        /// Its tiles are on another plane than the surface's, and reach
        /// it through their warp (`Map::globe_warps`).
        warped: bool,
    },
    Failed,
}

/// The host data a tile result holds until it's uploaded or dropped.
#[derive(Clone, Copy)]
pub(crate) struct HostData(Option<HostBytesId>, Option<orb_hal::HostImageId>);

impl HostData {
    pub(crate) fn of(result: &TileResult) -> Self {
        let image = match &result.image {
            Some(crate::TileImage {
                pixels: crate::ImagePixels::Host(image),
                ..
            }) => Some(*image),
            _ => None,
        };
        Self(result.host_bytes, image)
    }

    pub(crate) fn release(self, gpu: &mut dyn Gpu) {
        self.0
            .into_iter()
            .for_each(|bytes| gpu.release_host_bytes(bytes));
        self.1
            .into_iter()
            .for_each(|image| gpu.release_host_image(image));
    }
}

/// A tile result waiting for room in the scene, or for the sprites' pixels
/// its labels' icons need (`Map::apply_tile`).
#[derive(Debug)]
pub(crate) struct WaitingTile {
    pub(crate) generation: u64,
    /// The plan source's key.
    pub(crate) source: u32,
    pub(crate) tile: TileIndex,
    pub(crate) build: TileResult,
    /// [`Map::room_state`] when it was last tried.
    pub(crate) tried: Option<[u64; 7]>,
}

/// Raises the nodes' row maxima to those of new meshes; paint is written
/// again when they grow.
pub(crate) fn record_maxima(
    nodes: &mut HashMap<u32, NodeState>,
    meshes: &[ResultMesh],
    paint: &mut crate::paint::PaintState,
) {
    for mesh in meshes {
        let Some(node) = nodes.get_mut(&mesh.node) else {
            continue;
        };
        if node.maxima.len() < mesh.maxima.len() {
            node.maxima.resize(mesh.maxima.len(), 0.0);
        }
        for (max, &value) in node.maxima.iter_mut().zip(&mesh.maxima) {
            if value > *max {
                *max = value;
                paint.invalidate();
            }
        }
    }
}

/// A source whose endpoint resolved; `image` sources cache fewer tiles.
pub(crate) fn ready_source(
    endpoint: Result<(TileEndpoint, bool), String>,
    id: &str,
    image: bool,
    events: &mut crate::events::EventQueue,
) -> SourceState {
    match endpoint {
        Ok((endpoint, warped)) => SourceState::Ready {
            warped,
            pyramid: TilePyramid::new(
                endpoint.minzoom,
                if image {
                    endpoint.maxzoom
                } else {
                    endpoint.cover_zoom()
                },
                // Map::trim_cache bounds the cache by memory.
                usize::MAX,
            ),
            endpoint: Arc::new(endpoint),
            loading: 0,
        },
        Err(e) => {
            events.push(MapEvent::Error {
                message: format!("source {id}: {e}"),
            });
            SourceState::Failed
        }
    }
}

/// Drops a result's meshes and symbol buckets of nodes no longer planned
/// or laid out since (`stale`, by key); a symbol layer's bucket goes with
/// its nodes.
pub(crate) fn drop_stale(build: &mut TileResult, plan: &StylePlan, stale: &[u32]) {
    build
        .meshes
        .retain(|m| plan.node(m.node).is_some() && !stale.contains(&m.node));
    build.symbols.retain(|s| {
        let keys = [
            crate::style::node_key(s.layer, 0),
            crate::style::node_key(s.layer, 1),
        ];
        plan.node(keys[0]).is_some() && !keys.iter().any(|k| stale.contains(k))
    });
}

pub(crate) fn words(stream: &Stream, bytes: Option<HostBytesId>) -> Words<'_> {
    match stream {
        Stream::Core(words) => Words::Core(words),
        Stream::Host { offset, len } => Words::Host {
            bytes: bytes.expect("host streams come with host bytes"),
            offset: *offset,
            len: *len,
        },
    }
}

/// What a result mesh's rows refer to besides their words: the images
/// patched into them, the ramps among those (retained), and the
/// feature-state indices its features refer to (interned).
pub(crate) struct ResolvedRows {
    pub(crate) images: Vec<(u32, u32)>,
    pub(crate) ramps: Vec<u32>,
    pub(crate) states: Vec<u32>,
}

impl ResolvedRows {
    pub(crate) fn rows(&self) -> orb_render::MeshRows<'_> {
        orb_render::MeshRows {
            images: &self.images,
            states: &self.states,
        }
    }

    /// Hands back what [`resolve_rows`] retained, for rows that didn't
    /// upload.
    pub(crate) fn release(
        self,
        gpu: &mut dyn Gpu,
        atlas: &mut ImageAtlas,
        states: &mut FeatureStates,
    ) {
        for index in self.states {
            states.release(index);
        }
        atlas.release_ramps(gpu, self.ramps);
    }
}

/// Resolves a result mesh's images in the atlas and interns its features'
/// states.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_rows(
    gpu: &mut dyn Gpu,
    plan: &StylePlan,
    result: &TileResult,
    mesh: &ResultMesh,
    atlas: &mut ImageAtlas,
    namespace: u64,
    states: &mut FeatureStates,
    warnings: &mut Vec<String>,
) -> ResolvedRows {
    let mut images = Vec::new();
    let mut ramps = Vec::new();
    for image in &mesh.images {
        match atlas.resolve_key(gpu, namespace, &image.key) {
            Ok(image_id) => {
                if matches!(image.key, orb_render::ImageKey::Ramp { .. }) {
                    atlas.retain_ramp(image_id);
                    ramps.push(image_id);
                }
                images.push((image.word, image_id));
            }
            Err(e) => warnings.push(e.into()),
        }
    }
    let mut indices = Vec::new();
    if let (Some(ids), Some(node)) = (&mesh.state_ids, plan.node(mesh.node)) {
        let node = &plan.nodes[node];
        let source = plan.sources[node.source].key;
        let layer = plan.source_layer(node);
        indices = ids
            .iter()
            .enumerate()
            .map(|(slot, &i)| match result.ids.get(i as usize) {
                Some(id) => states.intern(source, layer, id, mesh.data.get(slot)),
                None => 0,
            })
            .collect();
    }
    ResolvedRows {
        images,
        ramps,
        states: indices,
    }
}

/// Uploads one result mesh into a resident tile, in the part of its
/// extent.
#[allow(clippy::too_many_arguments)]
pub(crate) fn upload_mesh(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    plan: &StylePlan,
    result: &TileResult,
    mesh: &ResultMesh,
    resident: &mut ResidentTile,
    atlas: &mut ImageAtlas,
    namespace: u64,
    states: &mut FeatureStates,
    warnings: &mut Vec<String>,
) -> Result<(), OutOfSpace> {
    let node = plan.node(mesh.node).expect("meshes of planned nodes") as u32;
    let data = MeshData {
        vertices: words(&mesh.vertices, result.host_bytes),
        indices: words(&mesh.indices, result.host_bytes),
        props: words(&mesh.props, result.host_bytes),
        row_words: mesh.row_words,
        chunks: &mesh.chunks,
    };
    let rows = resolve_rows(gpu, plan, result, mesh, atlas, namespace, states, warnings);
    let added = renderer.add_mesh(gpu, resident.gpu, (node, mesh.extent), &data, &rows.rows());
    let id = match added {
        Ok(id) => id,
        Err(e) => {
            rows.release(gpu, atlas, states);
            return Err(e);
        }
    };
    resident.meshes.push(Resident {
        hits: mesh.hits.clone(),
        node: mesh.node,
        mesh: id,
        layout_rev: result.revision,
        rows_rev: result.revision,
        ramps: rows.ramps,
        states: rows.states,
    });
    Ok(())
}

/// Uploads a tile's meshes and symbol buckets, drawing through `patches`:
/// all of them or, when the scene is out of space, none.
#[allow(clippy::too_many_arguments)]
pub(crate) fn upload(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    plan: &StylePlan,
    (key, result, patches): (TileKey, &TileResult, Option<(u32, &[f32])>),
    atlas: &mut ImageAtlas,
    namespace: u64,
    states: &mut FeatureStates,
    labels: &mut crate::labels::Labels,
    tile_zoom: f64,
    warnings: &mut Vec<String>,
) -> Result<ResidentTile, OutOfSpace> {
    // Symbol nodes' rows go up with their layer's bucket.
    let meshes: Vec<&ResultMesh> = result
        .meshes
        .iter()
        .filter(|m| {
            !plan
                .node(m.node)
                .is_some_and(|i| plan.nodes[i].family.is_symbol())
        })
        .collect();
    let rows: Vec<ResolvedRows> = meshes
        .iter()
        .map(|m| resolve_rows(gpu, plan, result, m, atlas, namespace, states, warnings))
        .collect();
    let builds: Vec<orb_render::MeshBuild> = meshes
        .iter()
        .zip(&rows)
        .map(|(m, rows)| orb_render::MeshBuild {
            node: plan.node(m.node).expect("meshes of planned nodes") as u32,
            extent: m.extent,
            data: MeshData {
                vertices: words(&m.vertices, result.host_bytes),
                indices: words(&m.indices, result.host_bytes),
                props: words(&m.props, result.host_bytes),
                row_words: m.row_words,
                chunks: &m.chunks,
            },
            rows: rows.rows(),
        })
        .collect();
    let build = orb_render::TileBuild {
        key,
        meshes: &builds,
        patches,
    };
    let (handle, ids) = match renderer.upload_tile(gpu, &build, false) {
        Ok(uploaded) => uploaded,
        Err(_) => {
            for rows in rows {
                rows.release(gpu, atlas, states);
            }
            return Err(OutOfSpace);
        }
    };
    let mut resident = ResidentTile::new(handle, tile_zoom);
    resident.meshes = meshes
        .iter()
        .zip(rows)
        .zip(ids)
        .map(|((m, rows), mesh)| Resident {
            hits: m.hits.clone(),
            node: m.node,
            mesh,
            layout_rev: result.revision,
            rows_rev: result.revision,
            ramps: rows.ramps,
            states: rows.states,
        })
        .collect();
    if let Err(e) = labels.upload(
        gpu,
        renderer,
        plan,
        result,
        &mut resident,
        atlas,
        namespace,
        states,
        tile_zoom,
    ) {
        atlas.release_ramps(gpu, remove_resident(renderer, resident, states));
        return Err(e);
    }
    Ok(resident)
}

/// Evicts the least recently used cached tile the view doesn't use, of
/// source `first` before others'.
pub(crate) fn evict_unused(sources: &mut [SourceState], first: usize) -> Option<ResidentTile> {
    let others = (0..sources.len()).filter(|&s| s != first);
    std::iter::once(first)
        .chain(others)
        .find_map(|s| match &mut sources[s] {
            SourceState::Ready { pyramid, .. } => pyramid.evict_lru(),
            _ => None,
        })
}

/// Evicts a tile that only stands in for a tile of a cover ([`TilePyramid::
/// evict_stand_in`]), `first`'s first and the ancestors of its `tile`
/// before others, to make room for `tile`.
pub(crate) fn evict_stand_in(
    sources: &mut [SourceState],
    first: usize,
    tile: TileIndex,
) -> Option<ResidentTile> {
    let others = (0..sources.len()).filter(|&s| s != first);
    std::iter::once(first)
        .chain(others)
        .find_map(|s| match &mut sources[s] {
            SourceState::Ready { pyramid, .. } => {
                pyramid.evict_stand_in((s == first).then_some(tile))
            }
            _ => None,
        })
}

/// What a tile result's meshes take in the scene, as [`upload`] inserts
/// them.
pub(crate) fn mesh_sizes(plan: &StylePlan, result: &TileResult) -> Vec<MeshSize> {
    let words = |stream: &Stream| match stream {
        Stream::Core(words) => words.len() as u32,
        Stream::Host { len, .. } => *len,
    };
    result
        .meshes
        .iter()
        .filter(|mesh| {
            !plan
                .node(mesh.node)
                .is_some_and(|i| plan.nodes[i].family.is_symbol())
        })
        .map(|mesh| MeshSize {
            vertices: words(&mesh.vertices),
            indices: words(&mesh.indices),
            props: words(&mesh.props),
            chunks: mesh.chunks.len() as u32,
        })
        .collect()
}

/// Whether the map's GPU memory (the scene's pools and tables, its
/// textures, and the image atlas's `atlas` bytes) is under `budget`: the
/// pools grow for arriving tiles before cached ones go, and the cache is
/// held to its share of the budget by `trim_cache`.
pub(crate) fn under_budget(renderer: &Renderer, atlas: u64, budget: u64) -> bool {
    let m = renderer.memory();
    let (tiles, glyphs) = renderer.texture_bytes();
    let allocated = m.vertices.allocated
        + m.indices.allocated
        + m.compacted.allocated
        + m.props.allocated
        + m.tables
        + tiles
        + glyphs
        + atlas;
    allocated < budget
}

/// What a resident tile costs to keep: the GPU bytes of its meshes and its
/// texture layer, and its picking data.
pub(crate) fn tile_cost(renderer: &Renderer, resident: &ResidentTile) -> u64 {
    let meshes: u64 = resident
        .meshes
        .iter()
        .map(|m| renderer.mesh_bytes(m.mesh) + m.hits.bytes())
        .sum();
    meshes + renderer.tile_bytes(resident.gpu)
}

/// Removes a tile from the scene. Returns the ramps its rows retained.
pub(crate) fn remove_resident(
    renderer: &mut Renderer,
    tile: ResidentTile,
    states: &mut FeatureStates,
) -> Vec<u32> {
    renderer.release_tile(tile.gpu);
    release_resident(tile, states)
}

/// Releases what a tile holds on the CPU, the feature states its features
/// refer to; returns the ramps its rows retained.
pub(crate) fn release_resident(tile: ResidentTile, states: &mut FeatureStates) -> Vec<u32> {
    let mut ramps = Vec::new();
    for mut mesh in tile.meshes {
        for index in mesh.states {
            states.release(index);
        }
        ramps.append(&mut mesh.ramps);
    }
    ramps
}
