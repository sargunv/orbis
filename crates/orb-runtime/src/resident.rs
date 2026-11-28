//! Tiles in the scene: what the map keeps of each beside the renderer's
//! handle, their upload with their rows and labels, and what they cost.

use rustc_hash::FxHashMap as HashMap;
use std::sync::Arc;

use orb_crs::TileIndex;
use orb_hal::{Gpu, HostBytesId};
use orb_render::{GpuTile, ImageAtlas, MeshData, MeshId, OutOfSpace, Renderer, TileKey};
use orb_source::endpoint::TileEndpoint;
use orb_source::{Backoff, Draw, TilePyramid};

use crate::MapEvent;
use crate::image::ResidentImage;
use crate::jobs::{ResultMesh, TileResult};
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
    /// Laid out as a neighbour of the tile Tile mode draws: only what
    /// reaches into the tiles beside it. Drawn, it loads again whole.
    pub(crate) neighbour: bool,
    /// Its labels drew glyphs of a range that failed: a glyph retry lays
    /// them out again ([`TileResult::glyphs_failed`]).
    pub(crate) glyphs_failed: bool,
    /// When the data it shows expires (Unix ms), if it came fresh
    /// ([`TileResult::fresh_until`]): drawn then, it loads again. Kept on
    /// the wall clock, which HTTP's lifetimes are measured on and which,
    /// unlike the monotonic one, runs on while the system sleeps.
    pub(crate) expires: Option<f64>,
    /// Its reloads since one last brought a lifetime: each counts as a
    /// failure from when it starts, so one that fails or never answers is
    /// tried again when the backoff is due, or when its server asked.
    pub(crate) reloads: Backoff,
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
            neighbour: false,
            glyphs_failed: false,
            expires: None,
            reloads: Backoff::default(),
        }
    }

    /// The data it shows came fresh until `fresh_until` (Unix seconds) if
    /// that's given, or for good: its reloads start over.
    pub(crate) fn renew(&mut self, fresh_until: Option<u64>) {
        self.expires = fresh_until.map(|until| until as f64 * 1000.0);
        self.reloads.succeeded();
    }

    /// When (ms on the monotonic clock) it loads again if drawn: when a
    /// reload is due again, while one is under way or failed, else when
    /// its data expires, converted with the wall clock's `lead`.
    pub(crate) fn reload_due(&self, lead: &WallLead) -> Option<f64> {
        (self.reloads.due()).or(self.expires.map(|expires| lead.due(expires)))
    }
}

/// The wall clock's lead over the services' monotonic one (ms): what
/// converts when data expires, a moment on the wall clock that HTTP's
/// lifetimes are measured on, into a due time on the monotonic clock that
/// frames and the map's wake are measured on. Read each frame, the lead
/// moves by the reads' jitter (`Date.now()` counts whole milliseconds),
/// which it keeps out, so that one expiry is the same due time each frame
/// and a host keeps one timer for it; and by the system's sleeps and
/// clock changes, which the monotonic clock doesn't count, which it takes.
#[derive(Debug, Default)]
pub(crate) struct WallLead(Option<f64>);

impl WallLead {
    /// How far a reading strays from the lead kept before it's taken
    /// (ms): past any read's jitter (Firefox's fingerprinting resistance
    /// rounds its clocks to 100 ms) and what NTP's slewing, at most half a
    /// millisecond a second, moves in half an hour.
    const STRAY: f64 = 1000.0;

    /// Reads the clocks at `now` on the monotonic one and `unix` on the
    /// wall clock.
    pub(crate) fn read(&mut self, now: f64, unix: f64) {
        let lead = unix - now;
        if self.0.is_none_or(|kept| (lead - kept).abs() > Self::STRAY) {
            self.0 = Some(lead);
        }
    }

    /// When (monotonic ms) data expiring at `expires` (Unix ms) is due
    /// again: never before it expires, the lead kept being off by
    /// [`STRAY`](Self::STRAY) at most, nor more than twice that after.
    pub(crate) fn due(&self, expires: f64) -> f64 {
        expires - self.0.unwrap_or(0.0) + Self::STRAY
    }
}

/// One node's mesh of a resident tile.
#[derive(Debug)]
pub(crate) struct Resident {
    /// The node's key.
    pub(crate) node: u32,
    pub(crate) mesh: MeshId,
    pub(crate) hits: Arc<crate::query::MeshHits>,
    /// The style revision its geometry was laid out with.
    pub(crate) layout_rev: u64,
    /// Gradient ramps its rows retain.
    pub(crate) ramps: Vec<u32>,
    /// The feature-state indices its features refer to.
    pub(crate) states: Vec<u32>,
    /// Per paint property, the largest number in its rows (a tile's
    /// tallest extrusion, say), as laid out.
    pub(crate) maxima: Vec<f32>,
}

impl Resident {
    /// Result mesh `mesh` of a tile laid out at `revision`, uploaded as
    /// `id` with its resolved `rows`.
    fn new(mesh: &ResultMesh, id: MeshId, rows: ResolvedRows, revision: u64) -> Self {
        Self {
            hits: mesh.hits.clone(),
            node: mesh.node,
            mesh: id,
            layout_rev: revision,
            ramps: rows.ramps,
            states: rows.states,
            maxima: mesh.maxima.clone(),
        }
    }
}

#[derive(Debug)]
pub(crate) enum SourceState {
    Resolving,
    Ready {
        endpoint: Arc<TileEndpoint>,
        pyramid: TilePyramid<ResidentTile>,
        /// Its tiles are on another plane than the surface's, and reach
        /// it through their warp (`Map::globe_warps`).
        warped: bool,
    },
    Failed,
}

/// A plan source's state across frames, carried by key over edits: what
/// it has loaded and what its frames drew of it.
#[derive(Debug)]
pub(crate) struct SourceSlot {
    pub(crate) state: SourceState,
    /// The tiles the last frame drew, in drawing order: the next frame
    /// hides those it no longer draws, so only a frame sets it.
    pub(crate) placed: Vec<TileIndex>,
    /// Tiles that left the draw list but whose labels still draw until a
    /// placement covers what replaced them (`Labels::hold`).
    pub(crate) held: Vec<TileIndex>,
    /// The last plan's tiles to draw, between their update and their
    /// placement; `None` where it drew nothing.
    pub(crate) draws: Option<Vec<Draw>>,
    /// The zoom whose tiles the source keeps loaded to stand in, as its
    /// last plan had it (`Map::coarse_level`).
    pub(crate) coarse: Option<u8>,
}

impl SourceSlot {
    pub(crate) fn new(state: SourceState) -> Self {
        Self {
            state,
            placed: Vec::new(),
            held: Vec::new(),
            draws: None,
            coarse: None,
        }
    }
}

/// The host data a tile result holds until it's uploaded or dropped.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HostData(Option<HostBytesId>, Option<orb_hal::HostImageId>);

impl HostData {
    pub(crate) fn of(result: &TileResult) -> Self {
        let Self(bytes, image) = (result.image.as_ref()).map_or(Self(None, None), Self::of_image);
        Self(bytes.or_else(|| result.payload.host_bytes()), image)
    }

    /// A tile image's: a host image, or the host bytes its texels lie in.
    pub(crate) fn of_image(image: &crate::jobs::TileImage) -> Self {
        match image.pixels {
            crate::ImagePixels::Host(image) => Self(None, Some(image)),
            crate::ImagePixels::Bytes { bytes, .. } => Self(Some(bytes), None),
            crate::ImagePixels::Core(_) => Self(None, None),
        }
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
    pub(crate) tried: Option<[u64; 6]>,
}

/// Raises the nodes' row maxima to those of new meshes, counting each
/// rise in `revision`.
pub(crate) fn record_maxima(
    nodes: &mut HashMap<u32, NodeState>,
    meshes: &[ResultMesh],
    revision: &mut u64,
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
                *revision += 1;
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
                // Tiles past a source's maxzoom overzoom its last ones.
                if image { endpoint.maxzoom } else { 24 },
                // Map::trim_cache bounds the cache by memory.
                usize::MAX,
            ),
            endpoint: Arc::new(endpoint),
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
/// or laid out since (`stale`, by key); a symbol bucket goes with any of
/// its layers' nodes.
pub(crate) fn drop_stale(build: &mut TileResult, plan: &StylePlan, stale: &[u32]) {
    let keep: Vec<bool> = (build.meshes.iter())
        .map(|m| plan.node(m.node).is_some() && !stale.contains(&m.node))
        .collect();
    // A kept mesh drawing a dropped one's words takes them as its own,
    // and those drawing them too draw them from it; then shares name the
    // meshes kept by their new places.
    for pool in 0..2 {
        let mut owner: Vec<Option<u32>> = vec![None; build.meshes.len()];
        for i in 0..build.meshes.len() {
            let Some(of) = build.meshes[i].shares[pool].filter(|_| keep[i]) else {
                continue;
            };
            if keep[of as usize] {
                continue;
            }
            match owner[of as usize] {
                Some(new) => build.meshes[i].shares[pool] = Some(new),
                None => {
                    let words = [
                        build.meshes[of as usize].vertices,
                        build.meshes[of as usize].indices,
                    ];
                    let mesh = &mut build.meshes[i];
                    *[&mut mesh.vertices, &mut mesh.indices][pool] = words[pool];
                    mesh.shares[pool] = None;
                    owner[of as usize] = Some(i as u32);
                }
            }
        }
    }
    let kept: Vec<u32> = (keep.iter())
        .scan(0, |before, &k| {
            let at = *before;
            *before += u32::from(k);
            Some(at)
        })
        .collect();
    for mesh in &mut build.meshes {
        for share in &mut mesh.shares {
            *share = share.map(|of| kept[of as usize]);
        }
    }
    let mut keep = keep.into_iter();
    build.meshes.retain(|_| keep.next().expect("one per mesh"));
    build.symbols.retain(|s| {
        s.layers.iter().all(|&layer| {
            let keys = [0, 1].map(|n| crate::style::node_key(layer, n));
            plan.node(keys[0]).is_some() && !keys.iter().any(|k| stale.contains(k))
        })
    });
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
    // Alone, it uploads the words it shares too.
    let data = MeshData {
        vertices: result.vertices(mesh).words(&result.payload),
        indices: result.indices(mesh).words(&result.payload),
        props: mesh.props.words(&result.payload),
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
    (resident.meshes).push(Resident::new(mesh, id, rows, result.revision));
    Ok(())
}

/// A tile result's meshes the scene holds ([`scene_meshes`]), with `rows`
/// for each (or none), as a build for the renderer.
fn mesh_builds<'a>(
    plan: &StylePlan,
    meshes: &[(usize, &'a ResultMesh)],
    rows: Option<&'a [ResolvedRows]>,
) -> Vec<orb_render::MeshBuild<'a>> {
    // A mesh shares another's words as the build's index of that mesh.
    let build_index = |of: u32| meshes.binary_search_by_key(&(of as usize), |m| m.0).ok();
    (meshes.iter().map(|&(_, m)| m).enumerate())
        .map(|(i, m)| orb_render::MeshBuild {
            node: plan.node(m.node).expect("meshes of planned nodes") as u32,
            extent: m.extent,
            spans: [m.vertices.range(), m.indices.range(), m.props.range()],
            row_words: m.row_words,
            chunks: &m.chunks,
            rows: rows.map(|r| r[i].rows()).unwrap_or_default(),
            shares: m.shares.map(|of| {
                of.map(|of| build_index(of).expect("a mesh shares an earlier one's words"))
            }),
        })
        .collect()
}

/// Takes room for a vector tile result's meshes' words, to write them over
/// frames ([`orb_render::Reserved`]); [`upload`] then makes the tile of
/// them.
pub(crate) fn reserve(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    plan: &StylePlan,
    key: TileKey,
    result: &TileResult,
) -> Result<orb_render::Reserved, orb_render::UploadError> {
    let meshes: Vec<_> = scene_meshes(plan, result).collect();
    let builds = mesh_builds(plan, &meshes, None);
    let build = orb_render::TileBuild {
        key,
        words: result.payload.words(),
        meshes: &builds,
        patches: None,
    };
    renderer.reserve(gpu, &build)
}

/// Uploads a tile's meshes and symbol buckets, drawing through `patches`:
/// all of them or, when the scene is out of space, none. A tile whose
/// meshes' words are `reserved` and written ([`reserve`]) is made of them;
/// on `Err` their room goes back.
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
    reserved: Option<orb_render::Reserved>,
) -> Result<ResidentTile, OutOfSpace> {
    let meshes: Vec<_> = scene_meshes(plan, result).collect();
    let rows: Vec<ResolvedRows> = meshes
        .iter()
        .map(|(_, m)| resolve_rows(gpu, plan, result, m, atlas, namespace, states, warnings))
        .collect();
    let builds = mesh_builds(plan, &meshes, Some(&rows));
    let build = orb_render::TileBuild {
        key,
        words: result.payload.words(),
        meshes: &builds,
        patches,
    };
    let uploaded = match reserved {
        Some(reserved) => renderer.finish_tile(gpu, &build, reserved),
        None => renderer.upload_tile(gpu, &build),
    };
    let (handle, ids) = match uploaded {
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
        .map(|((&(_, m), rows), id)| Resident::new(m, id, rows, result.revision))
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
        false,
    ) {
        atlas.release_ramps(gpu, remove_resident(renderer, resident, states));
        return Err(e);
    }
    Ok(resident)
}

/// Evicts the least recently used cached tile the view doesn't use, of
/// source `first` before others'.
pub(crate) fn evict_unused(sources: &mut [SourceSlot], first: usize) -> Option<ResidentTile> {
    let others = (0..sources.len()).filter(|&s| s != first);
    std::iter::once(first)
        .chain(others)
        .find_map(|s| match &mut sources[s].state {
            SourceState::Ready { pyramid, .. } => pyramid.evict_lru(),
            _ => None,
        })
}

/// Evicts a tile that only stands in for a tile of a cover ([`TilePyramid::
/// evict_stand_in`]), `first`'s first and the ancestors of its `tile`
/// before others, to make room for `tile`.
pub(crate) fn evict_stand_in(
    sources: &mut [SourceSlot],
    first: usize,
    tile: TileIndex,
) -> Option<ResidentTile> {
    let others = (0..sources.len()).filter(|&s| s != first);
    std::iter::once(first)
        .chain(others)
        .find_map(|s| match &mut sources[s].state {
            SourceState::Ready { pyramid, .. } => {
                pyramid.evict_stand_in((s == first).then_some(tile))
            }
            _ => None,
        })
}

/// A tile result's meshes the scene holds as meshes, by their index in it:
/// symbol nodes' rows go up with their layer's bucket instead.
fn scene_meshes<'a>(
    plan: &StylePlan,
    result: &'a TileResult,
) -> impl Iterator<Item = (usize, &'a ResultMesh)> {
    let symbol =
        |m: &ResultMesh| (plan.node(m.node)).is_some_and(|i| plan.nodes[i].family.is_symbol());
    result
        .meshes
        .iter()
        .enumerate()
        .filter(move |(_, m)| !symbol(m))
}

/// What a resident tile costs to keep: the GPU bytes of its meshes and its
/// texture layer, and its picking data.
pub(crate) fn tile_cost(renderer: &Renderer, resident: &ResidentTile) -> u64 {
    // The renderer's count of the tile holds its meshes, all of them.
    let hits: u64 = resident.meshes.iter().map(|m| m.hits.bytes()).sum();
    renderer.tile_bytes(resident.gpu) + hits
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

#[cfg(test)]
mod tests {
    use super::WallLead;

    /// One expiry is one due time while the clocks' reads jitter, never
    /// before the wall clock passes it; a sleep the monotonic clock
    /// doesn't count brings it closer at once.
    #[test]
    fn an_expiry_keeps_its_due_time_until_the_clocks_part() {
        const EPOCH: f64 = 1.7e12;
        let expires = EPOCH + 60_000.0;
        let mut lead = WallLead::default();
        let mut dues = Vec::new();
        for frame in 0..100 {
            let now = f64::from(frame) * 16.7;
            // `Date.now()`'s whole milliseconds beside a finer clock.
            lead.read(now, (EPOCH + now).floor());
            dues.push(lead.due(expires));
        }
        assert!(dues.iter().all(|&due| due == dues[0]), "{dues:?}");
        assert!(dues[0] >= 60_000.0, "due at {} ms", dues[0]);
        // An hour asleep: the wall clock ran on.
        lead.read(2000.0, EPOCH + 2000.0 + 3_600_000.0);
        assert!(
            lead.due(expires) < 2000.0,
            "due at {} ms",
            lead.due(expires)
        );
    }
}
