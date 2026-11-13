//! Elevations the app gives a `raster-dem` source it feeds
//! ([`HOST_TILES`](crate::HOST_TILES)), as heights in metres
//! ([`Map::set_host_elevations`]): ground that moves at the app's rate
//! without a tile's encode, decode and fresh upload each time.
//!
//! A call is a batch, one step of the app's ground. A tile the map waits
//! for takes its heights as its answer once the batch shows, and loads as
//! any tile does, from the heights shown when it arrives. The
//! resident ones are written over in place, their layers and memory kept:
//! frame by frame within the frame's budget, each tile's elevations go
//! into its raw DEM layer in the terrain's page and in the source's page.
//! Nothing draws the raw layers, and prepares that would read them wait
//! (`orb_render`'s staged layers), so a batch shows nothing until all of it
//! is in. Then one frame shows it whole: the tiles' prepares (padding and
//! slopes, and their neighbours' borders toward them), the CPU elevations
//! the camera, cover, labels and queries read, and the drape slots
//! hillshade and color-relief layers drew from the old slopes. No frame
//! shows two batches' heights at once. A batch given while another is
//! still going in replaces it, so an app that gives heights faster than
//! frames take them sees fewer steps rather than late ones; its layers
//! that the new batch leaves out go back to the heights shown.

use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};

use orb_crs::TileIndex;
use orb_hal::Gpu;
use orb_render::{PageId, Pixels, Renderer};
use orb_tile::DemGrid;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::map::Map;
use crate::resident::SourceState;

/// What [`Map::set_host_elevations`] did with a batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostElevations {
    /// The batch, as [`MapEvent::ElevationsShown`](crate::MapEvent::ElevationsShown)
    /// names it once it shows.
    pub batch: u64,
    /// Its tiles the map neither draws nor loads: their heights went
    /// nowhere, and the map asks for each again when it needs it
    /// ([`MapEvent::TileRequest`](crate::MapEvent::TileRequest)).
    pub dropped: Vec<TileIndex>,
}

/// A tile's elevations as its DEM layers hold them, and as the CPU reads
/// them: decoded from the packed texels, so both agree to the bit, once
/// something on the CPU reads them (the terrain; a hillshade's tiles need
/// texels alone).
#[derive(Debug)]
pub(crate) struct Packed {
    /// RGBA8 texels, a word each.
    texels: Vec<u32>,
    side: u32,
    unpack: [f64; 4],
    grid: OnceLock<Arc<DemGrid>>,
}

impl Packed {
    /// `heights` (rows top to bottom, `side`² of them) in the encoding
    /// `unpack` decodes: GL JS's `packDEMData`, to the encoding's step,
    /// clamped to what it holds.
    pub(crate) fn new(side: u32, heights: &[f32], unpack: [f64; 4]) -> Result<Self, String> {
        let n = (side as usize).pow(2);
        if heights.len() != n || n == 0 {
            return Err(format!(
                "a {side}×{side} DEM tile has {} heights",
                heights.len()
            ));
        }
        let [r, g, b, shift] = unpack;
        let factors = [r, g, b];
        let step = (factors.into_iter())
            .filter(|&f| f != 0.0)
            .min_by(f64::total_cmp)
            .ok_or("the source's encoding decodes no elevations")?;
        let ratios = factors.map(|f| if f == 0.0 { 0.0 } else { step / f });
        let top = (factors.iter().filter(|&&f| f != 0.0).count() as i32 * 8).min(24);
        let most = f64::from((1u32 << top) - 1);
        let scaled = |height: f32| {
            ((f64::from(height) + shift) / step)
                .round()
                .clamp(0.0, most)
        };
        let near = |a: f64, b: f64| (a - b).abs() <= b * 1e-9;
        let texels =
            if near(ratios[0], 1.0 / 65536.0) && near(ratios[1], 1.0 / 256.0) && ratios[2] == 1.0 {
                // Terrarium's and Mapbox's: the value's three bytes, high first.
                (heights.iter())
                    .map(|&h| {
                        let v = scaled(h) as u32;
                        (v >> 16) | (v >> 8 & 0xff) << 8 | (v & 0xff) << 16 | 0xff << 24
                    })
                    .collect()
            } else {
                (heights.iter())
                    .map(|&h| {
                        let v = scaled(h);
                        let c = |ratio: f64| ((v * ratio).floor() as i64).rem_euclid(256) as u32;
                        c(ratios[0]) | c(ratios[1]) << 8 | c(ratios[2]) << 16 | 0xff << 24
                    })
                    .collect()
            };
        Ok(Self {
            texels,
            side,
            unpack,
            grid: OnceLock::new(),
        })
    }

    /// The texels, as the DEM layers hold them.
    pub(crate) fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.texels)
    }

    pub(crate) fn side(&self) -> u32 {
        self.side
    }

    /// The elevations as the GPU decodes the texels.
    pub(crate) fn grid(&self) -> &Arc<DemGrid> {
        self.grid.get_or_init(|| {
            let grid = DemGrid::from_rgba(self.side, self.bytes(), self.unpack);
            Arc::new(grid.expect("packed texels are a whole square tile"))
        })
    }
}

/// A source's key and a tile.
type Key = (u32, TileIndex);

/// A tile's layers holding its elevations: in the terrain's page, and in
/// the source's.
type Layers = (Option<(PageId, u32)>, Option<(PageId, u32)>);

#[derive(Debug)]
enum Heights {
    Given(Vec<f32>),
    /// Heights packed before: shown, written back over a layer a replaced
    /// batch staged, or staged, written again over a tile's new layers.
    Shown(Arc<Packed>),
}

/// A tile's heights written over its layers and not yet shown.
#[derive(Debug)]
struct Staged {
    tile: TileIndex,
    packed: Arc<Packed>,
    /// Its layer in the terrain's page.
    terrain: Option<(PageId, u32)>,
    /// Its layer in the source's page.
    image: Option<(PageId, u32)>,
}

#[derive(Debug)]
struct Batch {
    id: u64,
    side: u32,
    unpack: [f64; 4],
    /// The source's id, for its event.
    source: String,
    /// Heights still to write, in the order given.
    queue: VecDeque<(TileIndex, Heights)>,
    /// Tiles the map waits for, answered once the batch shows: a tile
    /// that came in before would show its heights beside its neighbours'
    /// last ones.
    answers: Vec<(TileIndex, Heights)>,
    staged: Vec<Staged>,
}

#[derive(Debug, Default)]
pub(crate) struct HostHeights {
    /// The heights shown per source key and tile: what a load of the tile
    /// takes.
    shown: FxHashMap<Key, Arc<Packed>>,
    /// How many batches each tile showed: drape slots drawn from a tile
    /// redraw when it, or a neighbour it padded from, shows new heights.
    revisions: FxHashMap<Key, u64>,
    /// The batch going in, per source key.
    batches: FxHashMap<u32, Batch>,
    issued: u64,
    /// A batch showed this frame: the drape slots it changed redraw in it,
    /// whatever the frame's drape budget.
    pub(crate) showed: bool,
}

impl HostHeights {
    /// The heights a load of `tile` of the source keyed `source` takes.
    pub(crate) fn answer(&self, source: u32, tile: TileIndex) -> Option<&Arc<Packed>> {
        self.shown.get(&(source, tile))
    }

    /// Brings an arrived tile's result, made from heights given before,
    /// to the heights shown now.
    pub(crate) fn refresh(&self, source: u32, tile: TileIndex, build: &mut crate::TileResult) {
        let Some(shown) = self.shown.get(&(source, tile)) else {
            return;
        };
        if let Some(image) = &mut build.image
            && let crate::jobs::ImagePixels::Core(pixels) = &mut image.pixels
            && pixels.len() == shown.bytes().len()
            && pixels[..] != *shown.bytes()
        {
            pixels.copy_from_slice(shown.bytes());
            if build.dem.is_some() {
                build.dem = Some(Arc::clone(shown.grid()));
            }
        }
    }

    /// How many batches tile `tile` showed, for what drapes from it.
    pub(crate) fn revision(&self, source: u32, tile: TileIndex) -> u64 {
        self.revisions.get(&(source, tile)).copied().unwrap_or(0)
    }

    /// Whether a batch is going in.
    pub(crate) fn pending(&self) -> bool {
        !self.batches.is_empty()
    }

    /// Forgets source `source`'s heights of the tiles `which` picks: a
    /// reload asks for them again.
    pub(crate) fn forget(&mut self, source: u32, which: &dyn Fn(TileIndex) -> bool) {
        self.shown.retain(|&(s, t), _| s != source || !which(t));
    }

    /// Forgets every batch going in: a new style's sources are others.
    pub(crate) fn clear(&mut self) {
        self.batches.clear();
        self.shown.clear();
        self.revisions.clear();
    }
}

impl Map {
    /// Heights in metres for tiles of source `id`, a `raster-dem` source the
    /// app feeds ([`HOST_TILES`](crate::HOST_TILES)): per tile, `side`²
    /// of them, rows top to bottom, as the source's tiles' texels are. One
    /// call is one step of the app's ground, shown whole in one frame.
    ///
    /// A tile the map asked for
    /// ([`MapEvent::TileRequest`](crate::MapEvent::TileRequest)) takes its
    /// heights as its answer once the batch shows, so that it never comes
    /// in a step ahead of its neighbours. A tile the map draws, for its terrain or its
    /// layers, keeps its place and memory and has its heights written over,
    /// within the frames' budgets; the batch shows once all of it is in
    /// ([`MapEvent::ElevationsShown`](crate::MapEvent::ElevationsShown)),
    /// its hillshade and color-relief with it, and the elevations the
    /// camera, cover, labels and queries read. A call while the last
    /// batch is still going in replaces it: an app paced by
    /// `ElevationsShown` sees every step it gives. Heights round to the
    /// source's encoding's step and clamp to its range.
    pub fn set_host_elevations(
        &mut self,
        id: &str,
        side: u32,
        tiles: Vec<(TileIndex, Vec<f32>)>,
    ) -> Result<HostElevations, String> {
        let style = self.style.as_ref().ok_or("no style is loaded")?;
        let plan = style.worker.plan();
        let index = (plan.sources.iter())
            .position(|s| s.id == id)
            .ok_or_else(|| format!("no source {id}"))?;
        let SourceState::Ready {
            endpoint, pyramid, ..
        } = &style.sources[index]
        else {
            return Err(format!("{id} hasn't loaded"));
        };
        if !endpoint.host() {
            return Err(format!("{id} isn't a source the app feeds"));
        }
        let unpack = endpoint
            .dem_unpack
            .ok_or_else(|| format!("{id} isn't a raster-dem source"))?;
        let n = (side as usize).pow(2);
        if let Some((tile, h)) = tiles.iter().find(|(_, h)| h.len() != n || n == 0) {
            return Err(format!(
                "tile {tile:?} has {} heights, not {side}×{side}",
                h.len()
            ));
        }
        let key = plan.sources[index].key;
        let terrain = (style.terrain.as_ref()).filter(|t| t.key == key);
        let held = |tile: TileIndex| {
            pyramid.get(tile).is_some()
                || pyramid.is_loading(tile)
                || terrain
                    .is_some_and(|t| t.pyramid.get(tile).is_some() || t.pyramid.is_loading(tile))
        };
        self.host_heights.issued += 1;
        let batch = self.host_heights.issued;
        let mut dropped = Vec::new();
        let mut queue = VecDeque::new();
        let mut answers = Vec::new();
        for (tile, heights) in tiles {
            if self.host_tiles.waits(key, tile) {
                answers.push((tile, Heights::Given(heights)));
            } else if held(tile) {
                queue.push_back((tile, Heights::Given(heights)));
            } else {
                self.host_heights.shown.remove(&(key, tile));
                dropped.push(tile);
            }
        }
        // Layers a replaced batch wrote go back to the heights shown,
        // unless this one writes them; the tiles it was to answer that
        // this one doesn't keep its answers.
        if let Some(old) = self.host_heights.batches.remove(&key) {
            let given: FxHashSet<TileIndex> =
                (queue.iter().chain(&answers)).map(|(t, _)| *t).collect();
            for staged in old.staged.into_iter().rev() {
                if !given.contains(&staged.tile)
                    && let Some(shown) = self.host_heights.shown.get(&(key, staged.tile))
                {
                    queue.push_front((staged.tile, Heights::Shown(Arc::clone(shown))));
                }
            }
            answers.extend(old.answers.into_iter().filter(|(t, _)| !given.contains(t)));
        }
        // Shown, or answered, in the next frame's arrivals.
        self.host_heights.batches.insert(
            key,
            Batch {
                id: batch,
                side,
                unpack,
                source: id.to_string(),
                queue,
                answers,
                staged: Vec::new(),
            },
        );
        self.events.wake();
        Ok(HostElevations { batch, dropped })
    }

    /// A tile's load answered with heights given before: its result, made
    /// here and applied as any arrival.
    pub(crate) fn load_heights(
        &mut self,
        request: crate::TileRequest,
        edit: Option<u64>,
        terrain: bool,
        packed: &Arc<Packed>,
    ) {
        let result = match request.warp {
            true => Err("heights the app gives draw on the surface's own plane".to_string()),
            false => {
                let side = packed.side();
                Ok(crate::TileOutcome::Built(crate::TileResult {
                    image: Some(crate::jobs::TileImage {
                        width: side,
                        height: side,
                        pixels: crate::jobs::ImagePixels::Core(packed.bytes().to_vec()),
                    }),
                    dem: terrain.then(|| Arc::clone(packed.grid())),
                    ..crate::TileResult::default()
                }))
            }
        };
        self.tile_done(&request, edit, terrain)(result);
    }

    /// Writes what the frame's budget allows of the batches going in, and
    /// shows each once all of it is in.
    pub(crate) fn apply_heights(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        deadline: f64,
    ) {
        self.host_heights.showed = false;
        if self.host_heights.batches.is_empty() {
            return;
        }
        let clock = Arc::clone(&self.services.clock);
        let keys: Vec<u32> = self.host_heights.batches.keys().copied().collect();
        for key in keys {
            let mut batch = self.host_heights.batches.remove(&key).expect("listed");
            let at = (&*clock, deadline);
            while !self.stage_heights(gpu, renderer, key, &mut batch, at) {
                // All in. Tiles that arrived, or went and came back in
                // other layers, while it went in are written too, so that
                // it shows on every tile drawn.
                let mut moved = Vec::new();
                batch.staged.retain(|staged| {
                    let now = self.heights_layers(renderer, key, staged.tile);
                    let keep = now == (staged.terrain, staged.image) || now == (None, None);
                    if !keep {
                        moved.push((staged.tile, Heights::Shown(Arc::clone(&staged.packed))));
                    }
                    keep
                });
                if moved.is_empty() {
                    break;
                }
                batch.queue.extend(moved);
            }
            if !batch.queue.is_empty() {
                self.host_heights.batches.insert(key, batch);
                continue;
            }
            self.show_batch(renderer, key, batch);
        }
        self.tiles_deferred |= self.host_heights.pending();
    }

    /// The layers tile `tile` of the source keyed `key` holds its
    /// elevations in: the terrain's page's, and the source's.
    fn heights_layers(&self, renderer: &Renderer, key: u32, tile: TileIndex) -> Layers {
        let terrain = (self.style.as_ref())
            .and_then(|s| s.terrain.as_ref())
            .filter(|t| t.key == key);
        let on_terrain = terrain.and_then(|t| Some((t.page?, t.pyramid.get(tile)?.layer)));
        (on_terrain, renderer.image_layer(key, tile))
    }

    /// Writes the batch's queued heights over their layers within the
    /// frame's budget: whether the budget ran out first.
    fn stage_heights(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        key: u32,
        batch: &mut Batch,
        (clock, deadline): (&dyn orb_host::Clock, f64),
    ) -> bool {
        while let Some(&(tile, _)) = batch.queue.front() {
            let (on_terrain, on_image) = self.heights_layers(renderer, key, tile);
            let layers: Vec<(PageId, u32)> = on_terrain.into_iter().chain(on_image).collect();
            let bytes = layers.len() as u64 * u64::from(batch.side).pow(2) * 4;
            if !layers.is_empty() && !self.beat.affords(clock.now() < deadline, bytes) {
                return true;
            }
            let (_, heights) = batch.queue.pop_front().expect("in front");
            if let Some(&(page, _)) = layers
                .iter()
                .find(|(p, _)| renderer.page_size(*p) != [batch.side; 2])
            {
                let [w, h] = renderer.page_size(page);
                let message = format!(
                    "{}: heights {side}×{side} a tile over tiles of {w}×{h}",
                    batch.source,
                    side = batch.side
                );
                self.events.push(crate::MapEvent::Error { message });
                continue;
            }
            let packed = match heights {
                Heights::Shown(packed) => packed,
                Heights::Given(heights) => match Packed::new(batch.side, &heights, batch.unpack) {
                    Ok(packed) => Arc::new(packed),
                    Err(message) => {
                        self.events.push(crate::MapEvent::Error { message });
                        continue;
                    }
                },
            };
            for &(page, layer) in &layers {
                renderer.stage_dem(gpu, page, layer, Pixels::Core(packed.bytes()));
            }
            // The terrain's CPU elevations, made within the budget rather
            // than in the frame that shows them.
            if on_terrain.is_some() {
                packed.grid();
            }
            if !layers.is_empty() {
                self.beat.spend(bytes);
                self.beat.count();
            }
            batch.staged.retain(|s| s.tile != tile);
            batch.staged.push(Staged {
                tile,
                packed,
                terrain: on_terrain,
                image: on_image,
            });
        }
        false
    }

    /// Shows a batch all in: its layers' prepares, the terrain's CPU
    /// elevations and the heights loads take.
    fn show_batch(&mut self, renderer: &mut Renderer, key: u32, batch: Batch) {
        let mut pages: FxHashMap<PageId, Vec<u32>> = FxHashMap::default();
        let mut terrain = (self.style.as_mut())
            .and_then(|s| s.terrain.as_mut())
            .filter(|t| t.key == key);
        for staged in &batch.staged {
            // The tile left since: its layer went to another, which
            // wrote its own.
            if let Some((page, layer)) = staged.image
                && renderer.image_layer(key, staged.tile) == Some((page, layer))
            {
                pages.entry(page).or_default().push(layer);
            }
            if let (Some((page, layer)), Some(terrain)) = (staged.terrain, terrain.as_deref_mut())
                && terrain.set_heights(staged.tile, layer, staged.packed.grid())
            {
                pages.entry(page).or_default().push(layer);
            }
            let at = (key, staged.tile);
            self.host_heights
                .shown
                .insert(at, Arc::clone(&staged.packed));
            *self.host_heights.revisions.entry(at).or_default() += 1;
        }
        // The CPU's elevations are taken now, before this frame's camera
        // and cover read them: they show the batch in the frame its
        // texels do.
        if let Some(terrain) = terrain {
            terrain.elevation(&self.surface.tms().clone());
        }
        for (page, layers) in pages {
            renderer.publish_dems(page, &layers);
        }
        // The tiles it answers load from the heights shown.
        let current = self.style.as_ref().map(|s| s.id);
        for (tile, heights) in batch.answers {
            let packed = match heights {
                Heights::Shown(packed) => packed,
                Heights::Given(heights) => match Packed::new(batch.side, &heights, batch.unpack) {
                    Ok(packed) => Arc::new(packed),
                    Err(message) => {
                        self.events.push(crate::MapEvent::Error { message });
                        continue;
                    }
                },
            };
            self.host_heights
                .shown
                .insert((key, tile), Arc::clone(&packed));
            for (request, edit, terrain) in self.host_tiles.answered_by_heights(key, tile) {
                if Some(request.style) == current {
                    self.load_heights(request, edit, terrain, &packed);
                }
            }
        }
        self.host_heights.showed = true;
        self.events.push(crate::MapEvent::ElevationsShown {
            source: batch.source,
            batch: batch.id,
        });
        self.events.wake();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Heights pack as the source's own tiles would hold them, and read
    /// back to the encoding's step.
    #[test]
    fn heights_pack_in_the_sources_encoding() {
        let terrarium = orb_style::DemEncoding::Terrarium.unpack([0.0; 4]);
        let mapbox = orb_style::DemEncoding::Mapbox.unpack([0.0; 4]);
        let heights = [0.0, 258.5, -256.0, 4807.3];
        let packed = Packed::new(2, &heights, terrarium).unwrap();
        assert_eq!(&packed.bytes()[..8], &[128, 0, 0, 255, 129, 2, 128, 255]);
        assert_eq!(packed.grid().elevations()[..3], heights[..3]);
        assert!((packed.grid().elevations()[3] - 4807.3).abs() <= 1.0 / 512.0);
        let packed = Packed::new(2, &heights, mapbox).unwrap();
        let back = packed.grid().elevations();
        assert!(
            heights
                .iter()
                .zip(back)
                .all(|(a, b)| (a - b).abs() <= 0.05 + 1e-3)
        );
        // Past the encoding's range, clamped rather than wrapped.
        let packed = Packed::new(1, &[-50_000.0], terrarium).unwrap();
        assert_eq!(
            packed.grid().elevations(),
            [0.0],
            "black reads as sea level"
        );
        // An encoding of its own packs through the general path alike.
        let own = [256.0 * 2.0, 2.0, 2.0 / 256.0, 32768.0 * 2.0];
        let packed = Packed::new(2, &heights, own).unwrap();
        let back = packed.grid().elevations();
        assert!(
            heights
                .iter()
                .zip(back)
                .all(|(a, b)| (a - b).abs() <= 1.0 / 128.0)
        );
        assert!(Packed::new(2, &heights[..3], terrarium).is_err());
    }
}
