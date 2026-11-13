//! GL JS `cross_tile_symbol_index.ts`: one id for a label across the tiles
//! and zooms that show it, so placement keeps its fade and draws it once.
//!
//! Symbols match on their key and their anchor, rounded to about 4 pixels
//! at the coarser tile's zoom. GL JS switches to a KDBush index beyond 128
//! symbols of one key; a scan finds the same match (the first unused one
//! within tolerance), so Orbis scans.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::BTreeMap;

use crate::bucket::SymbolInstance;
use crate::geometry::EXTENT;
use crate::placement::view::TileId;

/// Rounds anchors to roughly a 4-pixel grid.
const ROUNDING: f64 = 512.0 / EXTENT / 2.0;

/// A bucket as the index knows it: its tile, its instances' keys and
/// anchors, and the cross-tile ids assigned to them.
#[derive(Debug)]
struct TileLayerIndex {
    tile: TileId,
    bucket: u64,
    /// By key: rounded anchors and their ids, in instance order.
    by_key: HashMap<u64, (Vec<[i64; 2]>, Vec<u32>)>,
}

impl TileLayerIndex {
    fn new(tile: TileId, instances: &[SymbolInstance], ids: &[u32], bucket: u64) -> Self {
        let mut by_key: HashMap<u64, (Vec<[i64; 2]>, Vec<u32>)> = HashMap::default();
        for (instance, &id) in instances.iter().zip(ids) {
            let entry = by_key.entry(instance.key).or_default();
            entry.0.push([
                (f64::from(instance.anchor[0]) * ROUNDING).floor() as i64,
                (f64::from(instance.anchor[1]) * ROUNDING).floor() as i64,
            ]);
            entry.1.push(id);
        }
        Self {
            tile,
            bucket,
            by_key,
        }
    }

    /// GL JS `getScaledCoordinates`.
    fn scaled(&self, instance: &SymbolInstance, child: TileId) -> [i64; 2] {
        let dz = i32::from(child.z) - i32::from(self.tile.z);
        let scale = ROUNDING / 2f64.powi(dz);
        let x = (f64::from(child.x) * EXTENT + f64::from(instance.anchor[0])) * scale;
        let y = (f64::from(child.y) * EXTENT + f64::from(instance.anchor[1])) * scale;
        let ox = f64::from(self.tile.x) * EXTENT * ROUNDING;
        let oy = f64::from(self.tile.y) * EXTENT * ROUNDING;
        [(x - ox).floor() as i64, (y - oy).floor() as i64]
    }

    /// GL JS `findMatches`: gives unmatched instances the ids of matching
    /// symbols here, each id once per zoom.
    fn find_matches(
        &self,
        instances: &[SymbolInstance],
        ids: &mut [u32],
        tile: TileId,
        used: &mut HashSet<u32>,
    ) {
        let tolerance = if self.tile.z < tile.z {
            1
        } else {
            1i64 << (self.tile.z - tile.z)
        };
        for (instance, id) in instances.iter().zip(ids) {
            if *id != 0 {
                continue;
            }
            let Some((positions, candidates)) = self.by_key.get(&instance.key) else {
                continue;
            };
            let [x, y] = self.scaled(instance, tile);
            for (p, &candidate) in positions.iter().zip(candidates) {
                if (p[0] - x).abs() <= tolerance
                    && (p[1] - y).abs() <= tolerance
                    && !used.contains(&candidate)
                {
                    used.insert(candidate);
                    *id = candidate;
                    break;
                }
            }
        }
    }
}

/// One symbol layer's index (GL JS `CrossTileSymbolLayerIndex`).
#[derive(Debug, Default)]
struct LayerIndex {
    indexes: BTreeMap<u8, HashMap<TileId, TileLayerIndex>>,
    used: HashMap<u8, HashSet<u32>>,
    lng: f64,
}

impl LayerIndex {
    /// GL JS `handleWrapJump`: re-wraps the index when the camera's
    /// longitude crosses into another world copy.
    fn handle_wrap_jump(&mut self, lng: f64) {
        let delta = ((lng - self.lng) / 360.0).round() as i32;
        if delta != 0 {
            for zoom in self.indexes.values_mut() {
                *zoom = std::mem::take(zoom)
                    .into_values()
                    .map(|mut index| {
                        index.tile.wrap += delta;
                        (index.tile, index)
                    })
                    .collect();
            }
        }
        self.lng = lng;
    }

    fn remove_ids(used: &mut HashMap<u8, HashSet<u32>>, z: u8, index: &TileLayerIndex) {
        if let Some(used) = used.get_mut(&z) {
            for (_, ids) in index.by_key.values() {
                for id in ids {
                    used.remove(id);
                }
            }
        }
    }

    /// GL JS `addBucket`.
    fn add_bucket(
        &mut self,
        tile: TileId,
        bucket: u64,
        instances: &[SymbolInstance],
        ids: &mut Vec<u32>,
        next_id: &mut u32,
    ) {
        if let Some(existing) = self.indexes.get(&tile.z).and_then(|z| z.get(&tile)) {
            if existing.bucket == bucket {
                return;
            }
            // A replaced bucket gives up its ids for the new one to claim.
            Self::remove_ids(&mut self.used, tile.z, existing);
        }
        ids.clear();
        ids.resize(instances.len(), 0);
        let mut used = std::mem::take(self.used.entry(tile.z).or_default());
        for (&z, indexes) in &self.indexes {
            if z > tile.z {
                for child in indexes.values() {
                    if child.tile.is_child_of(tile) {
                        child.find_matches(instances, ids, tile, &mut used);
                    }
                }
            } else if let Some(parent) = indexes.get(&tile.scaled_to(z)) {
                parent.find_matches(instances, ids, tile, &mut used);
            }
        }
        for id in ids.iter_mut() {
            if *id == 0 {
                *next_id += 1;
                *id = *next_id;
                used.insert(*id);
            }
        }
        self.used.insert(tile.z, used);
        self.indexes
            .entry(tile.z)
            .or_default()
            .insert(tile, TileLayerIndex::new(tile, instances, ids, bucket));
    }

    /// GL JS `removeStaleBuckets`.
    fn remove_stale(&mut self, current: &HashSet<u64>) {
        for (&z, indexes) in &mut self.indexes {
            indexes.retain(|_, index| {
                let keep = current.contains(&index.bucket);
                if !keep {
                    Self::remove_ids(&mut self.used, z, index);
                }
                keep
            });
        }
    }
}

/// Cross-tile ids for every symbol layer (GL JS `CrossTileSymbolIndex`).
#[derive(Debug, Default)]
pub(crate) struct CrossTileIndex {
    layers: HashMap<u32, LayerIndex>,
    next_id: u32,
}

/// A bucket being indexed: its tile, its instances and the ids to assign.
pub(crate) struct Indexed<'a> {
    pub(crate) tile: TileId,
    pub(crate) bucket: u64,
    pub(crate) instances: &'a [SymbolInstance],
    pub(crate) ids: &'a mut Vec<u32>,
}

impl CrossTileIndex {
    /// GL JS `addLayer`: assigns ids to the layer's buckets, highest zoom
    /// first, and drops buckets no longer drawn. A bucket already indexed
    /// keeps the ids it holds.
    pub(crate) fn add_layer(&mut self, layer: u32, buckets: &mut [Indexed<'_>], lng: f64) {
        let index = self.layers.entry(layer).or_default();
        index.handle_wrap_jump(lng);
        let mut current = HashSet::with_capacity_and_hasher(buckets.len(), Default::default());
        for b in buckets.iter_mut() {
            index.add_bucket(b.tile, b.bucket, b.instances, b.ids, &mut self.next_id);
            current.insert(b.bucket);
        }
        index.remove_stale(&current);
    }

    /// GL JS `pruneUnusedLayers`.
    pub(crate) fn retain_layers(&mut self, layers: &HashSet<u32>) {
        self.layers.retain(|k, _| layers.contains(k));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    fn instance(key: u64, x: i16, y: i16) -> SymbolInstance {
        SymbolInstance {
            key,
            anchor: [x, y],
            ..SymbolInstance::zeroed()
        }
    }

    #[test]
    fn matches_labels_across_zooms_and_assigns_new_ids() {
        let mut index = CrossTileIndex::default();
        let parent = TileId {
            z: 6,
            x: 8,
            y: 8,
            wrap: 0,
        };
        let child = TileId {
            z: 7,
            x: 17,
            y: 16,
            wrap: 0,
        };
        let parent_instances = [instance(1, 4096, 1024), instance(2, 100, 100)];
        let mut parent_ids = Vec::new();
        index.add_layer(
            1,
            &mut [Indexed {
                tile: parent,
                bucket: 1,
                instances: &parent_instances,
                ids: &mut parent_ids,
            }],
            0.0,
        );
        assert_eq!(parent_ids, [1, 2]);
        // Key 1 at the same place in the child (x 4096 of the parent is the
        // child's left edge, plus a rounding error); key 2 elsewhere.
        let child_instances = [instance(1, 3, 2048), instance(2, 4000, 4000)];
        let mut child_ids = Vec::new();
        let mut parent_again = parent_ids.clone();
        index.add_layer(
            1,
            &mut [
                Indexed {
                    tile: child,
                    bucket: 2,
                    instances: &child_instances,
                    ids: &mut child_ids,
                },
                Indexed {
                    tile: parent,
                    bucket: 1,
                    instances: &parent_instances,
                    ids: &mut parent_again,
                },
            ],
            0.0,
        );
        assert_eq!(child_ids, [1, 3]);
        // Buckets already indexed keep their ids; a new bucket for the
        // parent's tile takes over the ids of its symbols.
        let mut parent_new = Vec::new();
        index.add_layer(
            1,
            &mut [
                Indexed {
                    tile: child,
                    bucket: 2,
                    instances: &child_instances,
                    ids: &mut child_ids,
                },
                Indexed {
                    tile: parent,
                    bucket: 3,
                    instances: &parent_instances,
                    ids: &mut parent_new,
                },
            ],
            0.0,
        );
        assert_eq!(child_ids, [1, 3]);
        assert_eq!(parent_new, [1, 2]);
    }
}
