//! Owner-tile point records, shared by circles and Gaussian heat kernels.
//!
//! Each point is two words (packed i16 position, feature slot). One index per
//! point addresses the record; the shader synthesizes six quad vertices per
//! instance. Chunks hold at most 256 point instances. No radius or paint is baked into geometry.

use crate::{Mesh, MeshBuilder};
use orb_tile::Layer;

pub const STRIDE: usize = 2;

/// Layout points in feature order, skipping the source's buffer region.
/// As in GL JS's circle bucket, the tile is half open: [0, extent).
/// Lines and polygons also contribute their vertices, matching GL JS.
pub fn layout(layer: &Layer, features: &[u32]) -> Mesh {
    layout_sorted(layer, features, |_| None)
}

/// The sorted form keeps different feature keys in separate chunks so the
/// scene can merge their order across tiles without changing geometry.
pub fn layout_sorted(
    layer: &Layer,
    features: &[u32],
    sort_key: impl Fn(u32) -> Option<f32>,
) -> Mesh {
    let extent = i32::try_from(layer.extent()).expect("extent fits i32");
    assert!(
        extent <= i32::from(i16::MAX),
        "extent exceeds i16 positions"
    );
    let mut builder = MeshBuilder::new(STRIDE);
    for (slot, &feature) in features.iter().enumerate() {
        builder.set_sort_key(sort_key(feature));
        for part in layer.geometry(feature as usize).parts() {
            for ring in part.rings() {
                for &[x, y] in ring {
                    if x < 0 || y < 0 || x >= extent || y >= extent {
                        continue;
                    }
                    let pos = [x as i16, y as i16];
                    let packed = u32::from(x as u16) | u32::from(y as u16) << 16;
                    let v = builder.push_vertex(pos, &[packed, slot as u32]);
                    builder.push_point(v);
                }
            }
        }
    }
    builder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use orb_tile::{GeometryType, LayerBuilder};

    #[test]
    fn owns_half_open_tile_and_keeps_feature_slots() {
        let mut b = LayerBuilder::new("points", 4096);
        b.begin_feature(None, GeometryType::Point);
        b.begin_part();
        b.begin_ring();
        for p in [[-1, 50], [0, 0], [4095, 4095], [4096, 50], [50, 4096]] {
            b.push_vertex(p);
        }
        let mesh = layout(&b.finish(), &[0]);
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.indices, [0, 1]);
    }
}
