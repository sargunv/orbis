use std::ops::Range;

use crate::Layer;

/// The geometry kind of a feature. Multi-geometries are features with several
/// parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GeometryType {
    /// One part with one ring holding every point.
    Point,
    /// One part per line string, each with one ring.
    LineString,
    /// One part per polygon: an exterior ring followed by its holes. Rings
    /// are implicitly closed (the first vertex is not repeated at the end);
    /// exterior rings wind clockwise and holes counter-clockwise in tile
    /// coordinates (y down), as in MVT.
    Polygon,
}

/// The geometry of one feature.
#[derive(Clone, Debug)]
pub struct FeatureGeometry<'a> {
    layer: &'a Layer,
    parts: Range<usize>,
}

impl<'a> FeatureGeometry<'a> {
    pub(crate) fn new(layer: &'a Layer, parts: Range<usize>) -> Self {
        Self { layer, parts }
    }

    pub fn parts(&self) -> impl ExactSizeIterator<Item = Part<'a>> + 'a {
        let layer = self.layer;
        self.parts.clone().map(move |p| Part {
            layer,
            rings: layer.part_rings[p] as usize..layer.part_rings[p + 1] as usize,
        })
    }
}

/// One part of a feature's geometry: a set of points, one line string, or one
/// polygon.
#[derive(Clone, Debug)]
pub struct Part<'a> {
    layer: &'a Layer,
    rings: Range<usize>,
}

impl<'a> Part<'a> {
    pub fn rings(&self) -> impl ExactSizeIterator<Item = &'a [[i32; 2]]> + 'a {
        let layer = self.layer;
        self.rings.clone().map(move |r| {
            &layer.vertices[layer.ring_vertices[r] as usize..layer.ring_vertices[r + 1] as usize]
        })
    }
}
