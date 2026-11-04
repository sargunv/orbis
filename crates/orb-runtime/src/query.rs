//! Published picking data. Tile jobs supply only selected feature geometry;
//! render publication projects it into CSS pixels. Symbols can publish a
//! separate index and compare their draw order here when placement is ready.
use glam::{DVec2, DVec3, Vec2};
use orb_camera::View;
use orb_crs::{TileIndex, TileMatrixSet};

/// The topmost feature under a point in the last published frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    pub layer: String,
    pub source: String,
    pub source_layer: Option<String>,
    pub id: Option<String>,
    pub generation: u64,
}

/// Compact tile-local picking geometry, transported with a render mesh.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HitFeature {
    pub id: Option<String>,
    /// 0: polygons, 1: lines, 2: points.
    pub kind: u32,
    pub extent: u32,
    /// Circles: the radius; lines: half the width.
    pub radius: f32,
    /// Circles: the stroke width, bound apart from the radius.
    pub stroke: f32,
    pub height: f32,
    pub base: f32,
    pub parts: Vec<Vec<Vec<[i32; 2]>>>,
}

#[derive(Debug)]
pub(crate) struct ScreenHit {
    pub(crate) hit: Hit,
    pub(crate) order: usize,
    kind: u32,
    radius: f32,
    parts: Vec<Vec<Vec<Vec2>>>,
    clips: Vec<Vec<Vec2>>,
    min: Vec2,
    max: Vec2,
}

impl ScreenHit {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn project(
        feature: &HitFeature,
        hit: Hit,
        order: usize,
        tile: TileIndex,
        wrap: i32,
        regions: &[TileIndex],
        view: &View,
        tms: &TileMatrixSet,
        radius: f64,
        elevation: (f64, f64),
        shift: DVec2,
    ) -> Option<Self> {
        let transform = tms.data_to_world(tile, feature.extent);
        let period = tms.projection().wrap_period().unwrap_or(0.0);
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        let project = |p: [i32; 2], height: f64| {
            let world = transform.transform_point2(DVec2::new(p[0] as f64, p[1] as f64))
                + DVec2::new(f64::from(wrap) * period, 0.0)
                + shift;
            let relative = world - view.camera_world();
            let clip = view.view_proj() * DVec3::new(relative.x, relative.y, height).extend(1.0);
            if clip.w <= 0.0 {
                return None;
            }
            let ndc = clip.truncate() / clip.w;
            let viewport = view.viewport();
            Some(Vec2::new(
                ((ndc.x + 1.0) * viewport.width / 2.0) as f32,
                ((1.0 - ndc.y) * viewport.height / 2.0) as f32,
            ))
        };
        let mut parts = Vec::new();
        for part in &feature.parts {
            let mut rings = Vec::new();
            for ring in part {
                let points = ring
                    .iter()
                    .map(|p| project(*p, elevation.1))
                    .collect::<Option<Vec<_>>>()?;
                rings.push(points);
            }
            parts.push(rings);
            if elevation.1 > elevation.0 {
                for ring in part {
                    for (a, b) in ring
                        .iter()
                        .zip(ring.iter().cycle().skip(1))
                        .take(ring.len())
                    {
                        parts.push(vec![vec![
                            project(*a, elevation.0)?,
                            project(*b, elevation.0)?,
                            project(*b, elevation.1)?,
                            project(*a, elevation.1)?,
                        ]]);
                    }
                }
            }
        }
        for point in parts.iter().flatten().flatten() {
            min = min.min(*point);
            max = max.max(*point);
        }
        let radius = radius as f32;
        min -= Vec2::splat(radius);
        max += Vec2::splat(radius);
        let clips = regions
            .iter()
            .map(|region| {
                let bounds = tms.tile_bounds(*region);
                [
                    bounds.min,
                    DVec2::new(bounds.max.x, bounds.min.y),
                    bounds.max,
                    DVec2::new(bounds.min.x, bounds.max.y),
                ]
                .into_iter()
                .filter_map(|p| {
                    view.project(p + DVec2::new(f64::from(wrap) * period, 0.0) + shift)
                        .map(|p| p.as_vec2())
                })
                .collect()
            })
            .collect();
        Some(Self {
            hit,
            order,
            kind: feature.kind,
            radius,
            parts,
            min,
            max,
            clips,
        })
    }

    pub(crate) fn contains(&self, p: DVec2) -> bool {
        let p = p.as_vec2();
        if !p.is_finite() || !p.cmpge(self.min).all() || !p.cmple(self.max).all() {
            return false;
        }
        if self.hit.source_layer.is_some() && !self.clips.iter().any(|clip| inside(p, clip)) {
            return false;
        }
        self.parts.iter().any(|part| match self.kind {
            2 => part
                .iter()
                .flatten()
                .any(|q| q.distance_squared(p) <= self.radius * self.radius),
            1 => part.iter().any(|ring| {
                ring.windows(2)
                    .any(|pair| distance(p, pair[0], pair[1]) <= self.radius)
            }),
            _ => {
                part.first().is_some_and(|outer| inside(p, outer))
                    && !part.iter().skip(1).any(|hole| inside(p, hole))
            }
        })
    }
}

fn distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let d = b - a;
    let t = if d.length_squared() == 0.0 {
        0.0
    } else {
        ((p - a).dot(d) / d.length_squared()).clamp(0.0, 1.0)
    };
    p.distance(a + d * t)
}
fn inside(p: Vec2, ring: &[Vec2]) -> bool {
    let mut inside = false;
    for (a, b) in ring
        .iter()
        .zip(ring.iter().cycle().skip(1))
        .take(ring.len())
    {
        if distance(p, *a, *b) < 1e-7 {
            return true;
        }
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
    }
    inside
}

impl crate::Map {
    /// Synchronous, topmost hit in CSS pixels from the last rendered frame.
    pub fn hit_test(&self, point: DVec2) -> Option<Hit> {
        self.hits
            .iter()
            .rev()
            .find(|h| h.contains(point))
            .map(|h| h.hit.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn polygon_edges_holes_and_segments() {
        let outer = vec![
            Vec2::ZERO,
            Vec2::new(10., 0.),
            Vec2::splat(10.),
            Vec2::new(0., 10.),
        ];
        assert!(inside(Vec2::splat(5.), &outer));
        assert!(inside(Vec2::ZERO, &outer));
        assert!(!inside(Vec2::splat(11.), &outer));
        assert_eq!(
            distance(Vec2::new(5., 2.), Vec2::ZERO, Vec2::new(10., 0.)),
            2.
        );
    }
}

/// One frame's picking publication, alongside tile placement. Kept apart
/// from GPU uploads and from the future symbol placement index.
pub(crate) struct Publisher<'a> {
    pub(crate) hits: &'a mut Vec<ScreenHit>,
    pub(crate) plan: &'a crate::style::StylePlan,
    pub(crate) view: &'a View,
    pub(crate) tms: &'a TileMatrixSet,
    pub(crate) camera: &'a orb_camera::CameraState,
}
impl Publisher<'_> {
    pub(crate) fn tile(
        &mut self,
        resident: &crate::map::ResidentTile,
        index: usize,
        tile: TileIndex,
        draws: &[&orb_source::Draw],
    ) {
        for mesh in &resident.meshes {
            let Some(node_index) = self.plan.node(mesh.node) else {
                continue;
            };
            let plan = self.plan;
            let node = &plan.nodes[node_index];
            let layer = &plan.style.layers[node.layer];
            if !layer.is_visible_at(self.camera.zoom()) {
                continue;
            }
            for draw in draws {
                for feature in &mesh.hits {
                    // Each value per feature where it's bound per feature, else
                    // as the style says now (uniform edits don't relayout).
                    let zoom = self.camera.zoom();
                    let per_feature =
                        |kind| matches!(kind, orb_v8::Kind::Source | orb_v8::Kind::Composite);
                    let radius = match &layer.kind {
                        orb_v8::LayerKind::Circle(c) => {
                            let radius = if per_feature(c.paint.radius.kind()) {
                                feature.radius
                            } else {
                                c.paint.radius.evaluate(zoom)
                            };
                            let stroke = if per_feature(c.paint.stroke_width.kind()) {
                                feature.stroke
                            } else {
                                c.paint.stroke_width.evaluate(zoom)
                            };
                            radius + stroke
                        }
                        orb_v8::LayerKind::Line(l) if !per_feature(l.paint.width.kind()) => {
                            l.paint.width.evaluate(zoom) / 2.0
                        }
                        _ => feature.radius,
                    };
                    let elevation = match &layer.kind {
                        orb_v8::LayerKind::FillExtrusion(e) => {
                            let scale =
                                self.tms.projection().style_height_scale(self.camera.target);
                            let uniform = |kind| {
                                matches!(kind, orb_v8::Kind::Constant | orb_v8::Kind::Camera)
                            };
                            let base = if uniform(e.paint.base.kind()) {
                                e.paint.base.evaluate(self.camera.zoom())
                            } else {
                                feature.base
                            };
                            let height = if uniform(e.paint.height.kind()) {
                                e.paint.height.evaluate(self.camera.zoom())
                            } else {
                                feature.height
                            };
                            (f64::from(base) * scale, f64::from(height) * scale)
                        }
                        _ => (0.0, 0.0),
                    };
                    let (translate, anchor) = match &layer.kind {
                        orb_v8::LayerKind::Fill(f) => (
                            f.paint.translate.evaluate(self.camera.zoom()),
                            f.paint.translate_anchor.evaluate(self.camera.zoom()),
                        ),
                        orb_v8::LayerKind::Line(l) => (
                            l.paint.translate.evaluate(self.camera.zoom()),
                            l.paint.translate_anchor.evaluate(self.camera.zoom()),
                        ),
                        orb_v8::LayerKind::Circle(c) => (
                            c.paint.translate.evaluate(self.camera.zoom()),
                            c.paint.translate_anchor.evaluate(self.camera.zoom()),
                        ),
                        orb_v8::LayerKind::FillExtrusion(e) => (
                            e.paint.translate.evaluate(self.camera.zoom()),
                            e.paint.translate_anchor.evaluate(self.camera.zoom()),
                        ),
                        _ => ([0.0; 2], orb_v8::TranslateAnchor::Map),
                    };
                    let mut shift = DVec2::new(f64::from(translate[0]), f64::from(translate[1]));
                    if anchor == orb_v8::TranslateAnchor::Viewport {
                        let (sin, cos) = self.camera.bearing.to_radians().sin_cos();
                        shift = DVec2::new(
                            shift.x * cos - shift.y * sin,
                            shift.x * sin + shift.y * cos,
                        );
                    }
                    shift.y = -shift.y;
                    shift *= self.view.world_per_px();
                    let hit = crate::Hit {
                        layer: layer.id.clone(),
                        source: plan.sources[index].id.clone(),
                        source_layer: plan.source_layer(node).map(str::to_string),
                        id: feature.id.clone(),
                        generation: 0,
                    };
                    if let Some(hit) = crate::query::ScreenHit::project(
                        feature,
                        hit,
                        node.layer,
                        tile,
                        draw.wrap,
                        &draw.regions,
                        self.view,
                        self.tms,
                        f64::from(radius),
                        elevation,
                        shift,
                    ) {
                        self.hits.push(hit);
                    }
                }
            }
        }
    }
}

/// A published, transferable picking snapshot for a host's UI thread.
/// The UI never waits for a worker or re-decodes a vector tile to pick.
#[derive(Debug, Default)]
pub struct HitIndex {
    hits: Vec<ScreenHit>,
    generation: u64,
}
impl HitIndex {
    pub fn hit_test(&self, point: DVec2) -> Option<Hit> {
        self.hits.iter().rev().find(|h| h.contains(point)).map(|h| {
            let mut hit = h.hit.clone();
            hit.generation = self.generation;
            hit
        })
    }
    pub fn set_generation(&mut self, generation: u64) {
        self.generation = generation;
    }

    /// Accepts a snapshot emitted by `Map::hit_snapshot`, atomically.
    pub fn update(&mut self, bytes: &[u8]) -> Result<(), String> {
        if !bytes.len().is_multiple_of(4) {
            return Err("malformed hit snapshot".into());
        }
        let words: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let mut words = words.into_iter();
        let mut next = || {
            words
                .next()
                .ok_or_else(|| "malformed hit snapshot".to_string())
        };
        let mut hits = Vec::new();
        for _ in 0..next()? {
            let layer = crate::jobs::read_text(&mut next)?;
            let source = crate::jobs::read_text(&mut next)?;
            let source_layer = if next()? != 0 {
                Some(crate::jobs::read_text(&mut next)?)
            } else {
                None
            };
            let id = if next()? != 0 {
                Some(crate::jobs::read_text(&mut next)?)
            } else {
                None
            };
            let kind = next()?;
            let mut number = || next().map(f32::from_bits);
            let radius = number()?;
            let min = Vec2::new(number()?, number()?);
            let max = Vec2::new(number()?, number()?);
            let mut parts = Vec::new();
            for _ in 0..next()? {
                let mut rings = Vec::new();
                for _ in 0..next()? {
                    rings.push(read_points(&mut next)?);
                }
                parts.push(rings);
            }
            let mut clips = Vec::new();
            for _ in 0..next()? {
                clips.push(read_points(&mut next)?);
            }
            hits.push(ScreenHit {
                hit: Hit {
                    layer,
                    source,
                    source_layer,
                    id,
                    generation: 0,
                },
                order: 0,
                kind,
                radius,
                min,
                max,
                parts,
                clips,
            });
        }
        if words.next().is_some() {
            return Err("malformed hit snapshot".into());
        }
        self.hits = hits;
        Ok(())
    }
}
fn read_points(next: &mut impl FnMut() -> Result<u32, String>) -> Result<Vec<Vec2>, String> {
    let mut points = Vec::new();
    for _ in 0..next()? {
        let x = f32::from_bits(next()?);
        let y = f32::from_bits(next()?);
        points.push(Vec2::new(x, y));
    }
    Ok(points)
}
fn write_points(words: &mut Vec<u32>, points: &[Vec2]) {
    words.push(points.len() as u32);
    for p in points {
        words.extend([p.x.to_bits(), p.y.to_bits()]);
    }
}
impl crate::Map {
    /// A new picking snapshot only when geometry, paint or camera changed.
    /// Send it with the frame acknowledgement, then advance its generation
    /// on every acknowledgement even when the index is retained.
    pub fn hit_snapshot(&self) -> Option<Vec<u8>> {
        if !self.rebuild_hits {
            return None;
        }
        let mut words = vec![self.hits.len() as u32];
        for h in &self.hits {
            crate::jobs::write_text(&mut words, &h.hit.layer);
            crate::jobs::write_text(&mut words, &h.hit.source);
            for value in [&h.hit.source_layer, &h.hit.id] {
                words.push(u32::from(value.is_some()));
                if let Some(value) = value {
                    crate::jobs::write_text(&mut words, value);
                }
            }
            words.push(h.kind);
            for n in [h.radius, h.min.x, h.min.y, h.max.x, h.max.y] {
                words.push(n.to_bits());
            }
            words.push(h.parts.len() as u32);
            for part in &h.parts {
                words.push(part.len() as u32);
                for ring in part {
                    write_points(&mut words, ring);
                }
            }
            words.push(h.clips.len() as u32);
            for clip in &h.clips {
                write_points(&mut words, clip);
            }
        }
        Some(words.into_iter().flat_map(u32::to_le_bytes).collect())
    }
}
