//! A cover's tile level of detail: how tiles coarsen with distance in
//! pitched views, MapLibre Native's model with its settings (the tile
//! options maplibre-native-ffi's maps take).

/// How a pitched cover picks each tile's zoom (Native's `TileLodMode`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TileLodMode {
    /// Tiles keep the covering zoom within [`TileLod::min_radius`] tiles
    /// of it around the view's centre; past that each zoom out spans
    /// twice the distance of the one before.
    #[default]
    Default,
    /// Each tile takes the zoom at which it shows about as large as the
    /// centre's tiles, by its distance from the eye and how squarely it
    /// faces it: tiles nearer than the centre may take finer zooms, up to
    /// the source's greatest.
    Distance,
}

/// A map's tile level of detail. Below the pitch threshold a cover takes
/// one zoom across the view; past it tiles coarsen with distance by
/// [`mode`](Self::mode), eased in over the [`LOD_EASE`] degrees after the
/// threshold so that detail changes continuously with the camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileLod {
    pub mode: TileLodMode,
    /// [`TileLodMode::Default`]: tiles of the covering zoom around the
    /// view's centre that keep it, at least 1.
    pub min_radius: f64,
    /// How fast detail falls with distance: above 1 tiles coarsen nearer
    /// the centre, below 1 farther out.
    pub scale: f64,
    /// Radians of pitch past which tiles coarsen with distance.
    pub pitch_threshold: f64,
    /// Zoom levels added to the covering zoom everywhere: below 0, coarser
    /// tiles throughout.
    pub zoom_shift: f64,
}

/// Degrees past the pitch threshold over which detail starts varying.
pub const LOD_EASE: f64 = 5.0;

impl Default for TileLod {
    /// MapLibre Native's defaults.
    fn default() -> Self {
        Self {
            mode: TileLodMode::Default,
            min_radius: 3.0,
            scale: 1.0,
            pitch_threshold: 60f64.to_radians(),
            zoom_shift: 0.0,
        }
    }
}

impl TileLod {
    /// These settings with values out of range brought into it: a radius of
    /// at least 1, a positive scale, and finite numbers (Native's defaults
    /// where not).
    #[must_use]
    pub fn sanitized(self) -> Self {
        let default = Self::default();
        let finite = |v: f64, d: f64| if v.is_finite() { v } else { d };
        Self {
            mode: self.mode,
            min_radius: finite(self.min_radius, default.min_radius).max(1.0),
            scale: match finite(self.scale, default.scale) {
                s if s > 0.0 => s,
                _ => default.scale,
            },
            pitch_threshold: finite(self.pitch_threshold, default.pitch_threshold),
            zoom_shift: finite(self.zoom_shift, default.zoom_shift),
        }
    }

    /// How much of the variation applies at `pitch` degrees: none up to the
    /// threshold, all of it [`LOD_EASE`] past it.
    pub(crate) fn weight(&self, pitch: f64) -> f64 {
        let weight = ((pitch - self.pitch_threshold.to_degrees()) / LOD_EASE).clamp(0.0, 1.0);
        // The threshold in radians lands an ulp off its degrees.
        if weight < 1e-9 { 0.0 } else { weight }
    }

    /// [`TileLodMode::Default`]: whether a tile `levels` zooms coarser than
    /// the covering zoom splits, its nearest point `distance` tiles of the
    /// covering zoom from the view's centre along the farther axis, at
    /// `weight`. Native's `distToSplit`: the radius, then each level twice
    /// the last, `radius + 2^levels − 2`.
    pub(crate) fn splits(&self, levels: f64, distance: f64, weight: f64) -> bool {
        distance * self.scale * weight < self.min_radius + levels.exp2() - 2.0
    }

    /// [`TileLodMode::Distance`]: zoom levels a tile adds to the covering
    /// zoom at `weight`, its nearest point at `ratio` times the centre's
    /// distance from the eye and seen at `cos` from straight down. Native
    /// splits a tile while `distance · 2^z < cos^½ · centre distance /
    /// scale · 2^covering`.
    pub(crate) fn levels(&self, ratio: f64, cos: f64, weight: f64) -> f64 {
        weight * (-ratio.max(1e-12).log2() + 0.5 * cos.max(1e-6).log2() - self.scale.log2())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rings_double_past_the_radius() {
        let lod = TileLod::default();
        // A tile one zoom out splits within 3 tiles of the centre.
        assert!(lod.splits(1.0, 2.9, 1.0));
        assert!(!lod.splits(1.0, 3.0, 1.0));
        // Two zooms out, within 5; three, within 9.
        assert!(lod.splits(2.0, 4.9, 1.0));
        assert!(!lod.splits(2.0, 5.0, 1.0));
        assert!(!lod.splits(3.0, 9.0, 1.0));
        // Up to the threshold nothing coarsens.
        assert!(lod.splits(1.0, 1e9, 0.0));
    }

    #[test]
    fn weight_eases_in_past_the_threshold() {
        let lod = TileLod::default();
        assert_eq!(lod.weight(60.0), 0.0);
        assert!((lod.weight(62.5) - 0.5).abs() < 1e-9);
        assert_eq!(lod.weight(85.0), 1.0);
    }

    #[test]
    fn distance_levels_follow_distance_and_tilt() {
        let lod = TileLod::default();
        assert_eq!(lod.levels(1.0, 1.0, 1.0), 0.0);
        assert_eq!(lod.levels(4.0, 1.0, 1.0), -2.0);
        assert_eq!(lod.levels(1.0, 0.25, 1.0), -1.0);
        assert_eq!(lod.levels(0.5, 1.0, 0.5), 0.5);
    }

    #[test]
    fn settings_out_of_range_are_brought_in() {
        let lod = TileLod {
            min_radius: 0.0,
            scale: -1.0,
            pitch_threshold: f64::NAN,
            ..TileLod::default()
        }
        .sanitized();
        assert_eq!(lod.min_radius, 1.0);
        assert_eq!(lod.scale, 1.0);
        assert_eq!(lod.pitch_threshold, TileLod::default().pitch_threshold);
    }
}
