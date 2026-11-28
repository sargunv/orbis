//! What a view draws on.

use crate::Projection;

/// What a view draws on and how: its projection's plane or, at low
/// zoom, a globe the plane's tiles wrap around, in perspective or
/// axonometric. Style zoom is always the plane's (on Web Mercator, GL JS's
/// convention on its globe too).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surface {
    pub projection: Projection,
    pub globe: Option<Globe>,
    /// MapLibre Native's axonometric projection, when set: heights stand
    /// without perspective, leaning by its skew, and padding is ignored.
    /// The ground projects as it does in perspective.
    pub axonometric: Option<Skew>,
}

/// How an axonometric view leans what stands above the ground, as
/// MapLibre Native's `xSkew` and `ySkew` do (`(0, 1)` by default: heights
/// rise straight up): a height `h` pixels long at the target's scale moves
/// the picture by `x × h` and `y × h` in clip units at the eye's distance
/// in pixels, right and up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Skew {
    pub x: f64,
    pub y: f64,
}

impl Default for Skew {
    fn default() -> Self {
        Self { x: 0.0, y: 1.0 }
    }
}

impl Surface {
    pub fn plane(projection: Projection) -> Self {
        Self {
            projection,
            globe: None,
            axonometric: None,
        }
    }

    /// `projection`'s plane wrapped around `globe` at low zoom.
    pub fn globe(projection: Projection, globe: Globe) -> Self {
        Self {
            globe: Some(globe),
            ..Self::plane(projection)
        }
    }

    /// How much of a globe a view at style zoom `zoom` is: 1 a globe, 0
    /// the plane, and a blend of the two between (GPU positions mix; the
    /// CPU uses the globe).
    pub fn globe_amount(&self, zoom: f64) -> f64 {
        self.globe.map_or(0.0, |globe| globe.amount(zoom))
    }
}

/// What a view draws on at low zoom before it hands over to the plane
/// between two zooms (adaptive projections): a sphere the
/// plane's tiles wrap around ([`sphere`](crate::sphere)), or another plane
/// they are drawn on, such as Equal Earth's for the whole world. Views mix
/// the two surfaces' clip positions as a pure function of zoom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Globe {
    /// The low-zoom surface through the first zoom, the plane from the
    /// second; `None` keeps it at every zoom (v8 `vertical-perspective`, or
    /// `transition: false`).
    pub handover: Option<[f64; 2]>,
    /// What the low-zoom surface is.
    pub low: LowSurface,
}

/// A [`Globe`]'s low-zoom surface.
#[derive(Clone, Copy, Debug, PartialEq)]
#[expect(
    clippy::large_enum_variant,
    reason = "the size of the Option<Projection> it stands for; kept Copy like Projection"
)]
pub enum LowSurface {
    /// The sphere the plane's tiles wrap around ([`sphere`](crate::sphere)).
    Sphere,
    /// Another plane the tiles are drawn on.
    Plane(Projection),
}

impl Globe {
    /// v8 `globe`: a globe through zoom 11 and Web Mercator from zoom 12,
    /// as GL JS hands over (`projection_factory.ts`).
    pub const V8: Self = Self {
        handover: Some([11.0, 12.0]),
        low: LowSurface::Sphere,
    };

    /// The sphere at every zoom (v8 `vertical-perspective`).
    pub const SPHERE: Self = Self {
        handover: None,
        low: LowSurface::Sphere,
    };

    /// The zooms a low-zoom plane hands over to Web Mercator between by
    /// default: style-spec#1865's Equal Earth, `[6, 7]`.
    pub const PLANE_HANDOVER: [f64; 2] = [6.0, 7.0];

    /// Whether the low-zoom surface is the sphere.
    pub fn is_sphere(&self) -> bool {
        matches!(self.low, LowSurface::Sphere)
    }

    /// The low-zoom plane, unless the low-zoom surface is the sphere.
    pub fn plane(&self) -> Option<Projection> {
        match self.low {
            LowSurface::Sphere => None,
            LowSurface::Plane(plane) => Some(plane),
        }
    }

    /// How much of a globe a view at `zoom` is.
    pub fn amount(&self, zoom: f64) -> f64 {
        match self.handover {
            Some([from, to]) => ((to - zoom) / (to - from)).clamp(0.0, 1.0),
            None => 1.0,
        }
    }
}
